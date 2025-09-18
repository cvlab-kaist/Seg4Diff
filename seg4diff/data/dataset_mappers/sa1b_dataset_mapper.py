# ------------------------------------------------------------------------
# Semantic SAM
# Copyright (c) MicroSoft, Inc. and its affiliates.
# Modified by Xueyan Zou and Jianwei Yang.
# ------------------------------------------------------------------------
# Modified by Heeseong Shin from https://github.com/UX-Decoder/Semantic-SAM/blob/main/datasets/dataset_mappers/sam_baseline_dataset_mapper_json.py

import copy
import json
import logging
import os
import numpy as np
import torch
import random

from detectron2.structures import Instances, Boxes, PolygonMasks,BoxMode
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from pycocotools import mask as coco_mask
from detectron2.config import configurable

from copy import deepcopy
import base64
from io import BytesIO
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms as tr

def img_from_base64(imagestring):
    jpgbytestring = base64.b64decode(imagestring)
    image = BytesIO(jpgbytestring)
    image = Image.open(image).convert("RGB")
    return image

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    if not is_train:
        return T.ResizeShortestEdge(
                short_edge_length=800,
                max_size=1333,
            ),
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN,
            cfg.INPUT.MAX_SIZE_TRAIN,
            cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    if cfg.INPUT.COLOR_AUG_SSD:
        augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
    augs.append(T.RandomFlip())

    return augs
    #return augmentation


def resize(img, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask

def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img

def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    img_w, img_h = img_size
    mask = torch.zeros(img_h, img_w)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_h * img_w
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_w)
        y = np.random.randint(0, img_h)

        if x + cutmix_w <= img_w and y + cutmix_h <= img_h:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask

__all__ = ["SamBaselineDatasetMapperJSON"]

class SamBaselineDatasetMapperJSON:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """
    @configurable
    def __init__(
        self,
        is_train=True,
        strong_aug=False,
        *,
        augmentation,
        image_format,
        caption_dir=None,
    ):
        self.augmentation = augmentation
        logging.getLogger(__name__).info(
            "[SA1B_Dataset_Mapper] Full TransformGens used in training: {}".format(str(self.augmentation))
        )
        self._root = os.getenv("SAM_DATASETS", "datasets")

        self.strong_aug = strong_aug
        self.img_format = image_format
        self.is_train = is_train

        if caption_dir is not None:
            caption_dir = os.path.join("datasets", "captions", caption_dir)
            self.caption = json.load(open(caption_dir, 'r'))
            self.caption = {key: value for key, value in self.caption.items()}
        else:
            self.caption = None

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "strong_aug": cfg.INPUT.STRONG_AUG,
            "augmentation": tfm_gens,
            "image_format": cfg['INPUT']['FORMAT'],
            "caption_dir": cfg.DATASETS.CAPTION_DIR,
        }
        return ret
    
    def read_img(self, row):
        img = img_from_base64(row[-1])
        return img

    def read_json(selfself, row):
        anno=json.loads(row[1])
        return anno

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(self._root + dataset_dict["img_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        ori_shape = image.shape[:2]

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        
        anns = json.load(open(self._root + dataset_dict["ann_name"], 'r'))['annotations'] 
        dataset_dict['annotations'] = anns
        
        for anno in dataset_dict['annotations']:
            anno["bbox_mode"] = BoxMode.XYWH_ABS
            anno["category_id"] = 0

        utils.check_image_size(dataset_dict, image)

        padding_mask = np.ones(image.shape[:2])
        
        image, transforms = T.apply_transform_gens(self.augmentation, image)

        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)
        image_shape = image.shape[:2]

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        def copy_segm(obj):
            obj['segmentation'] = obj['segmentation'].copy()
            return obj

        def cutout(image, mask_size, p, cutout_inside, mask_color=(0, 0, 0)):
            image = np.asarray(image).copy()

            mask_size_half_x = mask_size[0] // 2
            mask_size_half_y = mask_size[1] // 2

            offset = [1 if sz % 2 == 0 else 0 for sz in mask_size]

            if np.random.random() > p:
                return image

            h, w = image.shape[:2]

            if cutout_inside:
                cxmin, cxmax = min(mask_size_half_x, w + offset[0] - mask_size_half_x), max(mask_size_half_x, w + offset[0] - mask_size_half_x)
                cymin, cymax = min(mask_size_half_y, h + offset[1] - mask_size_half_y), max(mask_size_half_y, h + offset[1] - mask_size_half_y)
            else:
                cxmin, cxmax = 0, w + offset[0]
                cymin, cymax = 0, h + offset[1]

            
            cx = np.random.randint(cxmin, cxmax + 1)
            cy = np.random.randint(cymin, cymax + 1)
            xmin = cx - mask_size_half_x
            ymin = cy - mask_size_half_y
            xmax = xmin + mask_size[0]
            ymax = ymin + mask_size[1]
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)
            image[ymin:ymax, xmin:xmax] = mask_color

            mask = torch.zeros(image.shape[:2])
            mask[ymin:ymax, xmin:xmax] = 1
            
            return torch.Tensor(image), (xmin, ymin, xmax, ymax), mask   

        mask = None
        if self.strong_aug:
            img = dataset_dict["image"]
            img_s1 = deepcopy(img)
            # tensor to PIL
            img_s1 = Image.fromarray(img_s1.permute(1, 2, 0).cpu().numpy())

            # color transform
            if random.random() < 0.8:
                img_s1 = tr.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
            img_s1 = tr.RandomGrayscale(p=0.2)(img_s1)
            img_s1 = blur(img_s1, p=0.5)
            
            mask_size = (int(image_shape[0] * np.random.uniform(0.1, 0.5))), int(image_shape[1] * np.random.uniform(0.1, 0.5))
            img_s1, box, mask = cutout(img_s1, mask_size, 1.0, True)
            dataset_dict["image_strong"] = img_s1.permute(2, 0, 1)

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)
                
            annos = [
                copy_segm(utils.transform_instance_annotations(obj, transforms, image_shape))
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            annos = sorted(annos, key=lambda x: x['segmentation'].sum(), reverse=True)
            
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            instances = utils.annotations_to_instances(annos, image_shape,mask_format='bitmask')
            if mask is not None:
                instances.gt_masks.tensor = instances.gt_masks.tensor & ~(mask.bool())

            if not instances.has('gt_masks'): 
                instances.gt_masks = PolygonMasks([])  # for negative examples
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            instances = utils.filter_empty_instances(instances)
            # Generate masks from polygon
            h, w = instances.image_size

            dataset_dict["instances"] = instances

        if self.caption is not None:
            image_id = dataset_dict["img_name"].split("/")[-1].split(".")[0]
            if image_id in self.caption:
                dataset_dict["caption"] = self.caption[image_id]
            else:
                print(f"Image id {image_id} not in caption file, using empty caption.")
                dataset_dict["caption"] = ""


        return dataset_dict