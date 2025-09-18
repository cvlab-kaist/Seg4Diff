# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import _ignore_torch_cuda_oom
from detectron2.data.datasets import load_sem_seg
from detectron2.utils.comm import get_rank, is_main_process

from torch.amp import autocast
from einops import rearrange

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher

import pdb
import sys

import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

@META_ARCH_REGISTRY.register()
class Seg4DiffUnsup(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        train_class_json: str,
        test_class_json: str,
        backbone_multiplier: float,
        class_weight: float,
        mask_weight: float,
        dice_weight: float,
        gt_only_prompt: bool,
        clip_filtering: bool,
        with_sos_eos: bool,
        dataset_name: str,
        norm_before_merge: bool,
        background_name: str,
        output_power: float,
        norm_after_merge: bool,
        hard_assignment: bool,
        negative_bg: bool,
        temperature: float,
        w: float,
        noise_steps: int,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
        """
        super().__init__()
        self.backbone = backbone
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility

        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
        
        self.train_class_json = train_class_json
        self.test_class_json = test_class_json
        
        self.with_sos_eos = with_sos_eos
        self.clip_filtering = clip_filtering
        self.gt_only_prompt = gt_only_prompt
        self.backbone.with_sos_eos = self.with_sos_eos
        
        print("gt_only_prompt: ", self.gt_only_prompt)
        print("clip_filtering: ", self.clip_filtering)
        print("with_sos_eos: ", self.with_sos_eos)
        
        # Load class names directly from JSON files instead of relying on CATSegHead/CATSegPredictor
        with open(self.train_class_json, 'r') as _f_train:
            self.class_texts = json.load(_f_train)
        with open(self.test_class_json, 'r') as _f_test:
            self.test_class_texts = json.load(_f_test)
        if self.test_class_texts is None:
            self.test_class_texts = self.class_texts
        
        self.dataset_name = dataset_name
        self.norm_before_merge = norm_before_merge
        self.norm_after_merge = norm_after_merge
        self.hard_assignment = hard_assignment
        self.negative_bg = negative_bg
        self.output_power = output_power
        self.temperature = temperature #0.07
        self.w = w #0.3
        
        # Add missing attributes for inference
        self.generate_val_images = False
        self.visualize_attention = False
        self.visualize_val_attention = False
        self.output_dir = "./output"
        self.val_count = 0
        self.train_count = 0
        self.use_attn_mlp = False
        self.attn_mlp = None
        self.kl_threshold = 0.1
        
        if "background" in self.test_class_texts:
            background_name = background_name.replace('-', ' ')
            self.test_class_texts[self.test_class_texts.index("background")] = background_name
            self.background_name = background_name
            print("background_name: ", background_name)
        
        self.matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
        )
        
        # self.register_buffer("mask_dist", torch.zeros(154, 64, 64), False)
        self.mask_dist = torch.zeros(154, 64, 64).to('cuda') if torch.cuda.is_available() else torch.zeros(154, 64, 64)
        self.noise_steps = noise_steps
    
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg, cfg.MODEL.BACKBONE.ATTENTION_LAYERS,)
        
        return {
            "backbone": backbone,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "class_weight": cfg.MODEL.SEM_SEG_HEAD.CLASS_WEIGHT,
            "mask_weight": cfg.MODEL.SEM_SEG_HEAD.MASK_WEIGHT,
            "dice_weight": cfg.MODEL.SEM_SEG_HEAD.DICE_WEIGHT,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "gt_only_prompt": cfg.MODEL.GT_ONLY_PROMPT,
            "clip_filtering": cfg.MODEL.CLIP_FILTERING,
            "with_sos_eos": cfg.MODEL.WITH_SOS_EOS,
            "dataset_name": cfg.DATASETS.TEST[0],
            "norm_before_merge": cfg.MODEL.NORM_BEFORE_MERGE,
            "norm_after_merge": cfg.MODEL.NORM_AFTER_MERGE,
            "background_name": cfg.MODEL.BACKGROUND_NAME,
            "output_power": cfg.MODEL.OUTPUT_POWER,
            "temperature": cfg.MODEL.TEMPERATURE,
            "w": cfg.MODEL.W,
            "hard_assignment": cfg.MODEL.HARD_ASSIGNMENT,
            "negative_bg": cfg.MODEL.NEGATIVE_BG,
            "noise_steps": cfg.MODEL.NOISE_STEPS,
        }

    @property
    def device(self):
        return self.clip_pixel_mean.device
    
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        
        images = [x / 255. for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        images_resized = F.interpolate(images.tensor, size=(1024, 1024), mode='bilinear', align_corners=False,)

        # Handle targets for inference
        if "sem_seg" in batched_inputs[0]:
            if batched_inputs[0]["sem_seg"] is not None:
                targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
            else:
                targets = None
        elif "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, images, max_mask=77)
        
        prompt = ""
        noise_steps = self.noise_steps
        
        # Forward the backbone
        if torch.is_autocast_enabled():
            out = self.backbone(images_resized, prompt, noise_steps=noise_steps, mean_heads=True)
        else:
            with autocast(device_type=self.device.type, dtype=torch.float16):
                out = self.backbone(images_resized, prompt, noise_steps=noise_steps, mean_heads=True)

        with torch.no_grad():
            slices_i2t = out['attn_cache'] # [B, l, 24, 4096, 77]
            if slices_i2t.ndim == 5:
                slices_i2t = rearrange(slices_i2t, "b l h d t -> b l d (h t)") # keep the head for inference

            if self.use_attn_mlp:
                b, l, n, t = slices_i2t.shape
                _outputs = rearrange(slices_i2t, "b l n t -> (b l) 1 n t")
                if torch.is_autocast_enabled():
                    _outputs = self.attn_mlp(_outputs)
                else:
                    with autocast(device_type=self.device.type, dtype=torch.float16):
                        _outputs = self.attn_mlp(_outputs)
                _outputs = rearrange(_outputs, "(b l) 1 n t -> b l n t", b=b)
            
            slices_i2t = slices_i2t.mean(dim=1)
            slices_i2t = self.att2mask(slices_i2t, p=self.output_power)

            slices_i2t = rearrange(slices_i2t, "b (h w) l -> b l h w", h=64, w=64).squeeze(0)
            slices_i2t = self.mask_merge(self.train_count, slices_i2t, self.kl_threshold, grid=None)

            _outputs = slices_i2t.unsqueeze(0)

            B = 1
            idxs = torch.arange(_outputs.shape[1])
            _outputs_tgt = torch.zeros(B, len(idxs), _outputs.shape[-2], _outputs.shape[-1], device=self.device)
            _outputs_tgt[:, idxs] = _outputs.float()
            _outputs = _outputs_tgt
            selected_idxs = idxs
            num_classes = len(idxs)

            # Visualize attention maps for validation (unsupervised semantic segmentation)
            if self.visualize_attention and is_main_process():
                # Create validation attention visualization folder
                val_attn_folder = f"{self.output_dir}/attn_vis/{self.val_count}/validation_attention"
                if not os.path.exists(val_attn_folder):
                    os.makedirs(val_attn_folder)
                
                # Get the merged attention maps and matched GT masks
                merged_attn = _outputs[0]  # [num_classes, H, W]
                
                # Save individual attention maps and GT masks
                for j, idx in enumerate(selected_idxs):
                    class_name = self.test_class_texts[idx] if idx < len(self.test_class_texts) else f"class_{idx}"
                    
                    # Save individual attention map
                    plt.figure(figsize=(8, 8))
                    plt.imshow(merged_attn[j].cpu().numpy(), cmap='viridis')
                    plt.title(f"Attention Map: {class_name}")
                    plt.axis('off')
                    plt.colorbar()
                    plt.tight_layout()
                    plt.savefig(f"{val_attn_folder}/attention_map_{j}_{class_name}.png", dpi=150, bbox_inches='tight')
                    plt.close()
                
                # Save merged attention map (argmax)
                merged_attn_argmax = merged_attn.argmax(dim=0).cpu().numpy()
                plt.figure(figsize=(10, 10))
                plt.imshow(merged_attn_argmax, cmap='tab10')
                plt.title("Merged Attention Map (Argmax)")
                plt.axis('off')
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(f"{val_attn_folder}/merged_attention_argmax.png", dpi=150, bbox_inches='tight')
                plt.close()
                
                # Create final comparison plot
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                
                # Original image
                image = batched_inputs[0]["image"].to(self.device)
                if image.max() > 1.:
                    image = image / 255.
                image = image.cpu().numpy().transpose(1, 2, 0)
                ax[0].imshow(image)
                ax[0].set_title("Original Image")
                ax[0].axis('off')
                
                # Merged attention map
                im1 = ax[1].imshow(merged_attn_argmax, cmap='tab10')
                ax[1].set_title("Merged Attention Map")
                ax[1].axis('off')
                plt.colorbar(im1, ax=ax[1])
                
                plt.tight_layout()
                plt.savefig(f"{val_attn_folder}/final_comparison.png", dpi=150, bbox_inches='tight')
                plt.close()

            if self.visualize_val_attention:
                self.plot_attention(_outputs, [], images_resized, selected_idxs.unsqueeze(0), self.test_class_texts, save_name=batched_inputs[0]["file_name"])

            if len(idxs) == 0:
                print("No valid indices found, returning empty results.")
                return []

            if "sem_seg" not in batched_inputs[0]:  
                mask_pred_results = torch.zeros(B, len(idxs), _outputs.shape[-2], _outputs.shape[-1], device=self.device)
            else:
                mask_pred_results = torch.zeros(B, num_classes, _outputs.shape[-2], _outputs.shape[-1], device=self.device)
            mask_pred_results[:, selected_idxs] = _outputs.float()
            mask_cls_results = torch.ones(B, num_classes, device=self.device)
        
            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])

                mask_pred_result = sem_seg_postprocess(
                    mask_pred_result, image_size, height, width
                )

                processed_results.append({"sem_seg": mask_pred_result})
        
        self.generate_val_images = False

        return processed_results

    def att2mask(self, att, p=1.0):
        # att: [HW, C]
        att -= att.min(dim=0, keepdim=True)[0]
        att /= att.max(dim=0, keepdim=True)[0] + 1e-6
        
        return att ** p

    def visualize_t2t(self, t2t, classnames=None, dir="vis/text/", name="attn"):
        plt.clf()
        plt.imshow(t2t.cpu().numpy())
        plt.savefig(dir + name + ".png")

    def visualize_cross(self, slices_t2i, slices_i2t, t2t=None, classnames=None, dir="logsumexp/", name="attn"):
        num_classes = slices_t2i.shape[0]
        
        plt.clf()
        slices_t2i = slices_t2i.reshape(-1, 64, 64).cpu().numpy()
        slices_i2t = slices_i2t.reshape(64, 64, -1).permute(2, 0, 1).cpu().numpy()
        fig, axs = plt.subplots(2, num_classes, figsize=(2 * num_classes, 4))
        
        if num_classes == 1:
            axs = np.expand_dims(axs, 1)
        
        for a in axs.flatten():
            a.set_xticklabels([])
            a.set_yticklabels([])
        
        axs[0, 0].set_ylabel("Image to Text")
        axs[1, 0].set_ylabel("Text to Image")
        
        if classnames is not None:
            for i, n in enumerate(classnames):
                axs[0, i].set_title(n)
        
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
        
        for i in range(num_classes):
            axs[0, i].imshow(slices_i2t[i])
            axs[1, i].imshow(slices_t2i[i])
        
        fig.savefig(dir + name + ".png")
        
    def prepare_targets(self, targets, images):
        h, w = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h, w), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def get_gt_indices(self, batched_inputs):
        if self.eval_idxs is None:
            if "sem_seg" in batched_inputs[0].keys():
                targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
                idxs = targets.flatten().bincount()[:171].nonzero().squeeze(-1)
            else:
                metadata = MetadataCatalog.get(self.dataset_name)
                targets = load_sem_seg(gt_root=metadata.seg_seg_root, image_root=metadata.image_root)
                sem_seg_file_name = None
                for t in targets:
                    if t['file_name'] == batched_inputs[0]['image_name']:
                        sem_seg_file_name = t['sem_seg_file_name']
                        break
                
                if sem_seg_file_name is not None:
                    import numpy as np
                    from PIL import Image
                    
                    sem_seg = Image.open(sem_seg_file_name)
                    targets = torch.tensor(np.array(sem_seg), dtype=torch.long)
                    idxs = targets.flatten().bincount()[:171].nonzero().squeeze(-1).to(self.device)
        else:
            if "sem_seg" in batched_inputs[0].keys():
                # load classnames from json
                image_name = batched_inputs[0]['file_name'].split('/')[-1].split('.')[0]
                idxs = torch.tensor(self.eval_idxs[image_name], device=self.device)
            else:
                image_name = batched_inputs[0]['image_name'].split('/')[-1].split('.')[0]
                try:
                    idxs = torch.tensor(self.eval_idxs[image_name], device=self.device)
                except:
                    return None
        return idxs

    def att2mask(self, att, p=1.0):
        # att: [B, N, HW, C]
        att = att - att.min(dim=2, keepdim=True)[0]
        att = att / (att.max(dim=2, keepdim=True)[0] + 1e-6)
        
        return att ** p

    def KL(self, x: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute symmetric KL divergence between x and Y over spatial dims.

        Args:
            x: Tensor of shape [*, H, W]
            Y: Tensor of same shape as x
        Returns:
            Tensor of shape [*], the KL divergence per sample
        """
        # elementwise log and difference
        quotient = torch.log(x) - torch.log(Y)
        # sum over spatial dimensions (-2, -1), then scale by 1/2
        kl_1 = torch.sum(x * quotient, dim=(-2, -1)) / 2
        kl_2 = -torch.sum(Y * quotient, dim=(-2, -1)) / 2
        return kl_1 + kl_2

    def mask_merge(self, iter, attns, kl_threshold, grid=None):
        """
        PyTorch version of the TensorFlow mask_merge function.
        Now supports both GPU and CPU modes based on configuration.

        Args:
            iter (int): current iteration index
            attns (torch.Tensor): attention maps
            kl_threshold (list or array): thresholds for KL at each iteration
            grid (torch.Tensor, optional): index grid for iter==0
        Returns:
            torch.Tensor: merged attention maps, shape (M, H, W)
        """
        # Original GPU implementation
        # Subsequent iterations: greedy clustering
        N, H, W = attns.shape  # attns: [num_attns, H, W]
        matched = set()
        new_list = []

        flat = attns.reshape(N, -1)        # [N, H*W]
        probs = torch.softmax(flat, dim=-1)
        attns = probs.view_as(attns)    # [N, H, W], sums to 1

        for i in range(N):
            if i in matched:
                continue
            matched.add(i)
            anchor = attns[i].unsqueeze(0)  # [1, H, W]
            anchor = anchor.expand(N, -1, -1)  # [N, H, W]

            # Compute KL to all and threshold
            kl_vals = self.KL(anchor, attns)  # expected shape [num_attns]
            mask = (kl_vals < kl_threshold).cpu()

            if mask.sum() > 0:
                matched_idx = torch.nonzero(mask.view(-1), as_tuple=False).squeeze().tolist()
                for idx in (matched_idx if isinstance(matched_idx, list) else [matched_idx]):
                    matched.add(idx)

                # Average grouped maps
                group = attns[mask]
                aggregated = group.mean(dim=0)  # [H, W]
                new_list.append(aggregated)

        if new_list:
            new_attns = torch.stack(new_list, dim=0)  # [M, H, W]
        else:
            new_attns = torch.empty((0, H, W), dtype=attns.dtype)
        return new_attns

    @torch.no_grad()
    def generate_image(self, cond_prompts, learnable_token=None):
        # Placeholder implementation - can be filled with actual logic if needed
        pass

    @torch.no_grad()
    def plot_attention(self, pred, gt, img, indices, classnames, save_name="attn_vis.png"):
        """
            pred: predicted attention maps, shape [B, N, H, W]
            gt: ground truth masks, shape [B, N, H, W]
            indices: list of indices for each batch, each item is a list of indices for the attention maps
            save_path: directory to save the attention maps and ground truth masks
        """
        # Placeholder implementation - can be filled with actual logic if needed
        pass

class ForkedPdb(pdb.Pdb):
    """
    PDB Subclass for debugging multi-processed code
    Suggested in: https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin