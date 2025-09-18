# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
import os

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.data.datasets import load_sem_seg

from torch.amp import autocast
from einops import rearrange

import pdb
import sys
import matplotlib.pyplot as plt

@META_ARCH_REGISTRY.register()
class Seg4DiffOVSS(nn.Module):
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
        gt_only_prompt: bool,
        dataset_name: str,
        norm_before_merge: bool,
        norm_after_merge: bool,
        background_name: str,
        output_power: float,
        temperature: float,
        w: float,
        noise_steps: int,
        output_dir: str,
        mask_loss_weight: float,
        use_attn_mlp: bool,
        visualize_attention: bool = False,
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
        
        self.gt_only_prompt = gt_only_prompt
        
        # Load class names directly from JSON files instead of relying on CATSegHead/CATSegPredictor
        import json
        with open(train_class_json, 'r') as _f_train:
            self.class_texts = json.load(_f_train)
        with open(test_class_json, 'r') as _f_test:
            self.test_class_texts = json.load(_f_test)
        if self.test_class_texts is None:
            self.test_class_texts = self.class_texts
        
        self.use_attn_mlp = use_attn_mlp
        if use_attn_mlp:
            self.attn_mlp = AttentionScoreLayer()

        self.dataset_name = dataset_name
        self.norm_before_merge = norm_before_merge
        self.norm_after_merge = norm_after_merge
        self.output_power = output_power
        self.temperature = temperature #0.07
        self.w = w #0.3

        self.noise_steps = noise_steps
        
        if "background" in self.test_class_texts:
            background_name = background_name.replace('-', ' ')
            self.test_class_texts[self.test_class_texts.index("background")] = background_name
            self.background_name = background_name
            print("background_name: ", background_name)
        else:
            self.background_name = None

        self.output_dir = output_dir

        self.visualize_attention = visualize_attention
        self.backbone_name = self.backbone.__class__.__name__
        
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg, cfg.MODEL.BACKBONE.ATTENTION_LAYERS,)
        
        return {
            "backbone": backbone,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "gt_only_prompt": cfg.MODEL.GT_ONLY_PROMPT,
            "dataset_name": cfg.DATASETS.TEST[0],
            "norm_before_merge": cfg.MODEL.NORM_BEFORE_MERGE,
            "norm_after_merge": cfg.MODEL.NORM_AFTER_MERGE,
            "background_name": cfg.MODEL.BACKGROUND_NAME,
            "output_power": cfg.MODEL.OUTPUT_POWER,
            "temperature": cfg.MODEL.TEMPERATURE,
            "w": cfg.MODEL.W,
            "noise_steps": cfg.MODEL.NOISE_STEPS,
            "output_dir": cfg.OUTPUT_DIR,
            "mask_loss_weight": cfg.MODEL.MASK_LOSS_WEIGHT,
            "use_attn_mlp": cfg.MODEL.USE_ATTN_MLP,
            "visualize_attention": cfg.MODEL.VISUALIZE_ATTENTION,
        }

    @property
    def device(self):
        return self.clip_pixel_mean.device
    
    def forward(self, batched_inputs):
        """
        Args:
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        
        images = [x / 255. for x in images]
        images = ImageList.from_tensors(images)
        images_resized = F.interpolate(images.tensor, size=(1024, 1024), mode='bilinear', align_corners=False,)

        idxs = self.get_gt_indices(batched_inputs)
        
        prompt = None
        num_classes = len(self.test_class_texts)        

        assert not self.training

        selected_idxs = idxs
        # if "background" in self.test_class_texts:
        #     selected_idxs = [i for i in selected_idxs if self.test_class_texts[i] != "background"]
        
        # Ensure indices are Python ints for list indexing
        classnames= [self.test_class_texts[int(i)].split("-")[0] for i in selected_idxs]
        prompt = " ".join(classnames)
        
        if self.backbone_name == "SD3Backbone" or self.backbone_name == "SD35Backbone":
            num_tokens = [len(self.backbone.pipe.tokenizer(t)["input_ids"]) - 2 for t in classnames]
            tokens = self.backbone.pipe.tokenizer(prompt)["input_ids"]
            tokens_text = self.backbone.pipe.tokenizer.batch_decode(tokens)
            tokens = tokens[1:-1]
        else:
            num_tokens = [len(self.backbone.pipe.tokenizer_2(t)["input_ids"]) - 1 for t in classnames]
            tokens = self.backbone.pipe.tokenizer_2(prompt)["input_ids"]
            tokens_text = self.backbone.pipe.tokenizer_2.batch_decode(tokens)
            tokens = tokens[:-1]

        full_indices = torch.arange(4096+154)
        
        if torch.is_autocast_enabled():
            out = self.backbone(images_resized, prompt, noise_steps=self.noise_steps, get_class_logits=True, mean_heads=True, selected_idxs=selected_idxs, indices=full_indices)
        else:
            with autocast(device_type=self.device.type, dtype=torch.float16):
                out = self.backbone(images_resized, prompt, noise_steps=self.noise_steps, get_class_logits=True, mean_heads=True, selected_idxs=selected_idxs, indices=full_indices)

        # semseg eval
        if self.backbone_name == "SD3Backbone" or self.backbone_name == "SD35Backbone":
            slices_i2t = out['attn_cache']
            # slices_t2i = feat[4096:, :4096]
            # slices_i2t = feat[:4096, 4096:]
        elif self.backbone_name == "FluxBackbone":
            slices_i2t = out['attn_cache'][:, :, -4096:, :-4096]
        
        text_len = slices_i2t.shape[-1]
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
            slices_i2t = rearrange(_outputs, "(b l) 1 n t -> b l n t", b=b)

        slices_i2t = slices_i2t[0].mean(dim=0)  # average over batch and layer

        if self.visualize_attention: # visualize cross attention maps
            self.visualize_cross(slices_i2t, classnames=tokens_text, name=f"attn_{batched_inputs[0]['file_name'].split('/')[-1].split('.')[0]}")

        num_classes = len(self.test_class_texts)

        merged_fg = torch.zeros(slices_i2t.shape[0], len(num_tokens), device=self.device)
        st = 1 if self.backbone_name == "SD3Backbone" or self.backbone_name == "SD35Backbone" else 0

        if self.norm_before_merge:
            slices_i2t -= slices_i2t.min(dim=0, keepdim=True)[0]
            slices_i2t /= slices_i2t.max(dim=0, keepdim=True)[0] + 1e-6

        for i, n in enumerate(num_tokens):
            merged_fg[:, i] = slices_i2t[:, st:st+n].mean(dim=-1)
            st += n

        if self.norm_after_merge:
            merged_fg -= merged_fg.min(dim=0, keepdim=True)[0]
            merged_fg /= merged_fg.max(dim=0, keepdim=True)[0] + 1e-6
        
        _outputs = merged_fg.T.reshape(-1, 64, 64).unsqueeze(0)
        
        outputs = torch.zeros(1, num_classes, _outputs.shape[-2], _outputs.shape[-1], device=self.device)
        outputs[:, selected_idxs] = _outputs.float()
        
        if self.visualize_attention:
            idxs = self.get_gt_indices(batched_inputs)
            if len(idxs) == 0:
                print("No valid indices found, returning empty results.")
                return []
            target_mask = torch.stack([batched_inputs[0]["sem_seg"].to(self.device) == x for x in idxs], dim=0)
            if target_mask.ndim == 4:
                target_mask = target_mask.squeeze(1)
            _targets = [{
                "labels": idxs,
                "masks": target_mask,
            }]

            self.plot_attention_with_gt(_outputs, _targets, images_resized, idxs.unsqueeze(0), self.test_class_texts, batched_inputs[0]['file_name'])

        image_size = images.image_sizes[0]
        height = batched_inputs[0].get("height", image_size[0])
        width = batched_inputs[0].get("width", image_size[1])

        output = sem_seg_postprocess(outputs[0], image_size, height, width)
        processed_results = [{'sem_seg': output}]
        return processed_results

    def visualize_cross(self, slices_i2t, t2t=None, classnames=None, name="attn"):
        num_classes = slices_i2t.shape[0]
        
        plt.clf()
        slices_i2t = slices_i2t.reshape(64, 64, -1).permute(2, 0, 1).cpu().numpy()
        
        # Create 11 rows and 7 columns grid
        rows, cols = 7, 11
        fig, axs = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
        
        # Ensure axs is always 2D
        if rows == 1:
            axs = axs.reshape(1, -1)
        if cols == 1:
            axs = axs.reshape(-1, 1)
        
        # Remove ticks and labels from all subplots
        for i in range(rows):
            for j in range(cols):
                axs[i, j].set_xticklabels([])
                axs[i, j].set_yticklabels([])
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
        
        # Plot the attention maps
        for i in range(min(num_classes, rows * cols)):
            row = i // cols
            col = i % cols
            axs[row, col].imshow(slices_i2t[i])
            
            # Add class names if provided
            if classnames is not None and i < len(classnames):
                axs[row, col].set_title(classnames[i], fontsize=8)
        
        # Hide unused subplots
        for i in range(num_classes, rows * cols):
            row = i // cols
            col = i % cols
            axs[row, col].set_visible(False)
        
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.tight_layout()
        
        if not os.path.exists(f"{self.output_dir}/cross_vis/"):
            os.makedirs(f"{self.output_dir}/cross_vis/")

        fig.savefig(f"{self.output_dir}/cross_vis/{name}.png")
    
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
        return idxs

    @torch.no_grad()
    def plot_attention_with_gt(self, pred, gt, img, indices, classnames, save_name="attn_vis.png"):
        """
            pred: predicted attention maps, shape [B, N, H, W]
            gt: ground truth masks, shape [B, N, H, W]
            indices: list of indices for each batch, each item is a list of indices for the attention maps
            save_path: directory to save the attention maps and ground truth masks
        """
        save_name = save_name.split('/')[-1].replace('.jpg', '')
        
        # visualize the matched attention map and gt masks
        if not os.path.exists(f"{self.output_dir}/attn_vis_with_gt"):
            os.makedirs(f"{self.output_dir}/attn_vis_with_gt")

        for bs in range(pred.shape[0]):

            attn_map = pred[bs].clone().detach()
            gt_mask = gt[bs]['masks'].clone().detach()
            gt_mask = F.interpolate(gt_mask[None].float(), size=(64, 64), mode='nearest')[0]
            attn_map = attn_map.cpu().numpy()
            fig, ax = plt.subplots(2, indices[bs].shape[0], figsize=(5 * indices[bs].shape[0], 12))

            if ax.ndim == 1:
                ax = ax[:, None]

            for j, idx in enumerate(indices[bs]):
                ax[0, j].imshow(attn_map[j])
                ax[0, j].set_title(f"Attention Map {j}: {classnames[idx]}")
                ax[1, j].imshow(gt_mask[j].cpu().numpy())
                ax[1, j].set_title(f"GT Mask {j}")
                ax[0, j].axis('off')
                ax[1, j].axis('off')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/attn_vis_with_gt/{save_name}.png")

            img = img[bs].cpu().numpy().transpose(1, 2, 0)
            attn_map_argmax = pred[bs].argmax(dim=0).cpu().numpy()
            gt_mask_argmax = gt_mask.argmax(dim=0).cpu().numpy()

            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(img)
            ax[0].set_title("Image")
            ax[0].axis('off')
            ax[1].imshow(attn_map_argmax)
            ax[1].set_title("Attention Map Argmax")
            ax[1].axis('off')
            ax[2].imshow(gt_mask_argmax)
            ax[2].set_title("GT Mask Argmax")
            ax[2].axis('off')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/attn_vis_with_gt/{save_name}_argmax.png")

            plt.close(fig)


class AttentionScoreLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_score_layer = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x: [B, 1, H, W]
        x = self.attn_score_layer(x)
        return x


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