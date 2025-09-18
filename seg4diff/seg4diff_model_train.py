# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import os
from einops import rearrange

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import _ignore_torch_cuda_oom
from detectron2.data.datasets import load_sem_seg
from detectron2.utils.comm import get_rank, is_main_process


from torch.utils.tensorboard import SummaryWriter

from torch.amp import autocast
from einops import rearrange
from diffusers import StableDiffusion3Pipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher

# import BitMasks
from detectron2.structures.masks import BitMasks

import matplotlib.pyplot as plt
import math
import pdb
import sys

from sklearn.cluster import KMeans
import torch.nn.functional as F

@META_ARCH_REGISTRY.register()
class Seg4DiffTrainer(nn.Module):
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
        mean_heads: bool,
        class_weight: float,
        mask_weight: float,
        dice_weight: float,
        gt_only_prompt: bool,
        train_dataset_name: str,
        dataset_name: str,
        norm_before_merge: bool,
        norm_after_merge: bool,
        background_name: str,
        output_power: float,
        temperature: float,
        w: float,
        eval_mode: str,
        noise_steps: int,
        output_dir: str,
        mask_loss_weight: float,
        dice_loss_weight: float,
        fm_weight: float,
        short_caption: bool,
        caption_dropout: float,
        max_mask: int,
        kl_threshold: float,
        use_attn_mlp: bool,
        cost_thresh: float,
        visualize_val_attention: bool = False,
        generate_val_images: bool = False,
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

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
        
        self.train_class_json = train_class_json
        self.test_class_json = test_class_json
        import json
        with open(self.train_class_json, 'r') as _f_train:
            self.train_class_texts = json.load(_f_train)
        with open(self.test_class_json, 'r') as _f_test:
            self.test_class_texts = json.load(_f_test)
        if self.test_class_texts is None:
            self.test_class_texts = self.train_class_texts

        self.use_attn_mlp = use_attn_mlp
        if use_attn_mlp:
            self.attn_mlp = AttentionScoreLayer()
            # self.attn_mlp = DeepAttentionScoreLayer()
            for name, params in self.attn_mlp.named_parameters():
                params.requires_grad = True

        finetune_backbone = backbone_multiplier > 0.
        self.backbone_multiplier = backbone_multiplier
        if finetune_backbone:
            self.backbone.transformer.train()
            for name, params in self.backbone.transformer.named_parameters():
                if self.backbone.use_lora:
                    params.requires_grad = "lora" in name
                else:
                    params.requires_grad = True
            
            if fm_weight <= 0.:
                for name, params in self.backbone.transformer.named_parameters():
                    if "lora" in name and int(name.split('.')[1]) >= 9:
                        params.requires_grad = False
        
        no_object_weight = 0.1
        self.matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        self.criterion = SetCriterion(
            len(self.train_class_texts),
            matcher=self.matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=["masks"],
            cost_thresh=cost_thresh,
        )
        
        self.mean_heads = mean_heads
        self.mask_loss_weight = mask_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.fm_weight = fm_weight
        self.short_caption = short_caption
        self.caption_dropout = caption_dropout
        self.max_mask = max_mask
        self.kl_threshold = kl_threshold

        self.norm_before_merge = norm_before_merge
        self.norm_after_merge = norm_after_merge

        self.output_power = output_power
        self.temperature = temperature
        self.w = w

        self.noise_steps = noise_steps
        self.backbone_kwargs = {
            "mean_heads": self.mean_heads
        }

        self.output_dir = output_dir
        self.logger = SummaryWriter(log_dir=output_dir + "/logs")
        self.train_dataset_name = train_dataset_name
        self.dataset_name = dataset_name
        self.gt_only_prompt = gt_only_prompt

        self.visualize_val_attention = visualize_val_attention

        self.sem_seg_head = None 

        classname_ids = self.backbone.pipe.tokenizer(self.train_class_texts)["input_ids"]
        self.classname_ids = [torch.tensor(ids[1:-1]) for ids in classname_ids]  # remove <s> and </s>

        self.generate_val_images = generate_val_images
        self.cond_prompts = [
            "The man at bat readies to swing at the pitch while the umpire looks on",
            "A large bus sitting next to a very tall building", 
            "A cat holding a sign that says hello world",
            "A cat wearing a glasses typing on a computer",
        ]
        self.iter = 0


    @classmethod
    def from_config(cls, cfg):
        # backbone = build_backbone(cfg, [i for i in range(24)]) # [23])
        # backbone = build_backbone(cfg, [7,8,9]) # [23])
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
            "mean_heads": cfg.MODEL.SEM_SEG_HEAD.MEAN_HEADS,
            "class_weight": cfg.MODEL.SEM_SEG_HEAD.CLASS_WEIGHT,
            "mask_weight": cfg.MODEL.SEM_SEG_HEAD.MASK_WEIGHT,
            "dice_weight": cfg.MODEL.SEM_SEG_HEAD.DICE_WEIGHT,
            "cost_thresh": cfg.MODEL.SEM_SEG_HEAD.COST_THRESH,
            "gt_only_prompt": cfg.MODEL.GT_ONLY_PROMPT,
            "train_dataset_name": cfg.DATASETS.TRAIN[0],
            "dataset_name": cfg.DATASETS.TEST[0],
            "norm_before_merge": cfg.MODEL.NORM_BEFORE_MERGE,
            "norm_after_merge": cfg.MODEL.NORM_AFTER_MERGE,
            "background_name": cfg.MODEL.BACKGROUND_NAME,
            "output_power": cfg.MODEL.OUTPUT_POWER,
            "temperature": cfg.MODEL.TEMPERATURE,
            "w": cfg.MODEL.W,
            "eval_mode": cfg.MODEL.EVAL_MODE,
            "noise_steps": cfg.MODEL.NOISE_STEPS,
            "output_dir": cfg.OUTPUT_DIR,
            "mask_loss_weight": cfg.MODEL.MASK_LOSS_WEIGHT,
            "dice_loss_weight": cfg.MODEL.DICE_LOSS_WEIGHT,
            "fm_weight": cfg.MODEL.FM_WEIGHT,
            "short_caption": cfg.MODEL.SHORT_CAPTION,
            "caption_dropout": cfg.MODEL.CAPTION_DROPOUT,
            "max_mask": cfg.MODEL.MAX_MASK,
            "kl_threshold": cfg.MODEL.KL_THRESHOLD,
            "use_attn_mlp": cfg.MODEL.USE_ATTN_MLP,
            "generate_val_images": cfg.MODEL.GENERATE_VAL_IMAGES,
            "visualize_val_attention": cfg.MODEL.VISUALIZE_VAL_ATTENTION,
        }

    @property
    def device(self):
        return self.pixel_mean.device

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

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, images, max_mask=self.max_mask, sort="sam" in self.train_dataset_name)

            ####################
            # prepare the prompt
            ####################

            prompt = ""
            if "coco" in self.train_dataset_name:
                indices, _ = self.get_gt_indices(batched_inputs)
            else:
                indices = None

            if "caption" in batched_inputs[0]:
                prompt = [x["caption"] for x in batched_inputs]
                if self.short_caption:  
                    prompt = [p.split(".")[0] + "." + p.split(".")[1] + "." for p in prompt] # make prompt short (up to 2 sentences)
            
            if self.gt_only_prompt:
                prompt = [" ".join([self.train_class_texts[i] for i in idx]) for idx in indices] if indices is not None else prompt

            if torch.rand(1) < self.caption_dropout:
                print("Caption dropout")
                prompt = ""

            if indices is not None:
                prompt_ids = self.backbone.pipe.tokenizer([p.lower() for p in prompt])["input_ids"]
                prompt_ids = [torch.tensor(ids)[:self.backbone.pipe.tokenizer_max_length] for ids in prompt_ids]

            noise_steps = None
        else:
            if self.gt_only_prompt:
                indices, _ = self.get_gt_indices(batched_inputs) if "coco" in self.train_dataset_name else None
                prompt = [" ".join([self.train_class_texts[i] for i in idx]) for idx in indices] if indices is not None else prompt
            else:
                prompt = ""
            noise_steps = self.noise_steps
        
        ####################
        # forward the backbone
        ####################

        if torch.is_autocast_enabled():
            out = self.backbone(images_resized, prompt, noise_steps=noise_steps, **self.backbone_kwargs)
        else:
            with autocast(device_type=self.device.type, dtype=torch.float16):
                out = self.backbone(images_resized, prompt, noise_steps=noise_steps, **self.backbone_kwargs)

        if self.training:
            num_classes = len(self.train_class_texts)

            losses = {}

            ####################
            # prepare the attention map
            ####################

            slices_i2t = out['attn_cache'] # [B, 24, 4096, 77]
            # slices_i2t = self.att2mask(slices_i2t, p=self.output_power)
            slices_i2t = slices_i2t.mean(dim=1)
            if slices_i2t.ndim == 4:
                slices_i2t = rearrange(slices_i2t, "b l n t-> b n (l t)")

            # torch.Size([1, 1, 4096, 77])
            _outputs = rearrange(slices_i2t, "b (h w) l -> b l h w", h=64, w=64)
            if self.use_attn_mlp:
                b, l, h, w = _outputs.shape
                _outputs = rearrange(_outputs, "b l h w -> (b l) 1 h w")
                if torch.is_autocast_enabled():
                    residual = self.attn_mlp(_outputs)
                else:
                    with autocast(device_type=self.device.type, dtype=torch.float16):
                        residual = self.attn_mlp(_outputs)
                _outputs = residual + _outputs  # add residual connection
                _outputs = rearrange(_outputs, "(b l) 1 h w -> b l h w", b=b)


            if indices is not None:
                
                ####################
                # prepare targets for GT indices
                ####################
                
                preds = []
                valid_targets = []
                for i, (idx, p) in enumerate(zip(indices, prompt_ids)):
                    if len(idx) == 0:
                        continue
                        
                    target_ids = [self.classname_ids[x] for x in idx]
                    loc = [torch.isin(p, x).nonzero().squeeze(-1) for x in target_ids]
                    valid_mask = [isinstance(l, torch.Tensor) and l.numel() > 0 for l in loc]

                    if not any(valid_mask):
                        continue
                    else:
                        pred = [_outputs[i, l].mean(dim=0) for l in loc if l.numel() > 0]
                        preds.extend(pred)

                        
                        valid_targets.extend([
                            targets[i]['masks'][j]
                            for j in range(len(idx))
                            if valid_mask[j]
                        ])

                if len(preds) == 0:
                    print("No valid preds, skipping.")
                    losses = {}
                    losses['loss_mask'] = torch.tensor(0., device=self.device)
                    losses['loss_dice'] = torch.tensor(0., device=self.device)
                else:
                    preds = torch.stack(preds, dim=0).to(self.device)
                    _outputs = preds.unsqueeze(0)  # [1, N, H, W]

                    valid_targets = torch.stack(valid_targets, dim=0).to(self.device)

                    # count None in preds
                    print(f"Number of valid masks in preds: {preds.shape[0]}/{valid_targets.shape[0]}")

                    indices = [(
                        torch.arange(preds.shape[0], device=self.device),
                        torch.arange(valid_targets.shape[0], device=self.device)    
                    )]

                    outputs = {
                        "pred_logits": torch.ones(1, preds.shape[0], 1, device=self.device),
                        "pred_masks": _outputs,  # [1, N, H, W]
                    }

                    targets = [{
                        "masks": valid_targets,  # shape [ N, H, W]
                    }]

                    num_masks = preds.shape[0] if preds is not None else 0
                    
                    # Compute all the requested losses
                    losses = {}
                    for loss in self.criterion.losses:
                        losses.update(self.criterion.get_loss(loss, outputs, targets, indices, num_masks))

                    losses['loss_mask'] = losses['loss_mask'] * self.mask_loss_weight
                    losses['loss_dice'] = losses['loss_dice'] * self.dice_loss_weight

            else:

                ####################
                # prepare targets with hungarian matching
                ####################
                
                outputs = {
                    "pred_logits": torch.ones(_outputs.shape[0], _outputs.shape[1], 1, device=self.device),
                    "pred_masks": _outputs
                }

                loss_mask, indices = self.criterion(outputs, targets)
                losses['loss_mask'] = loss_mask['loss_mask'] * self.mask_loss_weight
                losses['loss_dice'] = loss_mask['loss_dice'] * self.dice_loss_weight

            if self.fm_weight > 0.:
                # flow matching loss
                noise = out['noise']
                model_input = out['model_input']
                target = noise - model_input
                
                model_pred = out['model_pred']
                
                loss_fm = torch.mean(
                        ((model_pred - target) ** 2).reshape(target.shape[0], -1),
                        1,
                    ).mean()
                losses["loss_fm"] = loss_fm * self.fm_weight

            self.iter += 1
            self.generate_val_images = True
            return losses
        else:

            if self.generate_val_images and is_main_process():
                self.generate_image(self.cond_prompts)
                self.generate_val_images = False

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
                
                if self.norm_before_merge or self.norm_after_merge:
                    slices_i2t = self.att2mask(slices_i2t, p=self.output_power)
                
                slices_i2t = slices_i2t.mean(dim=1)

                slices_i2t = rearrange(slices_i2t, "b (h w) l -> b l h w", h=64, w=64).squeeze(0)
                slices_i2t = self.mask_merge(slices_i2t, self.kl_threshold, grid=None)

                # torch.Size([1, 1, 4096, 154])
                # _outputs = rearrange(slices_i2t, "b (h w) l -> b l h w", h=64, w=64)
                _outputs = slices_i2t.unsqueeze(0)

                B = _outputs.shape[0]

                if self.dataset_name == "sam_resized":
                    idxs = torch.arange(0, len(targets[0]["labels"])) if targets is not None else None
                    target_mask = targets[0]["masks"].to(self.device)
                else:
                    idxs, targets = self.get_gt_indices(batched_inputs)
                    idxs = idxs[0]
                    if idxs == None or len(idxs) == 0:
                        return []
                    if batched_inputs[0]["sem_seg"] is None:
                        target_mask = torch.stack([targets.to(self.device) == x for x in idxs], dim=0)
                        target_mask = F.interpolate(target_mask.unsqueeze(0).float(), size=(64, 64), mode='nearest')[0]
                    else:
                        target_mask = torch.stack([batched_inputs[0]["sem_seg"].to(self.device) == x for x in idxs], dim=0)
                    if target_mask.ndim == 4:
                        target_mask = target_mask.squeeze(1)

                num_classes = len(self.test_class_texts)

                # hungarian matching
                _outputs_dict = {
                    "pred_logits": torch.zeros(_outputs.shape[0], _outputs.shape[1], num_classes, device=self.device),
                    "pred_masks": _outputs.to(self.device),
                }
                
                _target = [{
                    "labels": idxs,
                    "masks": target_mask,
                }]
                
                # import pdb; pdb.set_trace()
                source, target = self.matcher(_outputs_dict, _target)[0][0]

                _outputs_tgt = torch.zeros(B, len(idxs), _outputs.shape[-2], _outputs.shape[-1], device=self.device)
                for src, tgt in zip(source, target):
                    _outputs_tgt[:, tgt] = _outputs[:, src]
                
                _outputs = _outputs_tgt
                selected_idxs = idxs

                if self.visualize_val_attention:
                    self.plot_attention(_outputs, _target, images_resized, selected_idxs.unsqueeze(0), self.test_class_texts, save_name=batched_inputs[0]["file_name"])

                if len(idxs) == 0:
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

                    # if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = sem_seg_postprocess(
                        mask_pred_result, image_size, height, width
                    )

                    processed_results.append({"sem_seg": mask_pred_result})

            return processed_results
        
    def prepare_targets(self, targets, images, max_mask=77, sort=True):
        h, w = images.tensor.shape[-2:]
        new_targets = []

        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            labels = targets_per_image.gt_classes
            
            if not isinstance(gt_masks, BitMasks) and gt_masks.shape[0] == 0:
                # ForkedPdb.set_trace()
                # if no masks, return empty targets
                new_targets.append(
                    {
                        "labels": torch.tensor([], dtype=torch.int64, device=gt_masks.device),
                        "masks": torch.zeros((0, h, w), dtype=gt_masks.dtype, device=gt_masks.device),
                    }
                )
                continue

            if isinstance(gt_masks, BitMasks):
                gt_masks = gt_masks.tensor

            padded_masks = torch.zeros((gt_masks.shape[0], h, w), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

            if sort:
                # filter out small masks
                valid_ratio = padded_masks.sum([1,2])/ padded_masks[0].numel()
                valid_mask = valid_ratio > 0.005
                padded_masks = padded_masks[valid_mask]
                labels = labels[valid_mask]

                # sort padded_masks and labels by padded_masks area
                areas = padded_masks.sum([1, 2])
                sorted_indices = torch.argsort(areas, descending=True)
                padded_masks = padded_masks[sorted_indices]
                labels = labels[sorted_indices]

            if padded_masks.shape[0] > max_mask:
                # import pdb; pdb.set_trace()
                padded_masks = padded_masks[:max_mask]
                labels = targets_per_image.gt_classes[:max_mask]

            new_targets.append(
                {
                    "labels": labels,
                    "masks": padded_masks,
                }
            )
        return new_targets

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

    def mask_merge(self, attns, kl_threshold, grid=None):
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
    def generate_image(self, cond_prompts):

        print("Generating unconditional image...")
        for i in range(2):
            image = self.backbone.pipe(
                prompt="",
                guidance_scale=7.5,
                generator=torch.manual_seed(42+i),
                num_inference_steps=28,
            )
            image = image.images[0]
            if os.path.exists(f"{self.output_dir}/imgs/{self.iter}") is False:
                os.makedirs(f"{self.output_dir}/imgs/{self.iter}")
            image.save(f"{self.output_dir}/imgs/{self.iter}/unconditional{i}.png")
            image = np.array(image).transpose(2, 0, 1)
        self.logger.add_image("unconditional", image, self.iter)
        print("Generating conditional image...")

        for i in range(len(cond_prompts)):
            # log conditional generation result
            image = self.backbone.pipe(
                prompt=cond_prompts[i],
                guidance_scale=7.5,
                generator=torch.manual_seed(42),
                num_inference_steps=28,
            )
            image = image.images[0]
            if os.path.exists(f"{self.output_dir}/imgs/{self.iter}") is False:
                os.makedirs(f"{self.output_dir}/imgs/{self.iter}")
            image.save(f"{self.output_dir}/imgs/{self.iter}/conditional{i}.png")
            image = np.array(image).transpose(2, 0, 1)
        self.logger.add_image("conditional", image, self.iter)

    @torch.no_grad()
    def get_gt_indices(self, batched_inputs):
        if "sem_seg" in batched_inputs[0].keys() and batched_inputs[0]["sem_seg"] is not None:
            targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
            idxs = [t.long().flatten().bincount()[:171].nonzero().squeeze(-1) for t in targets]
        else:
            metadata = MetadataCatalog.get(self.dataset_name)
            targets = load_sem_seg(gt_root=metadata.sem_seg_root, image_root=metadata.image_root)
            sem_seg_file_name = None

            if 'image_name' in batched_inputs[0].keys():
                for t in targets:
                    if t['file_name'] == batched_inputs[0]['image_name']:
                        sem_seg_file_name = t['sem_seg_file_name']
                        break
            else:
                sem_seg_file_name = targets[0]['sem_seg_file_name']
            if sem_seg_file_name is not None:
                from PIL import Image
                
                sem_seg = Image.open(sem_seg_file_name)
                targets = torch.tensor(np.array(sem_seg), dtype=torch.long)
                idxs = [targets.flatten().bincount()[:171].nonzero().squeeze(-1).to(self.device)]
        return idxs, targets

    @torch.no_grad()
    def plot_attention(self, pred, gt, img, indices, classnames, save_name="attn_vis.png"):
        """
            pred: predicted attention maps, shape [B, N, H, W]
            gt: ground truth masks, shape [B, N, H, W]
            indices: list of indices for each batch, each item is a list of indices for the attention maps
            save_path: directory to save the attention maps and ground truth masks
        """
        save_name = save_name.split('/')[-1].replace('.jpg', '')
        
        # Create main attention visualization directory
        if not os.path.exists(f"{self.output_dir}/attn_vis"):
            os.makedirs(f"{self.output_dir}/attn_vis")

        print(f"Saving attention maps to {self.output_dir}/attn_vis/{save_name}")

        for bs in range(pred.shape[0]):
            # Create individual folder for this image
            img_folder = f"{self.output_dir}/attn_vis/{save_name}_bs{bs}"
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

            attn_map = pred[bs].clone().detach()
            gt_mask = gt[bs]['masks'].clone().detach()
            gt_mask = F.interpolate(gt_mask[None].float(), size=(64, 64), mode='nearest')[0]
            attn_map = attn_map.cpu().numpy()
            
            # Save individual attention maps directly as numpy arrays
            for j, idx in enumerate(indices[bs]):
                # Save individual attention map as image using PIL for visualization
                from PIL import Image
                import matplotlib.pyplot as plt
                import matplotlib.colors as mcolors
                
                # Save attention map with viridis colormap
                attn_data = attn_map[j]
                plt.figure(figsize=(8, 8))
                plt.imshow(attn_data, cmap='viridis')
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(f"{img_folder}/attention_map_{j}_{classnames[idx]}.png", dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close()
                
                # Save individual GT mask as binary image
                gt_data = gt_mask[j].cpu().numpy()
                gt_binary = (gt_data > 0).astype(np.uint8) * 255
                gt_img = Image.fromarray(gt_binary, mode='L')
                gt_img.save(f"{img_folder}/gt_mask_{j}_{classnames[idx]}.png")

            # Save original image
            img_np = img[bs].cpu().numpy().transpose(1, 2, 0)
            # Normalize image to 0-255 range if needed
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
            
            # Save as image
            img_pil = Image.fromarray(img_np)
            img_pil.save(f"{img_folder}/original_image.png")

            # Save merged attention map (argmax) with pastel tone colormap
            attn_map_argmax = pred[bs].argmax(dim=0).cpu().numpy()
            
            # Create pastel colormap
            pastel_colors = [
                [255, 182, 193],  # Light pink
                [173, 216, 230],  # Light blue
                [144, 238, 144],  # Light green
                [255, 218, 185],  # Peach
                [221, 160, 221],  # Plum
                [176, 224, 230],  # Powder blue
                [255, 228, 196],  # Bisque
                [240, 248, 255],  # Alice blue
                [255, 240, 245],  # Lavender blush
                [245, 245, 220],  # Beige
            ]
            
            # Create pastel version for visualization
            unique_labels = np.unique(attn_map_argmax)
            pastel_attn = np.zeros((*attn_map_argmax.shape, 3), dtype=np.uint8)
            
            for i, label in enumerate(unique_labels):
                if label < len(pastel_colors):
                    pastel_attn[attn_map_argmax == label] = pastel_colors[label]
                else:
                    # Generate random pastel color for additional labels
                    pastel_attn[attn_map_argmax == label] = np.random.randint(150, 255, 3)
            
            pastel_attn_img = Image.fromarray(pastel_attn)
            pastel_attn_img.save(f"{img_folder}/merged_attention_argmax.png")

            # Save merged GT mask (argmax) with pastel tone colormap
            gt_mask_argmax = gt_mask.argmax(dim=0).cpu().numpy()
            
            # Create pastel version for GT
            pastel_gt = np.zeros((*gt_mask_argmax.shape, 3), dtype=np.uint8)
            for i, label in enumerate(unique_labels):
                if label < len(pastel_colors):
                    pastel_gt[gt_mask_argmax == label] = pastel_colors[label]
                else:
                    pastel_gt[gt_mask_argmax == label] = np.random.randint(150, 255, 3)
            
            pastel_gt_img = Image.fromarray(pastel_gt)
            pastel_gt_img.save(f"{img_folder}/merged_gt_argmax.png")

            # Save metadata about the visualization
            metadata = {
                'class_names': classnames,
                'indices': indices[bs].tolist(),
                'attention_shape': attn_map.shape,
                'gt_shape': gt_mask.shape,
                'image_shape': img_np.shape,
                'num_classes': len(indices[bs])
            }
            
            import json
            with open(f"{img_folder}/metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

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
