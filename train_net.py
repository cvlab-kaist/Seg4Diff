# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import copy
import itertools
import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List, Set, IO, cast

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine.hooks import BestCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import CityscapesInstanceEvaluator, CityscapesSemSegEvaluator, \
    COCOEvaluator, COCOPanopticEvaluator, DatasetEvaluators, SemSegEvaluator, verify_results, \
    DatasetEvaluator

from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer, TrainerBase, HookBase
from detectron2.utils.logger import _log_api_usage
from detectron2.utils.events import EventStorage
from detectron2.modeling.postprocessing import sem_seg_postprocess

from plain_train_net import create_ddp_model

from detectron2.utils.file_io import PathManager
import numpy as np
from PIL import Image
import glob
import weakref

import pycocotools.mask as mask_util

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
import json
from tqdm import tqdm

from scipy.optimize import linear_sum_assignment as LinearSumAssignment

import warnings
warnings.filterwarnings('ignore')
try:
    #from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
except:
    pass

def _fast_hist(label_true, label_pred, n_class):
    # Adapted from https://github.com/janghyuncho/PiCIE/blob/c3aa029283eed7c156bbd23c237c151b19d6a4ad/utils.py#L99
    pred_n_class = np.maximum(n_class,label_pred.max()+1)
    mask = (label_true >= 0) & (label_true < n_class) # Exclude unlabelled data.
    hist = np.bincount(pred_n_class * label_true[mask] + label_pred[mask],\
                       minlength=n_class * pred_n_class).reshape(n_class, pred_n_class)
    return hist

def hungarian_matching(pred,label,n_class):
  # X,Y: b x 512 x 512
  batch_size = pred.shape[0]
  tp = np.zeros(n_class)
  fp = np.zeros(n_class)
  fn = np.zeros(n_class)
  all = 0
  for i in range(batch_size):
    # import pdb; pdb.set_trace()
    # if np.unique(pred[i]) == np.unique(label[])
    hist = _fast_hist(label[i].flatten(),pred[i].flatten(),n_class)
    row_ind, col_ind = LinearSumAssignment(hist,maximize=True)
    all += hist.sum()
    fn += (np.sum(hist, 1) - hist[row_ind,col_ind])
    tp += hist[row_ind,col_ind]
    hist = hist[:, col_ind] # re-order hist to align labels to calculate FP
    fp += (np.sum(hist, 0) - np.diag(hist))
  return tp,fp,fn,all

class VOCbEvaluator(SemSegEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """
    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=np.int)
            pred[pred >= 20] = 20
            with PathManager.open(self.input_file_to_gt_file[input["file_name"]], "rb") as f:
                gt = np.array(Image.open(f), dtype=np.int)

            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))

# MaskFormer
from seg4diff import (
    DETRPanopticDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SamBaselineDatasetMapperJSON,
    add_seg4diff_config,
)

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DETR.
    """

    def __init__(self, cfg):
        # super().__init__(cfg)
        self._hooks: List[HookBase] = []
        self.iter: int = 0
        self.start_iter: int = 0
        self.max_iter: int
        self.storage: EventStorage
        _log_api_usage("trainer." + self.__class__.__name__)

        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False, )
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self._trainer.gradient_accumulation_steps = cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        # commented out to avoid duplicate hooks registration (e.g., BestCheckpointer)
        # self.register_hooks(self.build_hooks())

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg", "sem_seg_background"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )

        # if evaluator_type == "sem_seg_background":
        #     evaluator_list.append(
        #         VOCbEvaluator(
        #             dataset_name,
        #             distributed=True,
        #             output_dir=output_folder,
        #         )
        #     )
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
        ]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."

            evaluator_list = [
                # original Cityscapes IoU evaluator
                # CityscapesSemSegEvaluator(dataset_name),
                # generic semantic‐segmentation evaluator for ACC metrics
                SemSegEvaluator(
                    dataset_name,
                    distributed=cfg.MODEL.DEVICE != "cpu",
                    output_dir=output_folder,
                ),
            ]
        if evaluator_type == "cityscapes_panoptic_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_unsupervised(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """

        logger = logging.getLogger(__name__)
        # if isinstance(evaluators, DatasetEvaluator):
        #     evaluators = [evaluators]
        # if evaluators is not None:
        #     assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
        #         len(cfg.DATASETS.TEST), len(evaluators)
        #     )
        
        N_CLASS = len(model.test_class_texts)
        TP = np.zeros(N_CLASS)
        FP = np.zeros(N_CLASS)
        FN = np.zeros(N_CLASS)
        ALL = 0
        model.eval()
        for i, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            for idx, inputs in enumerate(tqdm(data_loader, desc="Evaluating")):
                outputs = model(inputs)
                output = outputs[0]["sem_seg"]
                image_size = (inputs[0]['image'].shape[1], inputs[0]['image'].shape[2])
                out_height, out_width = image_size
                sem_seg_out = sem_seg_postprocess(output, image_size, out_height, out_width)
                sem_seg_out = sem_seg_out.argmax(dim=0).cpu().numpy()
                labels = inputs[0]["sem_seg"].cpu().numpy()
                tp, fp, fn, all = hungarian_matching(sem_seg_out, labels, N_CLASS)
                TP += tp
                FP += fp
                FN += fn
                ALL += all

        acc = TP.sum()/ALL
        iou = TP / (TP + FP + FN)
        miou = np.nanmean(iou)
        print("final pixel accuracy:{}, mIoU:{}".format(acc, miou))

        print(
            f"""
            ############ UNSUPERVISED EVALUATION ############
            Model: {cfg.MODEL.WEIGHTS} kl_threshold: {cfg.MODEL.KL_THRESHOLD} \n
            Evaluate \n
            layer : {cfg.MODEL.BACKBONE.ATTENTION_LAYERS}  |  head {cfg.MODEL.HEAD} \n
            Dataset: {cfg.DATASETS.TEST} \n
                \t Final pixel accuracy: {acc} \n
                \t mIoU: {miou}
            #################################################
            """
        )
        
        logname = f'{cfg.DATASETS.TEST[0].split("_")[0]}_9th_mean'
        with open(f"{cfg.OUTPUT_DIR}/{logname}.txt", "a") as f:
            f.write(
                f""" 
                ############ UNSUPERVISED EVALUATION ############
                Model: {cfg.MODEL.WEIGHTS} kl_threshold: {cfg.MODEL.KL_THRESHOLD} \n
                Evaluate \n
                layer : {cfg.MODEL.BACKBONE.ATTENTION_LAYERS}  |  head {cfg.MODEL.HEAD} \n
                Dataset: {cfg.DATASETS.TEST} \n
                \t Final pixel accuracy: {acc} \n
                \t mIoU: {miou}
                #################################################
                """
            )
        return miou
    
    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
        # DETR-style dataset mapper for COCO panoptic segmentation
        elif cfg.INPUT.DATASET_MAPPER_NAME == "detr_panoptic":
            mapper = DETRPanopticDatasetMapper(cfg, True)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "SamBaselineDatasetMapperJSON":
            mapper = SamBaselineDatasetMapperJSON(cfg, True)
        else:
            mapper = None

        # return build_detection_train_loader(cfg, mapper=mapper,sampler=torch.utils.data.SequentialSampler(cfg.DATASETS.TRAIN),)
        return build_detection_train_loader(cfg, mapper=mapper,)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if dataset_name == "sam_resized":
            # For SAM dataset, we use a custom mapper
            return build_detection_test_loader(
                cfg,
                dataset_name,
                mapper=SamBaselineDatasetMapperJSON(cfg, is_train=False),
            )
        else:
            # For other datasets, we use the default mapper
            return build_detection_test_loader(cfg, dataset_name)
        
    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        use_lora = cfg.MODEL.BACKBONE.USE_LORA
        use_learnable_tokens = cfg.MODEL.BACKBONE.USE_LEARNABLE_TOKENS

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        # import ipdb;
        # ipdb.set_trace()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=True):
                if not value.requires_grad:
                    continue
                # if use_lora and "lora" not in module_param_name:
                #     continue
                if use_learnable_tokens and "prompt_tokens" not in module_param_name:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if "clip_model" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.CLIP_MULTIPLIER
                # for deformable detr

                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_seg4diff_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask_former")
    return cfg


def main(args):
    cfg = setup(args)
    torch.set_float32_matmul_precision("high")
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        # res = Trainer.test(cfg, model)
        if cfg.MODEL.EVAL_UNSUP:
           res = Trainer.test_unsupervised(cfg, model)
        else:
            res = Trainer.test(cfg, model)
            if cfg.TEST.AUG.ENABLED:
                res.update(Trainer.test_with_TTA(cfg, model))
            if comm.is_main_process():
                verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    if comm.is_main_process():
        best_ckpt = BestCheckpointer(
            eval_period=cfg.TEST.EVAL_PERIOD,  # how often to evaluate
            checkpointer=trainer.checkpointer,
            val_metric="sem_seg/mIoU",  # metric to monitor
            mode="max",  # mode for the metric
            )
    else:
        # If not the main process, we still need to register hooks to avoid errors
        # but we don't need to monitor any metrics.
        best_ckpt = BestCheckpointer(
            eval_period=cfg.TEST.EVAL_PERIOD,
            checkpointer=trainer.checkpointer,
            val_metric=None,  # no metric to monitor
            mode="max",  # mode for the metric
        )
    # ensure it runs alongside the default hooks:
    trainer.register_hooks([best_ckpt] + trainer.build_hooks())
    # trainer.register_hooks(trainer.build_hooks())
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
