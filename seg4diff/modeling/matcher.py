# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.amp import autocast
from detectron2.projects.point_rend.point_features import point_sample

import sys
import pdb

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


def batch_dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule

def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    prob = inputs.sigmoid()
    prob = inputs
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    # focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy(
    #     inputs, torch.ones_like(inputs), reduction="none"
    # )
    # focal_neg = (prob ** gamma) * F.binary_cross_entropy(
    #     inputs, torch.zeros_like(inputs), reduction="none"
    # )
    if alpha >= 0:
        focal_pos = focal_pos * alpha / hw
        focal_neg = focal_neg * (1 - alpha) / hw

    loss = torch.einsum("nc,mc->nm", focal_pos, targets) + torch.einsum(
        "nc,mc->nm", focal_neg, (1 - targets)
    )

    return loss #/ hw


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"
        self.num_points = 12544

    # original! 
    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets, cost_thresh=None):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Work out the mask padding size
        masks = [v["masks"] for v in targets]
        h_max = max([m.shape[1] for m in masks])
        w_max = max([m.shape[2] for m in masks])

        indices = []
        valid_mask = [] if cost_thresh is not None else None

        # Iterate through batch size
        for b in range(bs):

            if targets[b]["masks"].numel() == 0:
                # If there are no targets, we match all predictions to the no-object class
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                if valid_mask is not None:
                    valid_mask.append(torch.ones(num_queries, dtype=torch.bool))
                continue

            out_prob = outputs["pred_logits"][b].softmax(-1).float()  # [num_queries, num_classes]
            out_mask = outputs["pred_masks"][b].float()  # [num_queries, H_pred, W_pred]

            tgt_ids = targets[b]["labels"]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            # cost_class = -out_prob[:, tgt_ids]

            # Downsample gt masks to save memory
            tgt_mask = F.interpolate(tgt_mask[:, None], size=out_mask.shape[-2:], mode="nearest")

            # Flatten spatial dimension
            out_mask = out_mask.flatten(1)  # [batch_size * num_queries, H*W]
            tgt_mask = tgt_mask[:, 0].flatten(1)  # [num_total_targets, H*W]

            # Compute the focal loss between masks
            cost_mask = batch_sigmoid_focal_loss(out_mask, tgt_mask)

            # Compute the dice loss betwen masks
            cost_dice = batch_dice_loss(out_mask, tgt_mask)

            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                # + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )

            # If cost_thresh is set, we can filter out some predictions
            if valid_mask is not None:
                # Create a mask for the cost matrix where the cost is below the threshold
                is_below_thresh = C.mean(0) < cost_thresh
                valid_mask.append(is_below_thresh)
                print(is_below_thresh.sum() / is_below_thresh.numel(), "predictions below the threshold", cost_thresh)

            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))
            
            # try:
            #     indices.append(linear_sum_assignment(C))
            # except:
            #     print(cost_mask.isnan().sum(), cost_class.isnan().sum(), cost_dice.isnan().sum())    
            #     ForkedPdb().set_trace()
                
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ], valid_mask

    @torch.no_grad()
    def forward(self, outputs, targets, cost_thresh=None):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        #try:
        with autocast(enabled=False, device_type="cuda"):
            return self.memory_efficient_forward(outputs, targets, cost_thresh=cost_thresh)
        # except:
        #     ForkedPdb().set_trace()
            
    def __repr__(self):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)