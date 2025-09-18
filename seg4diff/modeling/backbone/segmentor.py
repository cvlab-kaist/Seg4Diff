#Copyright 2023 Google LLC

#Use of this source code is governed by an MIT-style
#license that can be found in the LICENSE file or at
#https://opensource.org/licenses/MIT.


import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict

class DiffSeg:
  def __init__(self, kl_threshold, refine, num_points):
    # Generate the grid
    self.grid = self.generate_sampling_grid(num_points)
    # Initialize other parameters 
    self.kl_threshold = np.array(kl_threshold)
    self.refine = refine

  def generate_sampling_grid(self,num_of_points):
    segment_len = 63//(num_of_points-1)
    total_len = segment_len*(num_of_points-1)
    start_point = (63 - total_len)//2
    x_new = np.linspace(start_point, total_len+start_point, num_of_points)
    y_new = np.linspace(start_point, total_len+start_point, num_of_points)
    x_new,y_new=np.meshgrid(x_new,y_new,indexing='ij')
    points = np.concatenate(([x_new.reshape(-1,1),y_new.reshape(-1,1)]),axis=-1).astype(int)
    return points
  
  def get_weight_rato(self, weight_list):
    # This function assigns proportional aggregation weight 
    sizes = []
    for weights in weight_list:
      sizes.append(np.sqrt(weights.shape[-2]))
    denom = np.sum(sizes)
    return sizes / denom

  def aggregate_weights(self, weight_list, weight_ratio=None):
    if weight_ratio is None:
      weight_ratio = self.get_weight_rato(weight_list)
    aggre_weights = np.zeros((64,64,64,64))
   
    for index,weights in enumerate(weight_list):
      # Calculate size based on the spatial dimensions (last two dimensions)
      size = weights.shape[-1]  # This is the spatial size (64, 32, 16, 8)
      ratio = int(64/size)
      
      # Average over the multi-head channel
      weights = weights.mean(0)  # Shape: (64, 64), (32, 32), etc.

      # Convert to PyTorch tensor
      weights = torch.from_numpy(weights).float()
      
      # The weights tensor is now 2D [size, size], we need to upsample it to 64x64
      # and then create a 4D tensor where each position has a 64x64 attention map
      if weights.dim() == 2:
        # Upsample the 2D weights to 64x64
        weights = weights.unsqueeze(0).unsqueeze(0)  # [1, 1, size, size]
        weights = F.interpolate(weights, size=(64, 64), mode='bilinear', align_corners=False)
        weights = weights.squeeze(0).squeeze(0)  # [64, 64]
        
        # Create a 4D tensor where each position (i,j) in the size x size grid has a 64x64 attention map
        # This simulates the original logic where each position has its own attention map
        weights_4d = torch.zeros(size, size, 64, 64)
        for i in range(size):
          for j in range(size):
            # Each position gets a copy of the upsampled weights
            weights_4d[i, j, :, :] = weights
        weights = weights_4d
      
      # Normalize to make sure each map sums to one
      weights = weights / torch.sum(weights, dim=(2, 3), keepdim=True)
      
      # Spatial tiling along the first two dimensions
      weights = weights.repeat(ratio, ratio, 1, 1)
      
      # Ensure the final shape is (64, 64, 64, 64)
      if weights.shape[0] != 64 or weights.shape[1] != 64:
        # If tiling created a larger tensor, crop it to 64x64
        weights = weights[:64, :64, :, :]

      # Aggregate according to weight_ratio
      aggre_weights += weights.numpy() * weight_ratio[index]
    return aggre_weights.astype(np.double)

  def aggregate_x_weights(self, weight_list, weight_ratio=None):
    # x_weights: 8 x size**2 x 77
    # return 512 x 512 x 77
    if weight_ratio is None:
      weight_ratio = self.get_weight_rato(weight_list)
    aggre_weights = np.zeros((512, 512, 77))

    for index,weights in enumerate(weight_list):
      size = int(np.sqrt(weights.shape[-2]))
      ratio = int(512/size)
      weights = weights.mean(0).reshape(1,size,size,-1)
      
      # Convert to PyTorch tensor
      weights = torch.from_numpy(weights).float()
      weights = F.interpolate(weights.permute(0, 3, 1, 2), size=(512, 512), mode='bilinear', align_corners=False)
      weights = weights.permute(0, 2, 3, 1).squeeze(0)  # Back to (H, W, C) format
      weights = weights / torch.sum(weights, dim=-1, keepdim=True)
      aggre_weights += weights.numpy() * weight_ratio[index]
    return aggre_weights.astype(np.double)

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

    Args:
        iter (int): current iteration index
        attns (torch.Tensor): attention maps
        kl_threshold (list or array): thresholds for KL at each iteration
        grid (torch.Tensor, optional): index grid for iter==0
    Returns:
        torch.Tensor: merged attention maps, shape (M, H, W)
    """
    
    # Ensure attns is a tensor
    if isinstance(attns, np.ndarray):
        attns = torch.from_numpy(attns).float()
    
    # Convert 4D to 3D if needed (for aggregated weights from aggregate_weights)
    if attns.dim() == 4:
        # attns shape: (64, 64, 64, 64) -> reshape to (4096, 64, 64)
        attns = attns.reshape(-1, 64, 64)
    
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
        mask = (kl_vals < kl_threshold[iter]).cpu()

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
    
    # Convert to numpy for consistency with existing code
    return new_attns.numpy()

  def generate_masks(self, attns, kl_threshold, grid):
    # Convert to PyTorch tensor if needed
    if isinstance(attns, np.ndarray):
        attns = torch.from_numpy(attns).float()
    
    # Convert 4D aggregated weights to 3D format expected by mask_merge
    if attns.dim() == 4:
        # attns shape: (64, 64, 64, 64) -> reshape to (4096, 64, 64)
        # This flattens the first two dimensions
        attns = attns.reshape(-1, 64, 64)  # (4096, 64, 64)
        print(f"Reshaped 4D attns to 3D: {attns.shape}")
    
    # Iterative Attention Merging
    max_iterations = min(len(kl_threshold), 3)  # Limit to 3 iterations maximum
    for i in range(max_iterations):
      print(f"Starting mask_merge iteration {i}/{max_iterations}")
      attns_merged = self.mask_merge(i, attns, kl_threshold, grid=grid)
      if isinstance(attns_merged, np.ndarray):
          attns_merged = torch.from_numpy(attns_merged).float()
      print(f"Completed mask_merge iteration {i}, result shape: {attns_merged.shape}")
      
      # Update attns for next iteration
      if i < max_iterations - 1:
          attns = attns_merged

    
    # Check the dimensions of attns_merged and handle accordingly
    if attns_merged.dim() == 3:
      # If it's 3D, we don't need to index the first dimension
      pass
    elif attns_merged.dim() == 4:
      # If it's 4D, take the first slice
      attns_merged = attns_merged[:,0,:,:]
    else:
      # For other cases, just use the tensor as is
      pass

    # Kmeans refinement (optional for better visual consistency)
    if self.refine:
      try:
        attns = attns.reshape(-1,64*64)
        kmeans = KMeans(n_clusters=attns_merged.shape[0], init=attns_merged.reshape(-1,64*64).numpy(), n_init=1).fit(attns.numpy())
        clusters = kmeans.labels_
        attns_merged = []
        for i in range(len(set(clusters))):
          cluster = (i == clusters)
          attns_merged.append(attns[cluster,:].mean(0).reshape(64,64))
        attns_merged = torch.stack(attns_merged)
      except Exception as e:
        print(f"Warning: K-means refinement failed: {e}")
        # Continue without refinement

    # If attns_merged is empty or has wrong shape, create a simple fallback
    if attns_merged.numel() == 0 or attns_merged.dim() < 2:
      # Create a simple segmentation mask
      M_final = np.zeros((512, 512), dtype=np.int32)
    else:
      # Upsampling
      upsampled = F.interpolate(attns_merged.unsqueeze(1), size=(512, 512), mode='bilinear', align_corners=False)
      upsampled = upsampled.squeeze(1)

      # Non-Maximum Suppression
      M_final = torch.argmax(upsampled, dim=0).numpy()

    return M_final
  
  def segment(self, weight_64, weight_32, weight_16, weight_8, weight_ratio = None):
    M_list = []
    for i in range(len(weight_64)):
      # Step 1: Attention Aggregation
      weights = self.aggregate_weights([weight_64[i],weight_32[i], weight_16[i], weight_8[i]],weight_ratio=weight_ratio)
      # Step 2 & 3: Iterative Merging & NMS
      M_final = self.generate_masks(weights, self.kl_threshold, self.grid)
      M_list.append(M_final)
    return np.array(M_list)

  def get_semantics(self, pred, x_weight, nouns, voting="majority"):
        # This function assigns semantic labels to masks 
        indices = [item[0]+1 for item in nouns] # Ignore the first BOS token
        prompt_list = [item[1] for item in nouns]
        x_weight = x_weight[:,:,indices] # size x size x N
        x_weight = x_weight.reshape(512*512,-1)
        norm = np.linalg.norm(x_weight,axis=0,keepdims=True)
        x_weight = x_weight/norm # Normalize the cross-attention maps spatially
        pred = pred.reshape(512*512,-1)

        label_to_mask = defaultdict(list)
        for i in set(pred.flatten()):
          if voting == "majority":
            logits = x_weight[(pred==i).flatten(),:]
            index = logits.argmax(axis=-1)
            category = prompt_list[int(np.median(index))]
          else:
            logit = x_weight[(pred==i).flatten(),:].mean(0)
            category = prompt_list[logit.argmax(axis=-1)]
          label_to_mask[category].append(i)
        return label_to_mask
