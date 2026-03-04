#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2024, University of Liege, KAUST and University of Oxford
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#


# 导入 PyTorch 相关库
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def lp_loss(pred, target, p=0.7, eps=1e-6):
    diff = torch.abs(pred - target) + eps  # Prevent numerical instability for zero power
    loss = torch.pow(diff, p).mean()
    return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def laplacian_smooth_loss(vertices, faces):
    """
    Uniform Laplacian geometric smoothness loss:
    L = mean_i || v_i - mean_{j in N(i)} v_j ||^2
    Args:
        vertices: (V, 3) vertex coordinates
        faces: (F, 3) triangle indices, long type
    Returns:
        smoothness loss value
    """
    V = vertices.shape[0]
    device = vertices.device
    dtype = vertices.dtype

    # Extract three edges of each triangle
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
    
    # Build undirected edges, accumulate both directions: i<->j, j<->k, k<->i
    src = torch.cat([i, j, k, j, k, i], dim=0)           # indices for neighbor and degree accumulation
    dst = torch.cat([j, k, i, i, j, k], dim=0)           # corresponding neighbor vertices

    # Compute degree (number of neighbors) for each vertex
    deg = torch.zeros(V, device=device, dtype=dtype)
    deg.index_add_(0, src, torch.ones(src.numel(), device=device, dtype=dtype))

    # Compute sum of neighbor coordinates for each vertex
    neighbor_sum = torch.zeros_like(vertices)
    neighbor_sum.index_add_(0, src, vertices[dst])

    # Avoid division by zero, compute Laplacian operator
    deg = deg.clamp_min(1.0).unsqueeze(1)
    lap = vertices - neighbor_sum / deg
    
    # Compute smoothness loss (mean squared L2 norm)
    loss = (lap.pow(2).sum(dim=1)).mean()
    return loss

def silhouette_bce_loss(pred_mask, gt_mask):
    """
    Silhouette loss: binary cross entropy
    pred_mask, gt_mask: (H,W) or (1,H,W) torch.float32, 0~1
    """
    if pred_mask.dim() == 3:
        pred_mask = pred_mask.squeeze(0)
    if gt_mask.dim() == 3:
        gt_mask = gt_mask.squeeze(0)
    pred_mask = pred_mask.clamp(0,1)
    gt_mask = gt_mask.clamp(0,1)
    return F.binary_cross_entropy(pred_mask, gt_mask)

def double_vertex_deviation_loss(vertices, vertices_ref, faces):
    # vertices: (N, 3) current vertices
    # vertices_ref: (N, 3) initial/reference vertices
    # faces: (M, 3)
    delta = vertices - vertices_ref  # (N, 3)
    # 构建所有边索引
    edges = torch.cat([faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]], dim=0)  # (3*M, 2)
    diff = delta[edges[:,0]] - delta[edges[:,1]]  # (E, 3)
    loss = (diff ** 2).sum(-1).mean()
    return loss

def pearson_correlation_loss(rend_depth, pred_depth, mask, oneside=False):
    """
    rend_depth: (H, W) rendered depth
    pred_depth: (H, W) gt/predicted depth
    mask: (H, W) valid region mask
    Returns: (loss, contrib_map)
    """
    # 1. Select valid region only
    rend_valid = rend_depth[mask]
    pred_valid = pred_depth[mask]
    # 2. Normalize (subtract mean, divide by std)
    rend_centered = rend_valid - rend_valid.mean()
    pred_centered = pred_valid - pred_valid.mean()
    rend_std = torch.sqrt((rend_centered ** 2).sum() + 1e-8)
    pred_std = torch.sqrt((pred_centered ** 2).sum() + 1e-8)
    if oneside:
        z_rend = rend_centered / rend_std
        z_pred = pred_centered / pred_std
        # Only keep region where z_rend < z_pred
        region = z_rend > z_pred
        if region.sum() < 10:
            contrib = torch.zeros_like(rend_depth)
            return rend_depth.new_tensor(0.0), contrib
        z_rend = z_rend[region]
        z_pred = z_pred[region]
        prod = z_rend * z_pred
        correlation = prod.sum() / (torch.sqrt((z_rend ** 2).sum() + 1e-8) * torch.sqrt((z_pred ** 2).sum() + 1e-8))
        contrib = torch.zeros_like(rend_depth).squeeze()
        contrib_mask = mask.squeeze()
        contrib_valid = torch.zeros_like(rend_valid)
        # Normalize prod to [0,1] in region only
        prod_norm = prod
        if prod.numel() > 1:
            min_p, max_p = prod.min(), prod.max()
            denom = (max_p - min_p).clamp_min(1e-8)
            prod_norm = (prod - min_p) / denom
        contrib_valid[region] = prod_norm
        contrib[contrib_mask] = contrib_valid
        return 1.0 - correlation, contrib
    else:
        # 3. Compute correlation coefficient
        prod = rend_centered * pred_centered
        correlation = prod.sum() / (rend_std * pred_std)
        # 4. Loss = 1 - Correlation (higher correlation, lower loss)
        # Contribution map normalized by std (consistent with Pearson distribution)
        contrib = torch.zeros_like(rend_depth).squeeze()
        contrib_mask = mask.squeeze()
        # Normalize to [-1,1] then map to [0,1] for visualization
        contrib_norm = prod / (rend_std * pred_std)
        contrib[contrib_mask] = (contrib_norm + 1) / 2  # [-1,1]→[0,1]
        return 1.0 - correlation, contrib
