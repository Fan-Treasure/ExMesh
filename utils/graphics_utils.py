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

import torch
import math
import numpy as np
from typing import NamedTuple
import xatlas
import nvdiffrast.torch as dr
import cv2
import scipy.ndimage


class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

class BasicMesh(NamedTuple):
    vertices : np.array        # (N,3)
    faces : np.array           # (M,3)
    vertex_colors : np.array   # (N,3)
    vertex_normals : np.array  # (N,3)
    uvs : np.array             # (K,2), k>=N
    uv_indices : np.array      # (M,3)
    vmapping : np.array        # (K), k>=N

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

# UV map extraction utilities
def extract_uv_map(mesh: BasicMesh):
    """
    Use xatlas for UV unwrapping, return BasicMesh with uvs.
    """
    if mesh.uvs is not None and isinstance(mesh.uvs, np.ndarray) and mesh.uvs.ndim == 2 and mesh.uvs.shape[1] == 2:
        return mesh

    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.uint32)
    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
    uvs = np.asarray(uvs, dtype=np.float32)
    vmapping = np.asarray(vmapping) 

    # Normalize to [0,1]
    uvs_min = np.min(uvs, axis=0, keepdims=True)
    uvs_max = np.max(uvs, axis=0, keepdims=True)
    denom = np.clip(uvs_max - uvs_min, 1e-8, None)
    uvs = (uvs - uvs_min) / denom
    
    return BasicMesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        vertex_colors=mesh.vertex_colors,
        vertex_normals=mesh.vertex_normals,
        uvs=uvs.astype(np.float32),
        uv_indices=indices.astype(np.int32),
        vmapping=vmapping.astype(np.int32)
    )

def vertex_color_to_uvmap(vertices, faces, uvs, vertex_colors, uv_indices, vmapping, texture_resolution, bg_color):
    """
    Bake mesh vertex colors to UV texture map using nvdiffrast.
    Args:
        uvs: (N_uv, 2) from xatlas
        uv_indices: (M, 3) new triangle indices from xatlas
        vertex_colors: (N, 3) mesh vertex colors
        vmapping: (N_uv,) mesh vertex to uvs mapping
        texture_resolution: (H, W)
    Returns:
        texture: (3, H, W) numpy, float32, [0,1]
    """
    H, W = texture_resolution
    device = "cuda"

    # Use uvs and uv_indices
    uv = torch.tensor(uvs, dtype=torch.float32, device=device)  # (N_uv,2)
    uv_ndc = uv * 2 - 1
    z = torch.zeros((uv_ndc.shape[0], 1), dtype=uv_ndc.dtype, device=device)
    w = torch.ones((uv_ndc.shape[0], 1), dtype=uv_ndc.dtype, device=device)
    uv_ndc_homo = torch.cat([uv_ndc, z, w], dim=1)
    pos = uv_ndc_homo.unsqueeze(0)  # (1, N_uv, 4)

    # Unwrap colors (each uv point corresponds to a color)
    color_uv = torch.tensor(vertex_colors, dtype=torch.float32, device=device)[vmapping]
    v_color = color_uv.unsqueeze(0)  # (1, N_uv, 3)

    faces_torch = torch.tensor(uv_indices, dtype=torch.int32, device=device)
    glctx = dr.RasterizeCudaContext(device=device)

    rast, _ = dr.rasterize(
        glctx=glctx,
        pos=pos,
        tri=faces_torch,
        resolution=(H, W)
    )  # rast: (1, H, W, 4)

    tex = dr.interpolate(v_color, rast, faces_torch)[0]
    if tex.dim() == 4:
        tex = tex.squeeze(0)
    mask = rast[0, ..., 3:] > 0
    tex = tex * mask.float()
    if bg_color == 1:
        background = torch.tensor([1.0, 1.0, 1.0], dtype=tex.dtype, device=tex.device).view(1, 1, 3)
        tex = tex + (1.0 - mask.float()) * background
    tex_np = tex.permute(2, 0, 1).detach().cpu().numpy()
    mask_np = mask.squeeze(-1).cpu().numpy()
    # Only dilate uvmap texture, do not return dilated mask
    tex_np, mask_dilated = dilate_texture_and_mask(tex_np, mask_np, pad=1)
    return tex_np, mask_np

def dilate_texture_and_mask(tex_np, mask_np, pad=1):
    # tex_np: (3, H, W), mask_np: (H, W)
    tex = tex_np.transpose(1,2,0)  # (H,W,3)
    mask = mask_np.astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=pad)
    tex_dilated = tex.copy()
    valid = mask > 0
    for i in range(pad):
        border = cv2.dilate(mask, kernel, iterations=1) - mask
        border_idx = np.where(border > 0)
        if len(border_idx[0]) == 0:
            mask = mask | border
            continue
        # Nearest neighbor fill
        dist, inds = scipy.ndimage.distance_transform_edt(~valid, return_indices=True)
        for c in range(3):
            channel = tex_dilated[...,c]
            channel[border_idx] = channel[inds[0][border_idx], inds[1][border_idx]]
            tex_dilated[...,c] = channel
        mask = mask | border
        valid = mask > 0
    return tex_dilated.transpose(2,0,1), mask_dilated

def uvmap_to_vertex_color(vertices, uvs, texture, vmapping, mask):
    """
    Map UV texture to mesh vertex colors.
    Args:
        vertices: (N,3) torch.Tensor (cuda/cpu)
        uvs: (N_uv,2) torch.Tensor, [0,1], (cuda/cpu)
        texture: (3,H,W) torch.Tensor, [0,1], (cuda)
        vmapping: (N_uv,) torch.LongTensor (cuda/cpu)
        mask: (H,W) torch.BoolTensor or torch.ByteTensor (cuda/cpu)
    Returns: (N,3) numpy
    """
    # Ensure all tensors are on the same device
    device = texture.device
    N = vertices.shape[0]
    if texture.dim() == 3:
        texture = texture.unsqueeze(0)
    H, W = texture.shape[2], texture.shape[3]

    # Batch pixel coordinates
    px = torch.clamp((uvs[:, 0] * (W-1)).round().long(), 0, W-1)
    py = torch.clamp((uvs[:, 1] * (H-1)).round().long(), 0, H-1)
    uv_valid = mask[py, px]
    if uv_valid.dtype != torch.bool:
        uv_valid = uv_valid.bool()

    # Valid uv
    uvs_valid = uvs[uv_valid]
    vmapping_valid = vmapping[uv_valid]
    # Batch grid_sample
    grid = uvs_valid.view(1, -1, 1, 2) * 2 - 1  # (1, N_valid, 1, 2)
    sampled = torch.nn.functional.grid_sample(texture, grid, align_corners=True, mode='bilinear')  # (1,3,N_valid,1)
    colors = sampled[0, :, :, 0].permute(1, 0)  # (N_valid, 3)

    # Aggregate to mesh vertices
    vertex_colors = torch.zeros((N, 3), dtype=torch.float32, device=device)
    counts = torch.zeros((N,), dtype=torch.float32, device=device)
    vertex_colors.index_add_(0, vmapping_valid, colors)
    counts.index_add_(0, vmapping_valid, torch.ones_like(vmapping_valid, dtype=torch.float32))
    mask_valid = counts > 0
    vertex_colors[mask_valid] /= counts[mask_valid][:, None]

    num_invalid = (~mask_valid).sum().item()

    # Nearest neighbor interpolation (for invalid vertices, back to cpu)
    if num_invalid > 0:
        from scipy.spatial import cKDTree
        verts_np = vertices.detach().cpu().numpy()
        colors_np = vertex_colors.detach().cpu().numpy()
        mask_valid_np = mask_valid.cpu().numpy()
        tree = cKDTree(verts_np[mask_valid_np])
        dists, idxs = tree.query(verts_np[~mask_valid_np], k=1)
        colors_np[~mask_valid_np] = colors_np[mask_valid_np][idxs]
        vertex_colors = torch.from_numpy(colors_np).to(device)

    return vertex_colors.cpu().numpy()
