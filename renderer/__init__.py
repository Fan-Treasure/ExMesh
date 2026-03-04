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
import torch.nn.functional as F
import nvdiffrast.torch as dr
from utils.point_utils import normal_from_depth_image


def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)
    st = max(int(scale/2)-1,0)
    if offset is not None:
        offset = offset[st::scale,st::scale]
    normal_ref = normal_from_depth_image(depth[st::scale,st::scale], 
                                            intrinsic_matrix.to(depth.device), 
                                            extrinsic_matrix.to(depth.device), offset)

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref

def _compute_vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute per-vertex normals (view space). verts: (N,3) view-space, faces: (M,3)."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=1)
    vn = torch.zeros_like(verts)
    for i in range(3):
        vn.index_add_(0, faces[:, i], fn)
    vn = F.normalize(vn + 1e-9, dim=1)
    return vn


def render(viewpoint_camera, mesh, pipe, bg_color: torch.Tensor):
    """
    Rendering with nvdiffrast: use mesh vertices and uvmap texture, output a dictionary compatible with training.
    Mesh should provide: _vertices(N,3), _uvs(N_uv,2), _uv_indices(M,3), _vmapping(N_uv,), _texture(3,Ht,Wt), _faces(M,3).
    Camera should provide: world_view_transform(4,4), full_proj_transform(4,4), image_height, image_width.
    """
    device = bg_color.device
    H, W = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)
    view = viewpoint_camera.world_view_transform
    proj = viewpoint_camera.projection_matrix
    
    # Convert camera MVP from D3D/GS NDC (z∈[0,1]) to OpenGL NDC (z∈[-1,1])
    proj_glndc = proj.clone()
    proj_glndc[2, :] = proj_glndc[2, :] * 2.0 - proj_glndc[3, :]

    # Rasterization preparation
    glctx = dr.RasterizeCudaContext()
    verts_world = mesh.get_vertices
    faces = mesh.get_faces.to(torch.int32).contiguous()
    ones = torch.ones((verts_world.shape[0], 1), device=verts_world.device, dtype=verts_world.dtype)
    v4 = torch.cat([verts_world, ones], dim=1)

    def rast_with_proj(proj_mat):
        mvp = proj_mat @ view
        pos_clip = (v4 @ mvp.t()).unsqueeze(0).contiguous()
        rast, triangle_id = dr.rasterize(glctx, pos_clip, faces, resolution=(H, W))
        cov = (rast[..., 3] > 0).float().mean().item()
        return pos_clip, rast, cov, triangle_id

    pos_clip, rast, cov, triangle_id = rast_with_proj(proj_glndc)

    # UV interpolation: each triangle uses its own UV vertices
    uvs = mesh.get_uvs
    uv_indices = mesh.get_uv_indices.to(torch.int32)
    uv_attr = uvs.unsqueeze(0)
    uv_map = dr.interpolate(uv_attr, rast, uv_indices)[0]
    
    # UV interpolation -> texture sampling
    grid = uv_map.clamp(0, 1) * 2 - 1
    tex = mesh.get_texture.unsqueeze(0).clamp(0, 1)
    rgb = F.grid_sample(tex, grid, align_corners=True, mode='bilinear', padding_mode='border')[0]
    
    # Depth and normal (normal in camera view space, will be converted to world space during training)
    # View space coordinates (using mesh vertices)
    ones = torch.ones((verts_world.shape[0], 1), device=device, dtype=verts_world.dtype)
    v4_view = torch.cat([verts_world, ones], dim=1) @ view
    verts_view = v4_view[:, :3]
    v4_view2 = torch.cat([verts_world, ones], dim=1) @ view.t()
    verts_view2 = v4_view2[:, :3]
    dist_attr = verts_view2.norm(dim=1, keepdim=True).unsqueeze(0).contiguous() 
    phys_depth = dr.interpolate(dist_attr, rast, faces)[0][..., 0]
    
    # Vertex normals
    vnorm = _compute_vertex_normals(verts_view, faces) 
    vnorm_attr = vnorm.unsqueeze(0) 
    vnorm_attr = vnorm_attr.contiguous()
    rend_normal = dr.interpolate(vnorm_attr, rast, faces)[0]
    if rend_normal.dim() == 4:
        rend_normal = rend_normal.squeeze(0)
    rend_normal = F.normalize(rend_normal, dim=2)
    rend_normal = rend_normal.permute(2, 0, 1)
    # Convert to world coordinates
    Rvw = viewpoint_camera.world_view_transform[:3, :3]
    rend_normal = (rend_normal.permute(1, 2, 0) @ Rvw.T).permute(2, 0, 1)

    # Background composition and alpha/mask
    alpha = rast[..., 3:].clamp(0, 1)
    alpha_aa = dr.antialias(alpha, rast, pos_clip, faces)
    if alpha_aa.dim() == 4:
        alpha_aa = alpha_aa.squeeze(0).squeeze(-1)
    alpha_aa = alpha_aa.unsqueeze(0)
    render_rgb = rgb * alpha_aa + bg_color.view(3, 1, 1) * (1 - alpha_aa)

    # Get per-face constant attribute for pixel triangle ID, and count pixel coverage (image_size)
    M = faces.shape[0]
    # For each triangle, construct constant attribute: attr[i] = i + 1 (offset by 1 to avoid background 0 conflict)
    face_ids_attr = (torch.arange(M, device=device, dtype=verts_world.dtype) + 1).view(1, M, 1)
    idx = torch.arange(M, device=device, dtype=torch.int32)
    face_ids_tri = torch.stack([idx, idx, idx], dim=1)
    # Interpolate to get per-pixel face ID (float constant), 0 means background
    tri_id_map = dr.interpolate(face_ids_attr, rast, face_ids_tri)[0][..., 0].squeeze(0)
    covered_mask = (alpha_aa[0] > 0)
    tri_ids_float = tri_id_map[covered_mask]
    # Only count pixels > 0 (remove background and fractional boundary)
    tri_ids = tri_ids_float[tri_ids_float > 0].round().to(torch.int64) - 1 
    in_range = (tri_ids >= 0) & (tri_ids < M)
    tri_ids = tri_ids[in_range]
    image_size = torch.bincount(tri_ids, minlength=M).to(verts_world.dtype)    

    
    return {
        "render": render_rgb,          # (3,H,W)
        "rend_alpha": alpha_aa,           # (1,H,W)
        "rend_normal": rend_normal,    # (3,H,W)
        "rend_depth": phys_depth,      # (1,H,W) 
        "scaling": image_size,
    }
 