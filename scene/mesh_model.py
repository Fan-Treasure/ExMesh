import open3d as o3d
import os
import torch
import torch.nn as nn
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.graphics_utils import BasicMesh, extract_uv_map, vertex_color_to_uvmap, uvmap_to_vertex_color
from collections import Counter, defaultdict, deque
import math

# Ensure access to built-in functions for linting
min, max, int, len, set, enumerate, list, hasattr, tuple, sorted, range, print = (
    min, max, int, len, set, enumerate, list, hasattr, tuple, sorted, range, print
)


class MeshModel:    
    def setup_functions(self):
        """
        Setup activation functions for mesh optimization.
        """
        self.exponential_activation = lambda x: 0.01 + torch.exp(x)
        self.inverse_exponential_activation = lambda y: torch.log(y - 0.01)
    
    def __init__(self, sh_degree=3):
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0
        # Geometry attributes
        self._vertices = torch.empty(0)          # Vertex coordinates (N, 3)
        self._faces = torch.empty(0)             # Face indices (M, 3)
        self._vertices_color = torch.empty(0)    # Vertex colors (N, 3)
        self._uvs = torch.empty(0)               # UV coordinates (K, 2), K>=N
        self._uv_indices = torch.empty(0)        # UV face indices (M, 3)
        self._vmapping = torch.empty(0)          # Vertex to UV mapping (K,)
        self._texture_mask = torch.empty(0)      # Mask for covered UV pixels
        # Rendering attributes
        self._texture = torch.empty(0)           # Texture map (H, W, 3)
        # Optimization attributes
        self.optimizer = None
        self.spatial_lr_scale = 0                # Spatial learning rate scale
        self._number_of_faces = 0
        self.split_size = 0                      # Triangle split threshold
        self.image_size = 0                      # Projected area per face
        self.degeneracy_ratio = 0                # Degeneracy ratio per face
        self.mask_grad_ema = 0                   # EMA of vertex mask gradients
        self.setup_functions()                   # Initialize activation functions
    
    @property
    def get_vertices(self):
        """
        Return vertex coordinates tensor.
        """
        return self._vertices
    
    @property
    def get_faces(self):
        """
        Return face indices tensor.
        """
        return self._faces
    
    @property
    def get_uvs(self):
        """
        Return UV coordinates tensor.
        """
        return self._uvs

    @property
    def get_texture(self):
        """
        Return texture parameter tensor.
        """
        return self._texture
    
    @property
    def get_uv_indices(self):
        """
        Return UV face indices tensor.
        """
        return self._uv_indices

    @property
    def get_vmapping(self):
        """
        Return vertex-to-UV mapping tensor.
        """
        return self._vmapping

    @property
    def get_texture_mask(self):
        """
        Return texture mask tensor.
        """
        return self._texture_mask

    def create_from_mesh(self, mesh: BasicMesh, texture_resolution, bg_color):
        """
        Initialize MeshModel parameters from a mesh file.
        Args:
            mesh: mesh object with vertices, faces, uvs, etc.
            texture_resolution: (H, W) tuple for texture size
            bg_color: background color value
        """
        self._vertices = torch.nn.Parameter(torch.tensor(mesh.vertices, dtype=torch.float32, device="cuda").requires_grad_(True))
        self._faces = torch.tensor(mesh.faces, dtype=torch.long, device="cuda")
        self._vertices_color = torch.tensor(mesh.vertex_colors, dtype=torch.float32, device="cuda")
        self._uvs = torch.tensor(mesh.uvs, dtype=torch.float32, device="cuda")
        self._uv_indices = torch.tensor(mesh.uv_indices, dtype=torch.long, device="cuda")
        self._vmapping = torch.tensor(mesh.vmapping, dtype=torch.long, device="cuda")
        H, W = texture_resolution
        texture_np, mask_np = vertex_color_to_uvmap(mesh.vertices, mesh.faces, mesh.uvs, mesh.vertex_colors, mesh.uv_indices, mesh.vmapping, (H, W), bg_color)
        self._texture_mask = torch.tensor(mask_np, dtype=torch.float32, device="cuda")
        self._texture = torch.nn.Parameter(torch.tensor(texture_np, dtype=torch.float32, device="cuda").requires_grad_(True))
        self.image_size = torch.zeros((self._faces.shape[0]), device="cuda")
        self.degeneracy_ratio = torch.zeros((self._faces.shape[0]), device="cuda")
        self._number_of_faces = self._faces.shape[0]
        self.mask_grad_ema = torch.zeros((self._vertices.shape[0]), device="cuda")
    
    def save(self, path):
        """
        Save MeshModel parameters and hyperparameters to the specified path.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        mesh_state_dict = {
            "vertices": self._vertices.detach().cpu(),
            "faces": self._faces.detach().cpu(),
            "uvs": self._uvs.detach().cpu(),
            "uv_indices": self._uv_indices.detach().cpu(),
            "vmapping": self._vmapping.detach().cpu(),
            "texture": self._texture.detach().cpu(),
            "texture_mask": self._texture_mask.detach().cpu(),
        }
        torch.save(mesh_state_dict, os.path.join(path, 'model_state_dict.pt'))

        hyperparameters = {
            "spatial_lr_scale": self.spatial_lr_scale,
        }
        torch.save(hyperparameters, os.path.join(path, 'hyperparameters.pt'))

    def load(self, path):
        mesh_state_dict = torch.load(os.path.join(path, 'model_state_dict.pt'))
        self._vertices = nn.Parameter(mesh_state_dict["vertices"].to("cuda").detach().clone().requires_grad_(True))
        self._faces = mesh_state_dict["faces"].to("cuda")
        self._uvs = mesh_state_dict["uvs"].to("cuda")
        self._uv_indices = mesh_state_dict["uv_indices"].to("cuda")
        self._vmapping = mesh_state_dict["vmapping"].to("cuda")
        self._texture = nn.Parameter(mesh_state_dict["texture"].to("cuda").detach().clone().requires_grad_(True))
        self._texture_mask = mesh_state_dict["texture_mask"].to("cuda")

        # Build optimizer for vertices and texture only
        param_groups = [
            {'params': [self._vertices], 'lr': 0.00001, "name": "vertices"},
            {'params': [self._texture], 'lr': 0.00001, "name": "texture"},
        ]
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
    
    def save_as_ply(self, save_path):
        """
        Export mesh as a PLY file, with vertex colors sampled from the UV texture map.
        """
        vertices_np = self._vertices.detach().cpu().numpy()
        faces_np = self._faces.cpu().numpy()
        vertex_colors = uvmap_to_vertex_color(self._vertices, self._uvs, self._texture, self._vmapping, self._texture_mask)
        vertex_colors = np.clip(vertex_colors, 0, 1)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
        mesh.triangles = o3d.utility.Vector3iVector(faces_np)
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        o3d.io.write_triangle_mesh(save_path, mesh)
    
    def training_setup(self, training_args, lr_features, lr_vertices_init):
        self.split_size = training_args.split_size
        self.add_shape = training_args.add_shape
        param_groups = [
            {'params': [self._vertices], 'lr': lr_vertices_init, "name": "vertices"},
            {'params': [self._texture], 'lr': lr_features, "name": "texture"}
        ]
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        self.triangle_scheduler_args = get_expon_lr_func(lr_init=lr_vertices_init,
                                                        lr_final=lr_vertices_init / 100,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.position_lr_max_steps)

    def compute_degeneracy_ratio(self, eps: float = 1e-8):
        """
        Compute degeneracy ratio for each face: area / (max edge length^2).
        Lower values indicate more degenerate faces. Returns tensor of shape [M].
        """
        if self._faces.numel() == 0 or self._vertices.numel() == 0:
            dev = self._vertices.device if isinstance(self._vertices, torch.Tensor) and self._vertices.numel() > 0 else "cuda"
            return torch.zeros((0,), device=dev)

        faces = self.get_faces  # (M, 3)
        verts = self.get_vertices  # (V, 3)

        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]

        e0 = v1 - v0
        e1 = v2 - v1
        e2 = v0 - v2

        l0 = torch.norm(e0, dim=1)
        l1 = torch.norm(e1, dim=1)
        l2 = torch.norm(e2, dim=1)
        lmax = torch.maximum(l0, torch.maximum(l1, l2))

        area = 0.5 * torch.norm(torch.cross(e0, v2 - v0, dim=1), dim=1)

        ratio = area / (lmax * lmax + eps)
        self.degeneracy_ratio = ratio.detach()
        return ratio

    def compute_face_areas(self):
        """
        Compute geometric area for all faces.
        """
        faces = self.get_faces
        verts = self.get_vertices 

        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        areas = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)
        return areas

    def update_learning_rate(self, iteration):
        """
        Update vertex learning rate according to scheduler for each iteration.
        Returns current learning rate.
        """
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "vertices":
                lr = self.triangle_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def merge_close_vertex(self, dead_mask, length_ratio_thresh=0.1):
        """
        Merge vertices for faces with degeneracy below threshold.
        For each candidate face, merge the shortest edge if its length ratio is below threshold.
        Args:
            dead_mask: Boolean mask for degenerate faces.
            length_ratio_thresh: Threshold for shortest edge length ratio.
        Returns:
            Number of successful merges.
        """     
        device = self._faces.device
        faces = self._faces
        V = self._vertices.shape[0]

        # Tolerance parameters for angle/distance/projection
        cos_thr_angle = 2
        cos_thr = math.cos(math.radians(90 - cos_thr_angle))
        right_angle_thr_angle = 5
        right_angle_thr = math.cos(math.radians(90 - right_angle_thr_angle))

        # Collect candidate faces to process
        candidate_faces_idx = torch.nonzero(dead_mask, as_tuple=True)[0]
        if candidate_faces_idx.numel() == 0:
            return 0
        verts = self._vertices

        faces_cpu = faces.detach().cpu()
        one_ring = [set() for _ in range(V)]
        for f in faces_cpu:
            i, j, k = int(f[0]), int(f[1]), int(f[2])
            one_ring[i].update([j, k])
            one_ring[j].update([i, k])
            one_ring[k].update([i, j])

        # Build merge plans for candidate faces
        merge_plans_all = []  # list of tuples (del_v, keep_v)

        for fi in candidate_faces_idx.tolist():
            # For each candidate face, check if the shortest edge ratio is below threshold
            tri = faces[fi]
            v0, v1, v2 = tri[0].item(), tri[1].item(), tri[2].item()
            p0 = verts[v0]
            p1 = verts[v1]
            p2 = verts[v2]

            # Compute cosines of triangle angles
            def cos_angle(a, b, c):
                ba = b - a
                ca = c - a
                return torch.dot(ba, ca) / (torch.norm(ba) * torch.norm(ca) + 1e-12)
            cos0 = cos_angle(p0, p1, p2)
            cos1 = cos_angle(p1, p2, p0)
            cos2 = cos_angle(p2, p0, p1)
            cos_list = [float(cos0), float(cos1), float(cos2)]
            abs_cos_list = [abs(c) for c in cos_list]

            # Determine merge vertices A and B
            # Skip if triangle is too sharp
            if min(cos_list) > right_angle_thr:
                continue

            # If nearly right triangle, select right angle vertex as B
            if min(abs_cos_list) < right_angle_thr:
                min_idx = abs_cos_list.index(min(abs_cos_list))
                B = [v0, v1, v2][min_idx]
                merge_type = "right"
            else:
                # If obtuse, select vertex with smaller angle on shortest edge as B
                edges = [(torch.norm(p1 - p0).item(), (v0, v1), (cos_list[0], cos_list[1])),
                        (torch.norm(p2 - p1).item(), (v1, v2), (cos_list[1], cos_list[2])),
                        (torch.norm(p0 - p2).item(), (v2, v0), (cos_list[2], cos_list[0]))]
                edges.sort(key=lambda x: x[0])
                min_len, (va, vb), (cos_a, cos_b) = edges[0]
                B = vb if cos_a < cos_b else va
                merge_type = "obtuse"
                perim = (edges[0][0] + edges[1][0] + edges[2][0])
                if (min_len / perim) >= float(length_ratio_thresh) and min(cos_list) >= -0.95:
                    continue  # Skip if shortest edge not short enough or not obtuse
        
            # Get endpoints of shortest edge
            edges = [(torch.norm(p1 - p0).item(), (v0, v1)),
                     (torch.norm(p2 - p1).item(), (v1, v2)),
                     (torch.norm(p0 - p2).item(), (v2, v0))]
            edges.sort(key=lambda x: x[0])
            min_len, (va, vb) = edges[0]
            # A is the endpoint of shortest edge that is not B
            A = va if vb == B else vb

            # For A and B, find one-ring neighbors and check if any pair (A1, B1) forms a nearly planar quad with A, B
            A_neighbors = [n for n in one_ring[A] if n != B and n != A]
            B_neighbors = [n for n in one_ring[B] if n != A and n != B]
            if len(A_neighbors) == 0 or len(B_neighbors) == 0:
                continue  # Skip if no valid neighbors

            chosen = None  # Store chosen neighbor pair
            pA = verts[A]
            pB = verts[B]

            # Batch geometry checks for all (A1, B1) neighbor pairs
            A1_idx = torch.tensor(A_neighbors, device=verts.device, dtype=torch.long)
            B1_idx = torch.tensor(B_neighbors, device=verts.device, dtype=torch.long)
            pA1 = verts[A1_idx][:, None, :]  # (Na,1,3)
            pB1 = verts[B1_idx][None, :, :]  # (1,Nb,3)
            seg = pB1 - pA1                   # (Na,Nb,3)
            seg_len_t = torch.norm(seg, dim=2)             # (Na,Nb)
            valid_seg = seg_len_t > 1e-9

            # Compute normal of triangle (A1, B1, B)
            tri1_normal = torch.cross(pB1 - pA1, pB - pA1, dim=2)  # (Na, Nb, 3)
            tri1_normal = tri1_normal / (torch.norm(tri1_normal, dim=2, keepdim=True) + 1e-12)
            vec_AA1 = pA - pA1  # (Na, Nb, 3)
            cos_AA1_tri1 = torch.abs((vec_AA1 * tri1_normal).sum(dim=2) / (torch.norm(vec_AA1, dim=2) * torch.norm(tri1_normal, dim=2) + 1e-12))

            # Compute normal of triangle (A1, B1, A)
            tri2_normal = torch.cross(pB1 - pA1, pA - pA1, dim=2)  # (Na, Nb, 3)
            tri2_normal = tri2_normal / (torch.norm(tri2_normal, dim=2, keepdim=True) + 1e-12)
            vec_BB1 = pB - pB1  # (Na, Nb, 3)
            cos_BB1_tri2 = torch.abs((vec_BB1 * tri2_normal).sum(dim=2) / (torch.norm(vec_BB1, dim=2) * torch.norm(tri2_normal, dim=2) + 1e-12))
            # Check planarity by cosine values
            mask = (valid_seg & (torch.abs(cos_AA1_tri1) <= cos_thr) & (torch.abs(cos_BB1_tri2) <= cos_thr))

            if mask.any():
                # Select neighbor pair with minimal score
                score = torch.where(mask, cos_AA1_tri1 + cos_BB1_tri2, torch.full_like(cos_AA1_tri1, 1e9))
                min_flat_idx = torch.argmin(score.view(-1))
                ia = (min_flat_idx // score.shape[1]).item()
                ib = (min_flat_idx % score.shape[1]).item()
                chosen = (A1_idx[ia].item(), B1_idx[ib].item())

            if chosen is None:
                continue
            
            keep_v, del_v = B, A
            merge_plans_all.append((del_v, keep_v, merge_type))

        # Sort merge plans: prioritize right triangles
        merge_plans = [x[:2] for x in merge_plans_all if x[2] == "right"] + [x[:2] for x in merge_plans_all if x[2] == "obtuse"]
        if len(merge_plans) == 0:
            return 0

        # Apply merge plans, skip if any face already deleted or vertices conflict
        faces_current = self._faces
        affected_vertices = set()
        faces_to_delete = set()
        face_replacements = {}
        applied_pairs = []  # Track applied (del_v, keep_v)

        # Build vertex-to-face mapping for fast lookup
        v2faces = [[] for _ in range(V)]
        for idx, f in enumerate(faces_cpu.tolist()):
            i, j, k = f
            v2faces[i].append(idx)
            v2faces[j].append(idx)
            v2faces[k].append(idx)

        applied = 0
        for del_v, keep_v in merge_plans:
            # Collect all vertices in faces containing del_v
            related_vertices = set()
            for f_idx in v2faces[del_v]:
                if f_idx in faces_to_delete:
                    continue
                tri = faces_current[f_idx].tolist()
                related_vertices.update(tri)

            if related_vertices & affected_vertices:
                continue  # Skip if conflict with previous merges

            # Delete or replace faces containing del_v
            local_deleted = set()
            local_replacements = {}
            for f_idx in v2faces[del_v]:
                if f_idx in faces_to_delete:
                    continue
                tri = faces_current[f_idx]
                if (tri == keep_v).any():
                    local_deleted.add(f_idx)
                else:
                    new_face = torch.where(tri == del_v, torch.tensor(keep_v, device=device, dtype=tri.dtype), tri)
                    local_replacements[f_idx] = new_face

            faces_to_delete.update(local_deleted)
            face_replacements.update(local_replacements)
            affected_vertices.update(related_vertices)
            applied += 1
            applied_pairs.append((del_v, keep_v))

        if applied == 0:
            return 0

        # Update mesh parameters and build new faces/uvs
        new_faces_list = []
        new_uv_faces_list = []
        keep_face_indices = []
        for f_idx in range(faces_current.shape[0]):
            if f_idx in faces_to_delete:
                continue
            if f_idx in face_replacements:
                new_faces_list.append(face_replacements[f_idx])
            else:
                new_faces_list.append(faces_current[f_idx])
            # Keep uv indices in sync with faces
            new_uv_faces_list.append(self._uv_indices[f_idx])
            keep_face_indices.append(f_idx)

        self._faces = torch.stack(new_faces_list, dim=0)
        self._uv_indices = torch.stack(new_uv_faces_list, dim=0)

        # Remap vmapping from deleted to kept vertices
        if len(applied_pairs) > 0:
            redirect = torch.arange(V, device=device, dtype=torch.long)
            if isinstance(self._vmapping, torch.Tensor) and self._vmapping.numel() > 0:
                d = torch.tensor([p[0] for p in applied_pairs], device=device, dtype=torch.long)
                k = torch.tensor([p[1] for p in applied_pairs], device=device, dtype=torch.long)
                redirect[d] = k
                self._vmapping = redirect[self._vmapping]

        # Remove unused vertices and remap indices
        used_v = torch.unique(self._faces.flatten())
        old2new = torch.full((V,), -1, dtype=torch.long, device=device)
        old2new[used_v] = torch.arange(used_v.shape[0], device=device)

        self._vertices = nn.Parameter(self._vertices[used_v].detach().clone().requires_grad_(True))
        self._vertices_color = self._vertices_color[used_v]
        self.mask_grad_ema = self.mask_grad_ema[used_v]
        self._faces = old2new[self._faces]
        self._vmapping = old2new[self._vmapping]
        self._compact_uvs()

        # Refresh statistics; UV/optimizer rebuild handled by upper logic
        self._number_of_faces = self._faces.shape[0]
        keep_face_indices_tensor = torch.tensor(keep_face_indices, device=self.image_size.device, dtype=torch.long)
        self.image_size = self.image_size[keep_face_indices_tensor]
        self.degeneracy_ratio = torch.zeros((self._faces.shape[0]), device=device)

        print(f"[merge_close_vertex] Applied merge plans: {applied} | New face count: {self._faces.shape[0]} | New vertex count: {self._vertices.shape[0]}")
        return applied
        
    def merge_vertex(self, mask):
        """
        Merge vertices for faces selected by mask.
        For each face to be deleted, find the vertex A opposite the longest edge and merge A into the first vertex B of the longest edge.
        """
        faces_to_merge = torch.nonzero(mask, as_tuple=True)[0]
        if len(faces_to_merge) == 0:
            return

        point_A_list, point_B_list, edge_counts = self.select_merge_pairs(faces_to_merge)
        # Shuffle merge order
        order_idx = torch.randperm(faces_to_merge.shape[0], device=faces_to_merge.device)
        faces_to_merge = faces_to_merge[order_idx]
        point_A_list = point_A_list[order_idx]
        point_B_list = point_B_list[order_idx]
        
        # Track affected vertices to avoid repeated merges
        affected_vertices = set()
        faces_to_delete = set()  # Indices of faces to delete
        face_replacements = {}   # Face replacement mapping: original face idx -> new face
        # Process vertex merges one by one
        for i, face_idx in enumerate(faces_to_merge):
            face_idx = face_idx.item()
            point_A = point_A_list[i].item()
            point_B = point_B_list[i].item()

            # Find all vertices related to A (vertices in faces containing A)
            faces_with_A = (self._faces == point_A).any(dim=1)
            related_vertices = set()
            for f_idx in torch.nonzero(faces_with_A, as_tuple=True)[0]:
                related_vertices.update(self._faces[f_idx].tolist())

            # Skip if conflict with previous merges
            if related_vertices & affected_vertices:
                continue
            # Delete or replace faces containing A
            del_count = 0
            local_deleted = set()
            local_replacements = {}
            for f_idx in torch.nonzero(faces_with_A, as_tuple=True)[0]:
                f_idx = f_idx.item()
                face = self._faces[f_idx]
                if point_B in face:
                    local_deleted.add(f_idx)
                    del_count += 1
                else:
                    new_face = torch.where(face == point_A, point_B, face)
                    local_replacements[f_idx] = new_face
            
            # Check link condition for vertex collapse
            if not self._check_link_condition(point_A, point_B, local_deleted, self._faces):
                continue
            
            faces_to_delete.update(local_deleted)
            face_replacements.update(local_replacements)
            affected_vertices.update(related_vertices)
        
        # Print number of deleted boundary faces
        faces_all_boundary_mask = (edge_counts == 1).any(dim=1)
        faces_to_delete_list = list(faces_to_delete)
        faces_to_delete_tensor = torch.tensor(faces_to_delete_list, device=self._faces.device)
        num_boundary_deleted = faces_all_boundary_mask[faces_to_delete_tensor].sum().item()
        print(f"[merge_vertex] Deleted faces: {len(faces_to_delete)}, boundary faces: {num_boundary_deleted}")

        # Build new face list and attributes
        new_faces = []
        new_face_indices = []
        
        for f_idx in range(self._faces.shape[0]):
            if f_idx in faces_to_delete:
                continue  # Skip deleted faces
            elif f_idx in face_replacements:
                new_faces.append(face_replacements[f_idx])
                new_face_indices.append(f_idx)
            else:
                new_faces.append(self._faces[f_idx])
                new_face_indices.append(f_idx)

        if len(new_faces) == 0:
            print("[merge_vertex] Warning: all faces deleted.")
            return

        # Update face data
        self._faces = torch.stack(new_faces)
        # Remove unused vertices
        used_vertices = torch.unique(self._faces.flatten())
        old_to_new_map = torch.full((self._vertices.shape[0],), -1, dtype=torch.long, device=self._vertices.device)
        old_to_new_map[used_vertices] = torch.arange(len(used_vertices), device=self._vertices.device)
        self._vertices = nn.Parameter(self._vertices[used_vertices].detach().clone().requires_grad_(True))
        self._vertices_color = self._vertices_color[used_vertices]
        self._faces = old_to_new_map[self._faces]
        self._number_of_faces = self._faces.shape[0]
        
        print(f"[merge_vertex] Merges processed: {len(faces_to_merge)} -> Faces deleted: {len(faces_to_delete)}")
        
        # Update optimizer
        self._rebind_optimizer(vertex_mapping=used_vertices.detach().clone())

    def select_merge_pairs(self, faces_to_merge):
        """
        For each face to be merged, select merge vertex (A) and keep vertex (B).
        Returns:
            point_A_list: tensor of merge vertex indices
            point_B_list: tensor of keep vertex indices
        """
        selected_faces = self._faces[faces_to_merge]  # (N, 3)
        selected_vertices = self._vertices[selected_faces]  # (N, 3, 3)
        edge_lens = torch.stack([
            torch.norm(selected_vertices[:, 1] - selected_vertices[:, 0], dim=1),  # v0-v1
            torch.norm(selected_vertices[:, 2] - selected_vertices[:, 1], dim=1),  # v1-v2  
            torch.norm(selected_vertices[:, 0] - selected_vertices[:, 2], dim=1),  # v2-v0
        ], dim=1)  # (N, 3)

        longest_edge_indices = edge_lens.argmax(dim=1)  # (N,)
        default_A_pos = torch.where(
            longest_edge_indices == 0,
            torch.full_like(longest_edge_indices, 2),
            torch.where(longest_edge_indices == 1,
                        torch.zeros_like(longest_edge_indices),
                        torch.ones_like(longest_edge_indices))
        )  # (N,) in {0,1,2}

        # Compute boundary info
        faces_all = self._faces
        edges = torch.stack([faces_all[:, [0,1]], faces_all[:, [1,2]], faces_all[:, [2,0]]], dim=1)
        edges = torch.sort(edges, dim=2).values
        flat_edges = edges.reshape(-1, 2)
        uniq_edges, inv, counts = torch.unique(flat_edges, dim=0, return_inverse=True, return_counts=True)
        edge_counts = counts[inv].reshape(-1, 3)
        sel_edge_counts = edge_counts[faces_to_merge]
        boundary_edge_mask = (sel_edge_counts == 1)
        b0 = boundary_edge_mask[:, 0] | boundary_edge_mask[:, 2]
        b1 = boundary_edge_mask[:, 0] | boundary_edge_mask[:, 1]
        b2 = boundary_edge_mask[:, 1] | boundary_edge_mask[:, 2]
        boundary_vertex_mask = torch.stack([b0, b1, b2], dim=1)
        sel_is_boundary = boundary_vertex_mask.any(dim=1)
        default_is_boundary = boundary_vertex_mask.gather(1, default_A_pos.unsqueeze(1)).squeeze(1).bool()
        opp_len_per_vertex = edge_lens[:, [1, 2, 0]].clone()
        opp_len_per_vertex[~boundary_vertex_mask] = -1e9
        best_boundary_A_pos = opp_len_per_vertex.argmax(dim=1)
        use_override = sel_is_boundary & (~default_is_boundary)
        A_pos_final = torch.where(use_override, best_boundary_A_pos, default_A_pos)

        arangeN = torch.arange(selected_faces.shape[0], device=selected_faces.device)
        point_A_list = selected_faces[arangeN, A_pos_final]

        candB1_pos = (A_pos_final + 1) % 3
        candB2_pos = (A_pos_final + 2) % 3
        candidate_B1 = selected_faces[arangeN, candB1_pos]
        candidate_B2 = selected_faces[arangeN, candB2_pos]
        all_edges = torch.cat([self._faces[:, [0, 1]], self._faces[:, [1, 2]], self._faces[:, [2, 0]]], dim=0).flatten()
        degrees = torch.bincount(all_edges, minlength=self._vertices.shape[0])
        deg_B1 = degrees[candidate_B1]
        deg_B2 = degrees[candidate_B2]
        pick_B1 = deg_B1 <= deg_B2
        point_B_list = torch.where(pick_B1, candidate_B1, candidate_B2)

        return point_A_list, point_B_list, edge_counts

    def _check_link_condition(self, point_A, point_B, local_deleted, faces):
        """
        Check the Link condition for vertex collapse (A -> B).
        Returns True if the collapse is allowed, False otherwise.
        """
        NA = set()
        faces_with_A_mask_all = (faces == point_A).any(dim=1)
        for fi in torch.nonzero(faces_with_A_mask_all, as_tuple=True)[0]:
            tri = faces[fi].tolist()
            for v in tri:
                if v != point_A and v != point_B:
                    NA.add(v)

        NB = set()
        faces_with_B_mask_all = (faces == point_B).any(dim=1)
        for fi in torch.nonzero(faces_with_B_mask_all, as_tuple=True)[0]:
            tri = faces[fi].tolist()
            for v in tri:
                if v != point_B and v != point_A:
                    NB.add(v)

        thirds = set()
        for fi in local_deleted:
            tri = faces[fi].tolist()
            others = [v for v in tri if v not in (point_A, point_B)]
            if len(others) == 1:
                thirds.add(others[0])

        overlap = (NA & NB) - thirds
        return len(overlap) == 0
    
    def _reset_optimizer(self):
        import gc
        lr_vertices = self.optimizer.param_groups[0]['lr']
        lr_texture = self.optimizer.param_groups[1]['lr']
        del self.optimizer
        gc.collect()
        torch.cuda.empty_cache()
        param_groups = [
            {'params': [self._vertices], 'lr': lr_vertices, "name": "vertices"},
            {'params': [self._texture], 'lr': lr_texture, "name": "texture"}
        ]
        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

    def _rebind_optimizer(self, vertex_mapping: torch.Tensor | None = None):
        """
        Rebind optimizer parameter handles to current self._vertices and self._texture, preserving momentum state as much as possible.
        """
        def _rebind_one_group(group, new_param: nn.Parameter, mapping: torch.Tensor | None):
            old_param = group["params"][0]
            old_state = self.optimizer.state.pop(old_param, None)
            group["params"][0] = new_param

            if old_state is None:
                return

            new_state = {}
            for k, v in old_state.items():
                if torch.is_tensor(v) and v.shape == old_param.shape:
                    # Copy momentum buffer if shape matches
                    v = v.to(new_param.device)
                    if mapping is None:
                        if new_param.shape == old_param.shape:
                            new_buf = v.detach().clone()
                        else:
                            # If shape changed and no mapping, copy prefix and zero out extra
                            new_buf = torch.zeros_like(new_param)
                            copy = min(v.shape[0], new_param.shape[0])
                            new_buf[:copy] = v[:copy]
                    else:
                        # Use new->old mapping to copy, set -1 positions to zero
                        assert mapping.shape[0] == new_param.shape[0]
                        new_buf = torch.zeros_like(new_param)
                        keep = mapping >= 0
                        if keep.any():
                            src_idx = mapping[keep].to(v.device, dtype=torch.long)
                            new_buf[keep] = v[src_idx]
                    new_state[k] = new_buf
                else:
                    # Preserve non-tensor or differently shaped states (e.g., step count, amsgrad vhat)
                    new_state[k] = v
            self.optimizer.state[new_param] = new_state

        # Vertex parameter
        if not isinstance(self._vertices, nn.Parameter):
            self._vertices = nn.Parameter(self._vertices.requires_grad_(True))
        for group in self.optimizer.param_groups:
            if group.get("name") == "vertices":
                _rebind_one_group(group, self._vertices, vertex_mapping)

        # Texture parameter (usually shape unchanged, preserve fully)
        if not isinstance(self._texture, nn.Parameter):
            self._texture = nn.Parameter(self._texture.requires_grad_(True))
        for group in self.optimizer.param_groups:
            if group.get("name") == "texture":
                _rebind_one_group(group, self._texture, mapping=None)

    def add_new_face(self, cap_max, dead_mask=None):
        """
        Dynamically add new triangles to densify the mesh (split faces).
        For each split, one new vertex is added, two faces are deleted, and four faces are created.
        Args:
            cap_max: Maximum number of triangles allowed after addition.
            dead_mask: Boolean mask for dead faces (shape [M]).
        Returns:
            new_dead_mask: Updated dead mask for all faces (shape [M']).
        """
        current_num_faces = self._faces.shape[0]
        old_vertices_count = self._vertices.shape[0]

        # Select faces to split
        add_idx, valid_top_mask = self.sample_new_faces(cap_max, dead_mask)
        if valid_top_mask is None:
            return dead_mask

        # Split faces and get new mesh components
        (new_vertex, del_idx, new_faces, new_vertices_color, new_uvs, new_uv_faces, new_uv_vmapping) = \
            self._split_faces(add_idx, dead_mask, valid_top_mask)

        # Remove split faces and update attributes
        keep_mask = torch.ones(current_num_faces, dtype=torch.bool, device=self._faces.device)
        keep_mask[del_idx] = False
        self._faces = self._faces[keep_mask]
        self._uv_indices = self._uv_indices[keep_mask]
        new_dead_mask = dead_mask[keep_mask]
        kept_image_size = self.image_size[keep_mask]

        # Concatenate new faces and vertices
        self._faces = torch.cat([self._faces, new_faces], dim=0)
        self._vertices = nn.Parameter(torch.cat([self._vertices.detach(), new_vertex], dim=0).requires_grad_(True))
        self._vertices_color = torch.cat([self._vertices_color, new_vertices_color], dim=0)
        self.mask_grad_ema = torch.zeros((self._vertices.shape[0]), device="cuda")
        self._number_of_faces = self._faces.shape[0]
        self.image_size = torch.cat([kept_image_size, torch.ones(new_faces.shape[0], device="cuda")], dim=0)
        self.degeneracy_ratio = torch.zeros((self._faces.shape[0]), device="cuda")
        self._uvs = torch.cat([self._uvs, new_uvs], dim=0)
        self._uv_indices = torch.cat([self._uv_indices, new_uv_faces], dim=0)
        self._vmapping = torch.cat([self._vmapping, new_uv_vmapping], dim=0)

        # Rebind optimizer momentum buffers for new vertices
        V_old = old_vertices_count
        V_new = self._vertices.shape[0]
        mapping = torch.cat([
            torch.arange(V_old, device=self._vertices.device, dtype=torch.long),
            torch.full((V_new - V_old,), -1, device=self._vertices.device, dtype=torch.long)
        ], dim=0)
        self._rebind_optimizer(vertex_mapping=mapping)

        # Expand dead_mask for new faces (default to False)
        new_face_count = new_faces.shape[0]
        if new_face_count > 0:
            pad = torch.zeros(new_face_count, dtype=torch.bool, device=new_dead_mask.device)
            new_dead_mask = torch.cat([new_dead_mask, pad], dim=0)
        self._compact_uvs()

        print(f"[add_new_face] Added triangles: {new_faces.shape[0]-del_idx.shape[0]}, total triangles: {self._faces.shape[0]}")

    def sample_new_faces(self, cap_max, dead_mask):
        """
        Sample indices of faces to be split for mesh densification.
        Returns:
            add_idx: indices of faces to split
            valid_top_mask: mask for top 50% area faces
        """
        current_num_faces = self._faces.shape[0]
        target_num = min(cap_max, int(current_num_faces * 1.25))
        num_face = max(0, target_num - current_num_faces)
        if num_face <= 0:
            return torch.empty(0, dtype=torch.long, device=self._faces.device), None

        areas = self.compute_face_areas()
        _, area_indices = torch.sort(areas, descending=True)
        num_total = areas.shape[0]
        num_top10 = max(1, num_total // 10)
        num_top50 = num_total // 2
        top10_indices = area_indices[:num_top10]
        mid_indices = area_indices[num_top10:num_top50]

        top10_mask = torch.zeros_like(areas, dtype=torch.bool)
        top10_mask[top10_indices] = True
        mid_mask = torch.zeros_like(areas, dtype=torch.bool)
        mid_mask[mid_indices] = True
        top10_mask[dead_mask] = False
        mid_mask[dead_mask] = False

        top10_candidates = torch.nonzero(top10_mask, as_tuple=True)[0]
        num_top10_pick = min(num_face, top10_candidates.shape[0])
        chosen_top10 = top10_candidates[torch.randperm(top10_candidates.shape[0], device=areas.device)[:num_top10_pick]]

        num_remain = num_face - num_top10_pick
        if num_remain > 0 and mid_mask.sum() > 0:
            mid_probs = areas.clone()
            mid_probs[~mid_mask] = 0
            mid_probs = mid_probs / (mid_probs.sum() + 1e-8)
            chosen_mid = torch.multinomial(mid_probs, min(num_remain, mid_mask.sum().item()), replacement=False)
        else:
            chosen_mid = torch.empty(0, dtype=torch.long, device=areas.device)

        add_idx = torch.cat([chosen_top10, chosen_mid], dim=0)
        valid_indices = area_indices[:num_top50]
        valid_top_mask = torch.zeros_like(areas, dtype=torch.bool)
        valid_top_mask[valid_indices] = True

        probs = torch.zeros_like(areas)
        probs[valid_indices] = areas[valid_indices]
        probs[dead_mask] = 0
        probs = probs / (probs.sum() + 1e-8)
        add_idx = torch.multinomial(probs, min(num_face, current_num_faces), replacement=False)

        return add_idx, valid_top_mask

    def _split_faces(self, face_indices, dead_mask, valid_top_mask):
        """
        Split specified faces into smaller faces for mesh densification.
        For each face, if adjacent faces are in the top 50% by area, perform a split and update mesh attributes.
        """
        MAX_DEGREE = 10  # Maximum allowed vertex degree
        face_indices = torch.tensor(face_indices, dtype=torch.long, device=self._vertices.device)
        # Build edge-to-face mapping
        edge2faces = defaultdict(list)
        faces_cpu = self._faces.cpu()
        for idx, face in enumerate(faces_cpu):
            for i in range(3):
                e = tuple(sorted((face[i].item(), face[(i+1)%3].item())))
                edge2faces[e].append(idx)

        # Count vertex degrees
        all_indices = self._faces.view(-1)
        vertex_degrees = torch.bincount(all_indices, minlength=self._vertices.shape[0])

        valid_top_mask_cpu = valid_top_mask.detach().cpu()
        new_vertices = []
        new_faces = []
        new_vertices_color = []
        del_face_indices = set()
        base_vertex_count = self._vertices.shape[0]
        new_uvs = []
        new_uv_faces = []
        new_uv_vmapping = []
        
        def get_adaptive_point_on_edge(A, B, C, D=None):
            """
            Compute interpolation parameter t for point on edge AB.
            If D is None (boundary face), use projection of C onto AB.
            If D is not None, use average projection of C and D onto AB.
            """
            AB = B - A
            AB_len2 = (AB * AB).sum()
            if AB_len2 < 1e-12:
                return 0.5, True  # Use midpoint for degenerate edge
            t1 = ((C - A) * AB).sum() / AB_len2
            if D is None:
                t = t1
            else:
                t2 = ((D - A) * AB).sum() / AB_len2
                t = 0.5 * (t1 + t2)
            if not (0.27 <= t <= 0.73):
                return t, False
            return t, True
        
        for i, fidx in enumerate(face_indices):
            fidx = fidx.item()
            if dead_mask is not None and bool(dead_mask[fidx].item()):
                continue
            if fidx in del_face_indices:
                continue
            face = self._faces[fidx]

            # Get vertex indices and degrees
            v0, v1, v2 = face[0].item(), face[1].item(), face[2].item()
            deg0 = vertex_degrees[v0].item()
            deg1 = vertex_degrees[v1].item()
            deg2 = vertex_degrees[v2].item()
            p0 = self._vertices[v0]; p1 = self._vertices[v1]; p2 = self._vertices[v2]

            # Build edge info for splitting
            edges_info = [
                {'vA': v0, 'vB': v1, 'vC': v2, 'len': torch.norm(p1 - p0).item(), 'opp_deg': deg2},
                {'vA': v1, 'vB': v2, 'vC': v0, 'len': torch.norm(p2 - p1).item(), 'opp_deg': deg0},
                {'vA': v2, 'vB': v0, 'vC': v1, 'len': torch.norm(p0 - p2).item(), 'opp_deg': deg1}
            ]

            # Select best edge to split
            best_edge = None
            best_score = -1.0
            for e in edges_info:
                if e['opp_deg'] >= MAX_DEGREE:
                    continue
                score = e['len'] / (pow(e['opp_deg'], 1.0) + 1e-6)
                if score > best_score:
                    best_score = score
                    best_edge = e

            if best_edge is None:
                continue
            vA = best_edge['vA']
            vB = best_edge['vB']
            vC = best_edge['vC']
            pA = self._vertices[vA]; pB = self._vertices[vB]; pC = self._vertices[vC]

            # Find adjacent face sharing edge vA-vB
            edge = tuple(sorted((vA, vB)))
            all_adj_faces = edge2faces[edge]
            adj_faces_valid = [f for f in all_adj_faces if f != fidx and f not in del_face_indices]
            adj_faces_valid = [f for f in adj_faces_valid if bool(valid_top_mask_cpu[f].item())]

            if len(adj_faces_valid) == 0:
                # If only one adjacent face (boundary), do 2-split
                if len(all_adj_faces) == 1:
                    idx_mid = base_vertex_count + len(new_vertices)
                    t, valid = get_adaptive_point_on_edge(pA, pB, pC)
                    if not valid:
                        continue
                    mid = (1 - t) * pA + t * pB
                    new_vertices.append(mid)
                    new_faces.append([vA, idx_mid, vC])
                    new_faces.append([idx_mid, vB, vC])
                    mid_color = (self._vertices_color[vA] + self._vertices_color[vB]) / 2
                    new_vertices_color.append(mid_color)
                    del_face_indices.add(fidx)

                    uv_tri = self._uv_indices[fidx]
                    uA = uv_tri[edges_info.index(best_edge)].item()
                    uB = uv_tri[(edges_info.index(best_edge)+1)%3].item()
                    uC = uv_tri[(edges_info.index(best_edge)+2)%3].item()
                    uvA = self._uvs[uA]; uvB = self._uvs[uB]; uvC = self._uvs[uC]
                    uv_mid = (1 - t) * uvA + t * uvB
                    u_mid = self._uvs.shape[0] + len(new_uvs)
                    new_uvs.append(uv_mid)
                    new_uv_vmapping.append(idx_mid)
                    new_uv_faces.append([uA, u_mid, uC])
                    new_uv_faces.append([u_mid, uB, uC])
                    continue
                else:
                    continue

            # Standard 4-split
            adj_fidx = adj_faces_valid[0]
            adj_face = self._faces[adj_fidx]
            adj_face_cpu = adj_face.cpu()
            vD = [v.item() for v in adj_face_cpu if v.item() not in (vA, vB)][0]
            pD = self._vertices[vD]
            if vertex_degrees[vD].item() >= MAX_DEGREE:
                continue
            t, valid = get_adaptive_point_on_edge(pA, pB, pC, pD)
            if not valid:
                continue
            mid = (1 - t) * pA + t * pB

            idx_mid = base_vertex_count + len(new_vertices)
            del_face_indices.update([fidx, adj_fidx])
            new_vertices.append(mid)
            new_faces.append([vA, idx_mid, vC])
            new_faces.append([idx_mid, vB, vC])
            new_faces.append([vA, vD, idx_mid])
            new_faces.append([idx_mid, vD, vB])
            mid_color = (self._vertices_color[vA] + self._vertices_color[vB]) / 2
            new_vertices_color.append(mid_color)

            uv_tri0 = self._uv_indices[fidx]
            uA0 = uv_tri0[edges_info.index(best_edge)].item()
            uB0 = uv_tri0[(edges_info.index(best_edge)+1)%3].item()
            uC0 = uv_tri0[(edges_info.index(best_edge)+2)%3].item()
            uvA0 = self._uvs[uA0]; uvB0 = self._uvs[uB0]; uvC0 = self._uvs[uC0]
            uv_mid0 = (1 - t) * uvA0 + t * uvB0
            base_uv_start = self._uvs.shape[0] + len(new_uvs)
            u_mid0 = base_uv_start
            new_uvs.append(uv_mid0)
            new_uv_vmapping.append(idx_mid)

            tri1 = self._faces[adj_fidx]
            uv_tri1 = self._uv_indices[adj_fidx]
            def find_pos(face_tensor, vid):
                f = face_tensor.tolist()
                if f[0] == vid: return 0
                if f[1] == vid: return 1
                return 2
            posA = find_pos(tri1, vA)
            posB = find_pos(tri1, vB)
            posD = find_pos(tri1, vD)
            uA1 = int(uv_tri1[posA].item())
            uB1 = int(uv_tri1[posB].item())
            uD1 = int(uv_tri1[posD].item())
            uvA1 = self._uvs[uA1]; uvB1 = self._uvs[uB1]; uvD1 = self._uvs[uD1]
            uv_mid1 = (1 - t) * uvA1 + t * uvB1

            # Reuse uv_mid0 if close, otherwise add new uv_mid1
            if torch.allclose(uv_mid1, uv_mid0, atol=1e-4, rtol=1e-4):
                u_mid1 = u_mid0
            else:
                u_mid1 = self._uvs.shape[0] + len(new_uvs)
                new_uvs.append(uv_mid1)
                new_uv_vmapping.append(idx_mid)

            new_uv_faces.append([uA0, u_mid0, uC0])
            new_uv_faces.append([u_mid0, uB0, uC0])
            new_uv_faces.append([uA1, uD1, u_mid1])
            new_uv_faces.append([u_mid1, uD1, uB1])

        if len(new_vertices) == 0:
            # no face to split
            return (torch.empty(0,3,device=self._vertices.device),
                torch.empty(0,dtype=torch.long,device=self._vertices.device),
                torch.empty(0,3,dtype=torch.long,device=self._vertices.device),
                torch.empty(0,3,device=self._vertices.device),
                torch.empty(0,2,device=self._vertices.device),
                torch.empty(0,3,dtype=torch.long,device=self._vertices.device),
                torch.empty(0,dtype=torch.long,device=self._vertices.device))
            
        new_vertex_tensor = torch.stack(new_vertices)
        new_faces_tensor = torch.tensor(new_faces, dtype=torch.long, device=self._vertices.device)
        new_vertices_color_tensor = torch.stack(new_vertices_color)
        del_face_indices = torch.tensor(list(del_face_indices), dtype=torch.long, device=self._vertices.device)
        #  UV attributes for new vertices and faces
        new_uvs_tensor = torch.stack(new_uvs) if len(new_uvs) > 0 else torch.empty(0,2,device=self._vertices.device)
        new_uv_faces_tensor = torch.tensor(new_uv_faces, dtype=torch.long, device=self._vertices.device) if len(new_uv_faces) > 0 else torch.empty(0,3,dtype=torch.long,device=self._vertices.device)
        new_uv_vmapping_tensor = torch.tensor(new_uv_vmapping, dtype=torch.long, device=self._vertices.device) if len(new_uv_vmapping) > 0 else torch.empty(0,dtype=torch.long,device=self._vertices.device)
        return new_vertex_tensor, del_face_indices, new_faces_tensor, new_vertices_color_tensor, new_uvs_tensor, new_uv_faces_tensor, new_uv_vmapping_tensor
        
    def capture(self):
        return (
            self.active_sh_degree,
            self._features_dc,
            self._features_rest,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._features_dc, 
        self._features_rest,
        self.max_radii2D, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
    
    def prune_boundary_vertices(self, dead_mask, n_rounds=1):
        """
        Iteratively remove boundary vertices marked by dead_mask.
        In each round, boundary vertices are re-evaluated and removed if they meet the mask and degree criteria.
        Args:
            dead_mask (torch.BoolTensor): Mask indicating vertices to be removed.
            n_rounds (int): Number of removal rounds.
        Returns:
            int: Total number of faces deleted.
        """
        def remove_small_components(faces, uv_indices, min_faces=50):
            """
            Remove small connected face components from the mesh.
            Only keeps components with at least min_faces faces.
            Returns keep_mask (bool tensor) and removed_count (int).
            """
            faces_np = faces.detach().cpu().numpy()
            num_faces = faces_np.shape[0]
            edge2faces = defaultdict(list)
            for fi, face in enumerate(faces_np):
                for i in range(3):
                    e = tuple(sorted((face[i], face[(i + 1) % 3])))
                    edge2faces[e].append(fi)

            face_neighbors = [[] for _ in range(num_faces)]
            for _, adj_faces in edge2faces.items():
                if len(adj_faces) > 1:
                    for fi in adj_faces:
                        face_neighbors[fi].extend([f for f in adj_faces if f != fi])

            visited = np.zeros(num_faces, dtype=bool)
            components = []
            for i in range(num_faces):
                if not visited[i]:
                    queue = deque([i])
                    comp = []
                    visited[i] = True
                    while queue:
                        cur = queue.popleft()
                        comp.append(cur)
                        for nb in face_neighbors[cur]:
                            if not visited[nb]:
                                visited[nb] = True
                                queue.append(nb)
                    components.append(comp)

            keep_idx = []
            removed_count = 0
            for comp in components:
                if len(comp) >= min_faces:
                    keep_idx.extend(comp)
                else:
                    removed_count += len(comp)

            if removed_count > 0 and len(keep_idx) > 0:
                keep_mask = torch.zeros(faces.shape[0], dtype=torch.bool, device=faces.device)
                keep_mask[torch.tensor(keep_idx, device=keep_mask.device, dtype=torch.long)] = True
            else:
                keep_mask = torch.ones(faces.shape[0], dtype=torch.bool, device=faces.device)
            return keep_mask, removed_count
        
        device = self._faces.device
        total_del = 0
        face_index_map = torch.arange(self._faces.shape[0], device=device)
        for round_idx in range(n_rounds):
            faces = self._faces
            if faces.numel() == 0:
                break
            # Find boundary edges (used by only one face)
            edges_all = torch.cat([
                faces[:, [0, 1]],
                faces[:, [1, 2]],
                faces[:, [2, 0]],
            ], dim=0)
            edges_all = torch.sort(edges_all, dim=1).values
            uniq_edges, counts = torch.unique(edges_all, dim=0, return_counts=True)
            boundary_edges = uniq_edges[counts == 1]
            if boundary_edges.numel() == 0:
                break
            boundary_vertices = torch.unique(boundary_edges.flatten())
            V = self._vertices.shape[0]
            boundary_vertex_mask = torch.zeros(V, dtype=torch.bool, device=device)
            boundary_vertex_mask[boundary_vertices] = True
            candidate_vertices_mask = (dead_mask.bool() & boundary_vertex_mask)
            uniq_edges_all, _ = torch.unique(edges_all, dim=0, return_counts=True)
            degrees = torch.bincount(uniq_edges_all.flatten(), minlength=V)
            allow_vertex_mask = candidate_vertices_mask & (degrees <= 6)
            if not allow_vertex_mask.any():
                break
            # Remove faces containing these vertices
            face_delete_mask = allow_vertex_mask[faces].any(dim=1)
            num_del = int(face_delete_mask.sum().item())
            total_del += num_del
            keep_mask = ~face_delete_mask
            self._faces = self._faces[keep_mask]
            self._uv_indices = self._uv_indices[keep_mask]
            face_index_map = face_index_map[keep_mask]
            if num_del == 0:
                break
        # Remove small connected face components
        keep_mask2, removed_count = remove_small_components(self._faces, self._uv_indices, min_faces=50)
        if removed_count > 0 and keep_mask2.sum() > 0:
            self._faces = self._faces[keep_mask2]
            self._uv_indices = self._uv_indices[keep_mask2]
            face_index_map = face_index_map[keep_mask2]
        # Remap face attributes
        self.image_size = self.image_size[face_index_map]
        self.degeneracy_ratio = self.degeneracy_ratio[face_index_map]
        # Remap vertices and UVs
        used_v = torch.unique(self._faces.flatten())
        old2new = torch.full((self._vertices.shape[0],), -1, dtype=torch.long, device=device)
        old2new[used_v] = torch.arange(used_v.shape[0], device=device)
        self._vertices = nn.Parameter(self._vertices[used_v].detach().clone().requires_grad_(True))
        self._faces = old2new[self._faces]
        self.mask_grad_ema = self.mask_grad_ema[used_v]
        self._vertices_color = self._vertices_color[used_v]
        self._vmapping = old2new[self._vmapping]
        self._compact_uvs()
        # Refresh optimizer parameter groups
        self._number_of_faces = self._faces.shape[0]
        self._rebind_optimizer(vertex_mapping=used_v.detach().clone())
        return total_del

    def _compact_uvs(self):
        """
        Remove unused UV vertices and remap uv_indices.
        This function compacts the UV array by keeping only UVs referenced by faces and updates the mapping accordingly.
        """
        if self._uv_indices.numel() == 0 or self._uvs.numel() == 0:
            return
        device = self._uvs.device
        used_uv = torch.unique(self._uv_indices.reshape(-1))
        uv_old2new = torch.full((self._uvs.shape[0],), -1, dtype=torch.long, device=device)
        uv_old2new[used_uv] = torch.arange(used_uv.numel(), device=device)
        self._uvs = self._uvs[used_uv]
        self._vmapping = self._vmapping[used_uv]
        self._uv_indices = uv_old2new[self._uv_indices]

    def recreate_uvmap(self, bg_color=1.0):
        """
        Recompute UV mapping and bake vertex colors into the texture map.
        Updates UVs, UV indices, mapping, texture, and texture mask.
        Args:
            bg_color (float): Background color value for baking.
        """
        device = self._faces.device
        vertices_np = self._vertices.detach().cpu().numpy()
        faces_np = self._faces.detach().cpu().numpy()
        vertex_colors_np = self._vertices_color.detach().cpu().numpy()
        mesh = BasicMesh(
            vertices=vertices_np,
            faces=faces_np,
            vertex_colors=vertex_colors_np,
            vertex_normals=None,
            uvs=None,
            uv_indices=None,
            vmapping=None
        )
        mesh = extract_uv_map(mesh)
        H, W = self._texture.shape[1], self._texture.shape[2]
        texture_np, mask_np = vertex_color_to_uvmap(
            mesh.vertices, mesh.faces, mesh.uvs, mesh.vertex_colors,
            mesh.uv_indices, mesh.vmapping, (H, W), bg_color
        )
        self._uvs = torch.tensor(mesh.uvs, dtype=torch.float32, device=device)
        self._uv_indices = torch.tensor(mesh.uv_indices, dtype=torch.long, device=device)
        self._vmapping = torch.tensor(mesh.vmapping, dtype=torch.long, device=device)
        self._texture = nn.Parameter(torch.tensor(texture_np, dtype=torch.float32, device=device)).requires_grad_(True)
        self._texture_mask = torch.tensor(mask_np, dtype=torch.float32, device=device)

    def compute_boundary_dead_mask(self, grad_percent=0.15, area_percent=0.2, white_thresh=1.0, grad_thresh=1e-8, bg_color: int = 1):
        """
        Compute the dead mask for boundary vertices.
        Vertices are marked as dead if they have high mask_grad_ema or belong only to small area faces, and both the vertex and all its neighbors are close to the background color.
        Args:
            grad_percent (float): Top percentage of vertices by mask_grad_ema.
            area_percent (float): Bottom percentage of faces by area.
            white_thresh (float): Luma threshold for white background.
            grad_thresh (float): Nonzero threshold for mask_grad_ema.
            bg_color (int): 1 for white background, 0 for black.
        Returns:
            torch.BoolTensor: Dead mask for vertices (V,)
        """
        mask_grad_ema = self.mask_grad_ema  # (V,)
        V = mask_grad_ema.shape[0]
        k = int(V * grad_percent)
        topk_vals, topk_idx = torch.topk(mask_grad_ema, k, largest=True, sorted=True)
        nonzero_topk_mask = topk_vals > grad_thresh
        topk_mask = torch.zeros_like(mask_grad_ema, dtype=torch.bool)
        if nonzero_topk_mask.any():
            topk_idx = topk_idx[nonzero_topk_mask]
            topk_mask[topk_idx] = True

        areas = self.compute_face_areas()  # (M,)
        M = areas.shape[0]
        k_face = int(M * area_percent)
        _, area_idx = torch.topk(areas, k_face, largest=False, sorted=True)
        small_area_mask = torch.zeros(M, dtype=torch.bool, device=areas.device)
        small_area_mask[area_idx] = True

        # Count faces and small area faces for each vertex
        faces = self.get_faces  # (M,3)
        V = self._vertices.shape[0]
        M = faces.shape[0]
        device = faces.device
        vertex_face_count = torch.zeros(V, dtype=torch.long, device=device)
        vertex_small_area_count = torch.zeros(V, dtype=torch.long, device=device)

        for i in range(3):
            idx = faces[:, i]
            vertex_face_count.index_add_(0, idx, torch.ones(M, dtype=torch.long, device=device))
            vertex_small_area_count.index_add_(0, idx, small_area_mask.long())

        has_face = vertex_face_count > 0
        all_small = (vertex_face_count == vertex_small_area_count) & has_face
        only_small_area_vertex_mask = all_small

        dead_mask = topk_mask | only_small_area_vertex_mask  # (V,)

        # Restrict to background color vertices
        luma = 0.2126 * self._vertices_color[:, 0] + 0.7152 * self._vertices_color[:, 1] + 0.0722 * self._vertices_color[:, 2]
        if bg_color == 1:
            color_mask = luma >= white_thresh
        else:
            black_thresh = float(max(0.0, min(1.0, 1.0 - white_thresh)))
            color_mask = luma <= black_thresh

        # Build one-ring neighbor sets
        faces = self.get_faces  # (M,3)
        V = self._vertices.shape[0]
        device = self._vertices.device
        one_ring = [set() for _ in range(V)]
        faces_cpu = faces.detach().cpu().numpy()
        for f in faces_cpu:
            i, j, k = int(f[0]), int(f[1]), int(f[2])
            one_ring[i].update([j, k])
            one_ring[j].update([i, k])
            one_ring[k].update([i, j])

        # Check if all neighbors are background color
        color_mask_cpu = color_mask.detach().cpu().numpy()
        neighbor_all_bg = []
        for vid, neighbors in enumerate(one_ring):
            if len(neighbors) == 0:
                neighbor_all_bg.append(False)
            else:
                neighbor_all_bg.append(all(color_mask_cpu[nid] for nid in neighbors))
        neighbor_all_bg = torch.tensor(neighbor_all_bg, dtype=torch.bool, device=device)

        # Final mask: vertex and all neighbors are background color
        dead_mask = dead_mask & color_mask & neighbor_all_bg
        return dead_mask

    def compute_dead_mask(self, image_size_thresh=2, degeneracy_thresh=0.1, area_percent=0.5):
        """
        Compute the dead mask for faces based on image size, degeneracy, and area.
        Faces are marked as dead if they are small in image size or degeneracy, and also belong to the bottom area_percent by area.
        Args:
            image_size_thresh (float): Threshold for image size.
            degeneracy_thresh (float): Threshold for degeneracy ratio.
            area_percent (float): Bottom percentage of faces by area.
        Returns:
            torch.BoolTensor: Dead mask for faces (M,)
        """
        areas = self.compute_face_areas()  # (M,)
        imp_mask = self.image_size <= image_size_thresh  # (M,)
        degeneracy = self.compute_degeneracy_ratio()     # (M,)
        deg_mask = degeneracy < degeneracy_thresh        # (M,)
        impdeg_mask = imp_mask | deg_mask                # (M,)

        # Restrict to faces in the bottom area_percent by area
        _, sorted_idx = torch.sort(areas, descending=False)
        bottom_k = max(1, int(areas.shape[0] * area_percent))
        bottom_idx = sorted_idx[:bottom_k]
        area_bottom_mask = torch.zeros_like(impdeg_mask, dtype=torch.bool, device=impdeg_mask.device)
        area_bottom_mask[bottom_idx] = True

        dead_mask = impdeg_mask & area_bottom_mask       # (M,)
        return dead_mask
    