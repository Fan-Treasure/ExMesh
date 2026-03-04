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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, BasicMesh, extract_uv_map
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
import open3d as o3d

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    mesh: BasicMesh
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    """
    Compute NeRF++ normalization parameters (center and translation radius) for scene scale normalization.
    Args:
        cam_info: list of camera info
    Returns:
        dict with translate and radius
    """
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchMesh(path):
    """
    Read ply mesh file and return BasicMesh object.
    Used for NeRF synthetic datasets, keeps mesh structure for MeshModel initialization.
    Args:
        path: mesh ply file path
    Returns:
        BasicMesh object with vertices, faces, colors, normals
    """
    try:
        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.vertices) == 0:
            raise ValueError("No vertex data in PLY file")
        
        # Extract vertex coordinates
        vertices = np.asarray(mesh.vertices)
        # Extract triangle face indices
        faces = np.asarray(mesh.triangles)
        # Extract vertex colors, set to gray if not available
        if mesh.vertex_colors and len(mesh.vertex_colors) > 0:
            vertex_colors = np.asarray(mesh.vertex_colors)
        else:
            vertex_colors = np.ones_like(vertices) * 0.5  # 灰色
        
        # Compute vertex normals if not available
        if mesh.vertex_normals and len(mesh.vertex_normals) > 0:
            vertex_normals = np.asarray(mesh.vertex_normals)
        else:
            mesh.compute_vertex_normals()
            vertex_normals = np.asarray(mesh.vertex_normals)
        
        print(f"Mesh loaded: {len(vertices)} vertices, {len(faces)} faces")
        return BasicMesh(vertices=vertices, faces=faces, 
                         vertex_colors=vertex_colors, vertex_normals=vertex_normals, uvs=None, uv_indices=None, vmapping=None)

    except Exception as e:
        print(f"Failed to load mesh file: {e}")
        return None

def create_random_triangle_mesh(target_faces: int = 500) -> BasicMesh:
    """
    If mesh.ply does not exist, construct a nearly regular polyhedral triangle mesh: subdivide an icosahedron to get uniform triangles.
    Args:
        target_faces: desired number of triangle faces
    Returns:
        BasicMesh(vertices, faces, vertex_colors, vertex_normals)
    """
    # Start from icosahedron (20 faces), each midpoint subdivision multiplies faces by 4.
    # Choose subdivision times n so that 20 * 4^n is close to target_faces.
    if target_faces < 20:
        n = 0
    else:
        n = int(max(0, round(np.log(max(target_faces, 1) / 20.0) / np.log(4.0))))

    base = o3d.geometry.TriangleMesh.create_icosahedron(radius=1.0)
    # Perform n midpoint subdivisions
    if n > 0:
        mesh = base.subdivide_midpoint(n)
    else:
        mesh = base

    # Normalize vertices to unit sphere for uniform triangles
    V = np.asarray(mesh.vertices)
    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    V_unit = V / norms
    mesh.vertices = o3d.utility.Vector3dVector(V_unit)
    # Recompute normals
    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    vertex_normals = np.asarray(mesh.vertex_normals)
    # Randomly generate vertex colors in [0,1]
    vertex_colors = np.random.rand(vertices.shape[0], 3)

    print(f"Generated random subdivided icosahedron mesh: subdivisions={n}, vertices={len(vertices)}, faces={len(faces)}")
    return BasicMesh(vertices=vertices,
                     faces=faces,
                     vertex_colors=vertex_colors,
                     vertex_normals=vertex_normals,
                     uvs=None,
                     uv_indices=None,
                     vmapping=None,
                     )

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # Search for mesh.ply
    mesh_path = os.path.join(path, "mesh.ply")
    if os.path.exists(mesh_path):
        # Load existing mesh
        print(f"Loading mesh file: {mesh_path}")
        mesh = fetchMesh(mesh_path)
    else:
        # If mesh.ply does not exist, generate a random colored triangle sphere mesh
        print(f"Mesh file not found: {mesh_path}, generating random triangle sphere mesh for initialization")
        mesh = create_random_triangle_mesh(target_faces=2000)

    mesh = extract_uv_map(mesh)
    scene_info = SceneInfo(mesh=mesh,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=mesh_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    """
    Read Blender/NeRF-synthetic transforms_*.json and generate CameraInfo list.
    Args:
        path: dataset path
        transformsfile: json filename
        white_background: use white background
        extension: image extension
    Returns:
        list of CameraInfo
    """
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = frame["file_path"] + extension

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    """
    Read Blender/NeRF-synthetic scene info, including mesh, cameras, normalization, etc.
    Args:
        path: dataset path
        white_background: use white background
        eval: evaluation mode
        extension: image extension
    Returns:
        SceneInfo object
    """
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # Search for mesh.ply
    mesh_path = os.path.join(path, "mesh.ply")
    if os.path.exists(mesh_path):
        print(f"Loading mesh file: {mesh_path}")
        mesh = fetchMesh(mesh_path)
    else:
        print(f"Mesh file not found: {mesh_path}, generating random triangle sphere mesh for initialization")
        mesh = create_random_triangle_mesh(target_faces=2000)

    mesh = extract_uv_map(mesh)
    scene_info = SceneInfo(mesh=mesh,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=mesh_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}