import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Set GPU device index
import torch, numpy as np, glob
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.read_write_model import read_model
from PIL import Image

scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE").to(device)

for scene in scenes:
    input_dir = f"workdir/DTU/scan{scene}"
    sparse_dir = os.path.join(input_dir, "sparse", "0")
    out_dir = f"output_dtu_colmap/scan{scene}"
    os.makedirs(out_dir, exist_ok=True)

    # Load COLMAP data
    cameras, colmap_images, points3D = read_model(sparse_dir)
    images = []
    extrinsics = []
    intrinsics = []

    # Sort image names for consistent order
    for image_id, image_data in sorted(colmap_images.items(), key=lambda x: x[1].name):
        image_name = image_data.name
        image_path = os.path.join(input_dir, "images", image_name)
        if os.path.exists(image_path):
            images.append(image_path)
            
            # Extract extrinsic parameters
            R = image_data.qvec2rotmat()
            t = image_data.tvec
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = t
            extrinsics.append(extrinsic)
            
            # Extract intrinsic parameters
            camera = cameras[image_data.camera_id]
            if camera.model == "PINHOLE":
                fx, fy, cx, cy = camera.params
            elif camera.model == "SIMPLE_PINHOLE":
                f, cx, cy = camera.params
                fx = fy = f
            elif camera.model in ["SIMPLE_RADIAL", "RADIAL"]:
                f, cx, cy = camera.params[:3]
                fx = fy = f
            else:
                fx = fy = camera.params[0] if len(camera.params) > 0 else 1000
                cx = camera.width / 2
                cy = camera.height / 2
            
            intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            intrinsics.append(intrinsic)

    if not images:
        print(f"No images or poses found: {input_dir}")
        continue

    extrinsics_array = np.array(extrinsics)
    intrinsics_array = np.array(intrinsics)

    # Run depth inference
    prediction = model.inference(images, extrinsics=extrinsics_array, intrinsics=intrinsics_array)

    num = prediction.depth.shape[0]
    for i in range(num):
        stem = os.path.splitext(os.path.basename(images[i]))[0]
        save_path = os.path.join(out_dir, f"{stem}.npz")
        save_dict = {
            "depth": np.round(prediction.depth[i], 6),
        }
        if prediction.conf is not None:
            save_dict["conf"] = np.round(prediction.conf[i], 2)
        np.savez_compressed(save_path, **save_dict)

        # Save depth visualization
        depth = prediction.depth[i]
        depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_vis = (depth_vis * 255).astype(np.uint8)
        depth_img = Image.fromarray(depth_vis)
        depth_img.save(os.path.join(out_dir, f"{stem}.png"))

    print(f"scan{scene}: Saved {num} npz and depth visualizations to {out_dir}")

import shutil
for scene in scenes:
    src = f"output_dtu_colmap/scan{scene}/"
    target_dir = f"../ExMesh/workdir/DTU/scan{scene}/mono_priors/da3/"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)  # Remove if exists
    shutil.copytree(src, target_dir)  # Copy results to target
    print(f"Copied {src} -> {target_dir}")