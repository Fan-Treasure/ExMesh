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
import torch
from random import randint
from utils.loss_utils import *
from utils.mono_utils import get_pred_depth
import sys
from scene import Scene, MeshModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.graphics_utils import uvmap_to_vertex_color
from renderer import render
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import lpips
import torchvision
from utils.log import Tee
import datetime
import numpy as np
from PIL import Image
import numpy as np
import cv2


def training(
    dataset,   
    opt, 
    pipe,
    testing_iterations,
    save_iterations,
    checkpoint, 
    debug_from,
    ):
    first_iter = 0
    # Prepare output directories and TensorBoard logger
    tb_writer = prepare_output_and_logger(dataset)
    # Set background color
    bg_color = 1 if dataset.white_background else 0
    background = torch.tensor([bg_color, bg_color, bg_color], dtype=torch.float32, device="cuda")

    # Load mesh model and scene
    mesh = MeshModel(dataset.sh_degree)
    scene = Scene(dataset, mesh, opt.texture_resolution, bg_color)
    # Save initial UV map
    save_uvmap_dir = os.path.join(dataset.model_path, "uvmap")
    os.makedirs(save_uvmap_dir, exist_ok=True)
    uvmap = mesh.get_texture.detach().cpu().clamp(0, 1)  # (3, H, W)
    save_path = os.path.join(save_uvmap_dir, "uvmap_init.png")
    torchvision.utils.save_image(uvmap, save_path)
    print(f"Initial uv map has been saved to: {save_path}")
    
    # Setup optimizer and learning rate
    mesh.training_setup(opt, opt.feature_lr, opt.lr_vertices_init)

    # Restore from checkpoint if available
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        mesh.restore(model_params, opt)

    # CUDA timers for iteration timing
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # Get all training camera views
    viewpoint_stack = scene.getTrainCameras().copy()
    H, W = viewpoint_stack[0].image_height, viewpoint_stack[0].image_width
    gt_depths = get_pred_depth(dataset, viewpoint_stack, (H,W), depth_type='da3')

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training")
    first_iter += 1

    new_round = True
    removed_them = False

    loss_fn = l1_loss

    # Save initial mesh
    auto_ply_path = os.path.join(dataset.model_path, "mesh", f"auto_mesh_iter_0.ply")
    save_mesh_dir = os.path.join(dataset.model_path, "mesh")
    os.makedirs(save_mesh_dir, exist_ok=True)
    mesh.save_as_ply(auto_ply_path)
    
    # Initialize EMA vertex reference
    mesh.vertices_ref = mesh.get_vertices.detach().clone()

    # Training loop
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # Update learning rate
        mesh.update_learning_rate(iteration)

        # Refill camera pool if empty
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            if not new_round and removed_them:
                new_round = True
                removed_them = False
            else:
                new_round = False

        # Randomly select a training camera view
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Enable debug mode if needed
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Use random or fixed background
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, mesh, pipe, bg)
        image = render_pkg["render"]
        
        # Triangle projection stats
        image_size = render_pkg["scaling"].detach()

        # Update triangle stats
        mask = image_size > mesh.image_size
        mesh.image_size[mask] = image_size[mask]

        # Compute losses
        gt_image = viewpoint_cam.original_image.cuda()
        train_bg = torch.tensor([bg_color, bg_color, bg_color], dtype=torch.float32, device="cuda").view(3, 1, 1).expand_as(gt_image)
        if viewpoint_cam.gt_alpha_mask is not None:
            gt_mask = viewpoint_cam.gt_alpha_mask.cuda()
            gt_image = gt_image * gt_mask + train_bg * (1 - gt_mask)
        else:
            img_name = viewpoint_cam.image_name
            mask_path = os.path.join(dataset.source_path, "train_mask", f"{img_name}_gtmask.png")
            gt_mask_pil = Image.open(mask_path).convert("L")
            gt_mask = torchvision.transforms.functional.to_tensor(gt_mask_pil).squeeze(0).to(image.device)
        
        rend_depth = render_pkg["rend_depth"].squeeze(0)  # (H, W)
        depth_mask = (gt_mask > 0) & (rend_depth > 0)
        if depth_mask.dim() == 3 and depth_mask.shape[0] == 1:
            depth_mask = depth_mask.squeeze(0)
        gt_depth = gt_depths[viewpoint_cam.image_name]
        
        # Pearson correlation depth loss
        depth_loss, depth_contrib = pearson_correlation_loss(rend_depth, gt_depth, depth_mask)
        depth_loss = depth_loss * opt.lambda_depth
        
        pixel_loss = loss_fn(image, gt_image)
        # Composite pixel loss (L1/L2 + SSIM)
        loss_image = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # Silhouette loss
        pred_mask = render_pkg["rend_alpha"].squeeze(0)  # (H,W)
        silhouette_loss = silhouette_bce_loss(pred_mask, gt_mask) * opt.lambda_silhouette
        rend_normal  = render_pkg['rend_normal']

        if iteration < opt.split_from_iter:
            silhouette_loss.backward(retain_graph=True)
            with torch.no_grad():
                mask_grad_intensity = torch.norm(mesh._vertices.grad, dim=1)
                mesh.mask_grad_ema += mask_grad_intensity
            mesh._vertices.grad = None
            mesh._texture.grad = None
            
        # Laplacian smooth loss
        smooth_loss1 = laplacian_smooth_loss(mesh.get_vertices, mesh.get_faces) * opt.lambda_smooth
        smooth_loss = smooth_loss1
        deviation_loss = double_vertex_deviation_loss(mesh.get_vertices, mesh.vertices_ref, mesh.get_faces) * opt.lambda_deviation

        # Total loss
        loss = loss_image + smooth_loss + deviation_loss + silhouette_loss + depth_loss

        loss.backward()
        iter_end.record()
        
        with torch.no_grad():
            # Update progress bar and log loss
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "depth": f"{depth_loss.item():.5f}",
                    "sil": f"{silhouette_loss.item():.5f}",
                    "img": f"{loss_image.item():.5f}",
                    "smo": f"{smooth_loss.item():.5f}",
                    "dev": f"{deviation_loss.item():.5f}",
                    "#faces": mesh.get_faces.shape[0],
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Logging and model saving
            training_report(tb_writer, iteration, pixel_loss, loss, loss_fn, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if iteration in save_iterations:
                print("\n[ITER {}] Saving Triangles".format(iteration))
                scene.save(iteration)

            if iteration == 1:
                # Initial vertex merge and UV map recreation
                vertex_colors = uvmap_to_vertex_color(mesh._vertices, mesh._uvs, mesh._texture, mesh._vmapping, mesh._texture_mask)
                mesh._vertices_color = torch.tensor(vertex_colors, dtype=torch.float32, device="cuda")
                areas = mesh.compute_face_areas()
                area_thr = torch.quantile(areas, 0.1)
                for i in range(10):
                    degeneracy = mesh.compute_degeneracy_ratio()
                    deg_mask = degeneracy < 0.1
                    areas = mesh.compute_face_areas()
                    area_mask = areas < area_thr
                    dead_mask = deg_mask | area_mask
                    mesh.merge_close_vertex(dead_mask, length_ratio_thresh=0.2)
                mesh.recreate_uvmap(bg_color=bg_color)
                mesh._reset_optimizer()
                mesh.vertices_ref = mesh.get_vertices.detach().clone()
                mesh.mask_grad_ema = torch.zeros(mesh._vertices.shape[0], device="cuda")
                removed_them = True
                new_round = False

            # Prune boundary vertices by area and mask_grad_ema
            if iteration % opt.densification_interval == 0 and iteration < opt.split_from_iter:
                vertex_colors = uvmap_to_vertex_color(mesh._vertices, mesh._uvs, mesh._texture, mesh._vmapping, mesh._texture_mask)
                mesh._vertices_color = torch.tensor(vertex_colors, dtype=torch.float32, device="cuda")
                dead_mask = mesh.compute_boundary_dead_mask(grad_percent=0.2, area_percent=0.2, white_thresh=0.995, grad_thresh=1e-8, bg_color=bg_color)  # (V,)
                mesh.prune_boundary_vertices(dead_mask, n_rounds=5)
                mesh.vertices_ref = mesh.get_vertices.detach().clone()
                # reset mask_grad_ema
                mesh.mask_grad_ema = torch.zeros(mesh._vertices.shape[0], device="cuda")
                removed_them = True
                new_round = False

            # Vertex split and merge
            elif iteration >= opt.split_from_iter and iteration % opt.densification_interval == 0 and iteration <= opt.split_until_iter:
                vertex_colors = uvmap_to_vertex_color(mesh._vertices, mesh._uvs, mesh._texture, mesh._vmapping, mesh._texture_mask)
                mesh._vertices_color = torch.tensor(vertex_colors, dtype=torch.float32, device="cuda")
                if iteration == 1000:
                    # Extra boundary pruning at iter 1000
                    dead_mask = mesh.compute_boundary_dead_mask(grad_percent=0.2, area_percent=0.2, white_thresh=0.995, grad_thresh=1e-8, bg_color=bg_color)
                    mesh.prune_boundary_vertices(dead_mask, n_rounds=5)
                dead_mask = mesh.compute_dead_mask(image_size_thresh=10, degeneracy_thresh=opt.degeneracy_threshold*2, area_percent=0.5)
                mesh.add_new_face(cap_max=opt.max_shapes, dead_mask=dead_mask)
                if iteration % (opt.densification_interval * 4) == 0:
                    # Vertex merge every 2000 iterations
                    dead_mask = mesh.compute_boundary_dead_mask(grad_percent=0.2, area_percent=0.2, white_thresh=0.995, grad_thresh=1e-8, bg_color=bg_color)
                    mesh.prune_boundary_vertices(dead_mask, n_rounds=2)
                    for i in range(5):
                        dead_mask = mesh.compute_dead_mask(image_size_thresh=0, degeneracy_thresh=opt.degeneracy_threshold, area_percent=0.3)
                        mesh.merge_close_vertex(dead_mask, length_ratio_thresh=0.2)
                removed_them = True
                new_round = False
                mesh.image_size = torch.zeros(mesh.get_faces.shape[0], device="cuda")
                # Reset EMA reference after topology change
                mesh.vertices_ref = mesh.get_vertices.detach().clone()

            if iteration in (500, 2000, 4000, 6000):
                # Recreate UV map at key iterations
                mesh.recreate_uvmap(bg_color=bg_color)
                mesh._reset_optimizer()
                print(f"[ITER {iteration}] UV map recreated")

            # Optimizer step and zero gradients
            if iteration < opt.iterations:
                mesh.optimizer.step()
                mesh.optimizer.zero_grad(set_to_none = True)
                # EMA update for reference vertices
                beta = getattr(opt, "smooth_ref_ema", 0.99)
                mesh.vertices_ref.mul_(beta).add_((1.0 - beta) * mesh.get_vertices.detach())
        # Save mesh and images at intervals
        if (iteration == 1) or (iteration <= 4000 and iteration % 500 == 499) or (iteration <= 4000 and iteration % 500 == 0) or (iteration > 4000 and iteration % 1000 == 999) or (iteration > 4000 and iteration % 1000 == 0):
            auto_ply_path = os.path.join(dataset.model_path, "mesh", f"auto_mesh_iter_{iteration}.ply")
            mesh.save_as_ply(auto_ply_path)
            save_img_dir = os.path.join(dataset.model_path, "render_images")
            os.makedirs(save_img_dir, exist_ok=True)
            # Save UV map
            save_uvmap_dir = os.path.join(dataset.model_path, "uvmap")
            uvmap = mesh.get_texture.detach().cpu().clamp(0, 1)  # (3, H, W)
            save_path = os.path.join(save_uvmap_dir, f"iter_{iteration}_uvmap.png")
            torchvision.utils.save_image(uvmap, save_path)
            
            # Save GT and rendered RGB images
            gt_img_np = (gt_image.permute(1,2,0).clamp(0,1).detach().cpu().numpy() * 255).astype(np.uint8)
            render_img_np = (image.permute(1,2,0).clamp(0,1).detach().cpu().numpy() * 255).astype(np.uint8)
            gt_img_show = gt_img_np[:, :, [2,1,0]]
            render_img_show = render_img_np[:, :, [2,1,0]]

            # Save normal maps
            rend_normal = render_pkg['rend_normal']  # (3,H,W)
            rend_normal_np = ((rend_normal.permute(1,2,0) * 0.5 + 0.5).clamp(0,1).detach().cpu().numpy() * 255).astype(np.uint8)
            rend_normal_show = rend_normal_np[:, :, [2,1,0]]

            # Save depth maps (grayscale)
            rend_depth_full = render_pkg["rend_depth"].squeeze(0).detach().cpu().numpy()  # (H,W)
            gt_depth_np = gt_depth.detach().cpu().numpy()
            depth_mask_np = (depth_mask.detach().cpu().numpy())
            depth_vis = np.zeros_like(rend_depth_full, dtype=np.float32)
            rd = rend_depth_full[depth_mask_np]
            depth_vis[depth_mask_np] = (rd - rd.min()) / (rd.max() - rd.min() + 1e-8)
            rend_depth_gray_u8 = (depth_vis * 255).astype(np.uint8)
            rend_depth_gray_color = cv2.cvtColor(rend_depth_gray_u8, cv2.COLOR_GRAY2BGR)
            gt_depth_vis = np.zeros_like(gt_depth_np, dtype=np.float32)
            gd = gt_depth_np[depth_mask_np]
            gt_depth_vis[depth_mask_np] = (gd - gd.min()) / (gd.max() - gd.min() + 1e-8)
            gt_depth_gray_u8 = (gt_depth_vis * 255).astype(np.uint8)
            gt_depth_gray_color = cv2.cvtColor(gt_depth_gray_u8, cv2.COLOR_GRAY2BGR)

            # Save depth contribution (colormap)
            contrib_np = depth_contrib.detach().cpu().numpy()  # (H,W)
            contrib_vis = np.zeros_like(contrib_np, dtype=np.float32)
            cv = contrib_np[depth_mask_np]
            contrib_vis[depth_mask_np] = (cv - cv.min()) / (cv.max() - cv.min() + 1e-8)
            contrib_u8 = (contrib_vis * 255).astype(np.uint8)
            contrib_color = cv2.applyColorMap(contrib_u8, cv2.COLORMAP_JET)

            # Save mask difference (colormap)
            mask_diff = torch.abs(pred_mask.detach().cpu() - gt_mask.detach().cpu())
            mask_diff_img = (mask_diff.numpy() * 255).astype(np.uint8)
            if mask_diff_img.ndim == 3:
                mask_diff_img = mask_diff_img.squeeze()
            mask_diff_color = cv2.applyColorMap(mask_diff_img, cv2.COLORMAP_JET)
            
            row0 = np.concatenate([gt_img_show, render_img_show, rend_normal_show, rend_normal_show], axis=1)
            row1 = np.concatenate([rend_depth_gray_color, gt_depth_gray_color, contrib_color, mask_diff_color], axis=1)
            image_to_show = np.concatenate([row0, row1], axis=0)
            debug_path = os.path.join(save_img_dir, f"iter_{iteration}_panel.jpg")
            cv2.imwrite(debug_path, image_to_show)
        
    print("Training is done")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, pixel_loss, loss, loss_fn, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/pixel_loss', pixel_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                pixel_loss_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                total_time = 0.0
                
                save_dir = os.path.join(scene.model_path, f"test_render_iter_{iteration}")
                os.makedirs(save_dir, exist_ok=True)

                for idx, viewpoint in enumerate(config['cameras']):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    image = torch.clamp(renderFunc(viewpoint, scene.mesh, *renderArgs)["render"], 0.0, 1.0)
                    end_event.record()
                    torch.cuda.synchronize()
                    runtime = start_event.elapsed_time(end_event)
                    total_time += runtime

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if viewpoint.gt_alpha_mask is not None:
                        gt_mask = viewpoint.gt_alpha_mask.cuda()
                        train_bg = torch.tensor([0,0,0], dtype=torch.float32, device="cuda").view(3, 1, 1).expand_as(gt_image)
                        gt_image = gt_image * gt_mask + train_bg * (1 - gt_mask)
                    
                    img_name = getattr(viewpoint, "image_name", str(idx))
                    save_render_path = os.path.join(save_dir, f"{config['name']}_{img_name}_render.png")
                    save_gt_path = os.path.join(save_dir, f"{config['name']}_{img_name}_gt.png")
                    torchvision.utils.save_image(image.detach().cpu(), save_render_path)
                    torchvision.utils.save_image(gt_image.detach().cpu(), save_gt_path)
                    
                    rend_normal = renderFunc(viewpoint, scene.mesh, *renderArgs)["rend_normal"]
                    rend_normal_write = rend_normal.permute(1,2,0) * 0.5 + 0.5 
                    normal_path = os.path.join(save_dir, f"{config['name']}_{img_name}_rend_normal.png")
                    torchvision.utils.save_image(rend_normal_write.permute(2,0,1).detach().cpu(), normal_path)
                    
                    rend_depth = renderFunc(viewpoint, scene.mesh, *renderArgs)["rend_depth"]
                    rend_depth_img = rend_depth.squeeze(0).detach().cpu()
                    fg_mask = rend_depth_img > 0
                    rend_depth_vis = torch.zeros_like(rend_depth_img)
                    if fg_mask.any():
                        min_val = rend_depth_img[fg_mask].min()
                        max_val = rend_depth_img[fg_mask].max()
                        rend_depth_vis[fg_mask] = (rend_depth_img[fg_mask] - min_val) / (max_val - min_val + 1e-8)
                    rend_depth_vis = rend_depth_vis.clamp(0, 1)
                    rend_depth_path = os.path.join(save_dir, f"{config['name']}_{img_name}_rend_depth.png")
                    torchvision.utils.save_image(rend_depth_vis.unsqueeze(0), rend_depth_path)

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    pixel_loss_test += loss_fn(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips_fn(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                pixel_loss_test /= len(config['cameras'])       
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])  
                total_time /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], pixel_loss_test, psnr_test, ssim_test, lpips_test))

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', pixel_loss_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            #tb_writer.add_histogram("scene/opacity_histogram", scene.triangles.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.mesh.get_vertices.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument("--no_dome", action="store_true", default=False)
    parser.add_argument("--outdoor", action="store_true", default=False)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    log_dir = os.path.join(args.model_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"train_log_{now_str}.txt")
    sys.stdout = Tee(log_path, "a")

    print("Optimizing " + args.model_path)

    lpips_fn = lpips.LPIPS(net='vgg').to(device="cuda")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args),
             op.extract(args),
             pp.extract(args),
             args.test_iterations,
             args.save_iterations,
             args.start_checkpoint,
             args.debug_from,
             )
    
    # All done
    print("\nTraining complete.")