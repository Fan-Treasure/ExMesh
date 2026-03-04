import os
import imageio
import numpy as np
import torch
import torch.nn.functional as F

def get_pred_depth(dataset, viewpoints, res, depth_type="da3"):
    """
    Read depth for all viewpoints at once, store as tensor, return dict[image_name] = (H,W) CUDA tensor
    """
    H, W = res
    gt_depths = {}

    for vp in viewpoints:
        img_name = getattr(vp, "image_name", str(getattr(vp, "colmap_id", "0")))
        colmap_id = getattr(vp, "colmap_id", None)

        if depth_type == "da2":
            depth_idx_path = os.path.join(dataset.source_path, "mono_priors", "depthanythingv2", f"{img_name}.png")
            codebook_path = os.path.join(dataset.source_path, "mono_priors", "depthanythingv2", f"{img_name}.npy")
            depth_idx = imageio.imread(depth_idx_path).astype(np.int64)  # (h, w)
            codebook = np.load(codebook_path)  # (65536,)
            arr = codebook[depth_idx]  # (h, w)
        elif depth_type == "da3":
            npz_path = os.path.join(dataset.source_path, "mono_priors", "da3", f"{img_name}.npz")
            data = np.load(npz_path)
            arr = data["depth"]  # (h, w)

        gt_depth = torch.from_numpy(arr).float()  # (h, w)
        # Resize to target resolution
        if gt_depth.shape != (H, W):
            gt_depth = gt_depth.unsqueeze(0).unsqueeze(0)
            gt_depth = F.interpolate(gt_depth, size=(H, W), mode='bilinear', align_corners=False)
            gt_depth = gt_depth.squeeze(0).squeeze(0)
        gt_depth = gt_depth.to("cuda")
        gt_depths[img_name] = gt_depth

    return gt_depths