import os

scene_ids = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
scene_ids = [105]
gpu_id = 2  # Set this to the GPU you want to use
data_base_path='workdir/DTU'

scene_params = {  # These parameters are provided for reproducibility, but they can be further optimized for better results.
    24:  {"lambda_smooth": 3000, "lambda_depth": 0.1},
    37:  {"lambda_smooth": 3000, "lambda_depth": 0.1},
    40:  {"lambda_smooth": 3000, "lambda_depth": 0.03},
    55:  {"lambda_smooth": 3000, "lambda_depth": 0.01},
    63:  {"lambda_smooth": 3000, "lambda_depth": 0.03},
    65:  {"lambda_smooth": 3000, "lambda_depth": 0.01},
    69:  {"lambda_smooth": 1000, "lambda_depth": 0.1},
    83:  {"lambda_smooth": 3000, "lambda_depth": 0.01},
    97:  {"lambda_smooth": 3000, "lambda_depth": 0.01},
    105: {"lambda_smooth": 3000, "lambda_depth": 0.03},
    106: {"lambda_smooth": 3000, "lambda_depth": 0.1},
    110: {"lambda_smooth": 1000, "lambda_depth": 0.01},
    114: {"lambda_smooth": 3000, "lambda_depth": 0.03},
    118: {"lambda_smooth": 1000, "lambda_depth": 0.01},
    122: {"lambda_smooth": 1000, "lambda_depth": 0.03},
}

for sid in scene_ids:
    params = scene_params[sid] 
    scan_dir = f"workdir/DTU/scan{sid}"
    model_dir = f"outputs/optimized_meshes_10000/dtu_scan{sid}"
    
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {scan_dir} -m {model_dir} " + \
          f"--lambda_smooth {params['lambda_smooth']} --lambda_depth {params['lambda_depth']}"
    print(f"Running: {cmd}")
    os.system(cmd)
    
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_mesh {model_dir}/mesh/auto_mesh_iter_10000.ply " + \
          f"--scan_id {sid} --output_dir {model_dir}/eval " + \
          f"--mask_dir {data_base_path} " + \
          f"--DTU {data_base_path}"
    print(cmd)
    os.system(cmd)
