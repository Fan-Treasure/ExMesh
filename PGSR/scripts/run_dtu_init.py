import os

scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
scenes = [65, 83, 110]
data_base_path='../workdir/DTU'
out_base_path='../outputs/coarse_meshes'
eval_path='../workdir/DTU'
out_name='test'
gpu_id=2

for scene in scenes:
    cmd = f'rm -rf {out_base_path}/dtu_scan{scene}/{out_name}/*'
    print(cmd)
    os.system(cmd)

    cmd = f'cp -rf {data_base_path}/scan{scene}/sparse/0/* {data_base_path}/scan{scene}/sparse/'
    print(cmd)
    os.system(cmd)

    common_args = "--quiet -r2 --ncc_scale 0.5"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    common_args = "--quiet --num_cluster 1 --voxel_size 0.01 --max_depth 5.0 --use_depth_filter"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)
    
    cmd = f"cp ../outputs/coarse_meshes/dtu_scan{scene}/test/mesh/tsdf_fusion_post.ply ../workdir/DTU/scan{scene}/mesh.ply"
    print(cmd)
    os.system(cmd)
