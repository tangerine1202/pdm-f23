import os
import argparse
import multiprocessing
import glob
import copy
from collections import Counter
import numpy as np
import open3d as o3d

from tqdm.auto import tqdm
import time

from hw1 import (
    depth_image_to_point_cloud,
    rotation_matrix_to_quaternion,
    execute_global_registration,
    local_icp_algorithm
)

GT_T_SCALE = 0.1


def reconstruct(args):
    data_root = args.data_root
    voxel_size = args.voxel_size

    color_src = args.color_src
    seq_len = len(glob.glob(f'{data_root}{color_src}/*.png'))
    if seq_len == 0:
        raise ValueError(f'No images found in data_root: {data_root}')

    T_0j = np.eye(4)
    poses = []
    merged_pcd = o3d.geometry.PointCloud()
    prev_down, prev_fpfh = None, None
    # for i in range(1, seq_len + 1):
    for i in tqdm(range(1, seq_len + 1)):
        rgb_img = o3d.io.read_image(f'{data_root}{color_src}/{i}.png')
        dep_img = o3d.io.read_image(f'{data_root}depth/{i}.png')
        now_pcd = depth_image_to_point_cloud(rgb_img, dep_img, method=args.depth_unprojection)
        now_down, now_fpfh = preprocess_point_cloud(now_pcd, voxel_size, method=args.voxel_down)

        if prev_down is not None:
            # global registration
            trans_init = np.identity(4)
            reg_global = execute_global_registration(
                now_down, prev_down, now_fpfh, prev_fpfh, voxel_size)
            trans_init = reg_global.transformation
            # local registration
            reg_p2p = local_icp_algorithm(
                now_down, prev_down, voxel_size, trans_init, max_iter=50, method=args.icp)

            # transform from now to previous
            T_ij = reg_p2p.transformation
            # transform from now to 0
            T_0j = T_0j @ T_ij

        # estimate pose
        t = T_0j[:3, 3]
        R = T_0j[:3, :3]
        q = rotation_matrix_to_quaternion(R)
        pose = np.concatenate([t, q])
        poses.append(pose)

        # merge point cloud
        transformed_now_pcd = copy.deepcopy(now_pcd)
        transformed_now_pcd.transform(T_0j)
        merged_pcd += transformed_now_pcd

        prev_down, prev_fpfh = now_down, now_fpfh

    result_pcd = merged_pcd
    # pred_cam_poses = np.array(poses)

    return result_pcd


def process_chunk(points_chunk, colors_chunk, voxel_size, result_queue):
    dc = {}
    for i in range(len(points_chunk)):
        x_idx = int(points_chunk[i][0] // voxel_size)
        y_idx = int(points_chunk[i][1] // voxel_size)
        z_idx = int(points_chunk[i][2] // voxel_size)
        voxel_idx = (x_idx, y_idx, z_idx)
        color_key = tuple(colors_chunk[i])
        if voxel_idx not in dc:
            dc[voxel_idx] = Counter()
        dc[voxel_idx][color_key] += 1
    result_queue.put_nowait(dc)


def custom_voxel_down_multiproc(pcd, voxel_size):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    num_processes = multiprocessing.cpu_count()
    chunk_size = len(points) // num_processes

    result_queue = multiprocessing.Queue()
    processes = []
    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_processes - 1 else len(points)
        points_chunk = points[start_idx:end_idx]
        colors_chunk = colors[start_idx:end_idx]
        process = multiprocessing.Process(
            target=process_chunk,
            args=(points_chunk, colors_chunk, voxel_size, result_queue))
        processes.append(process)
        process.start()

    dc = {}
    cnt = 0
    while cnt < num_processes:
        if result_queue.empty():
            continue
        chunk_dc = result_queue.get()
        for voxel_idx, color_counter in chunk_dc.items():
            if voxel_idx not in dc:
                dc[voxel_idx] = Counter()
            dc[voxel_idx] += color_counter
        cnt += 1

    for process in processes:
        process.join()

    voxel_points = np.empty((len(dc), 3))
    voxel_colors = np.empty((len(dc), 3))
    for i, (voxel_idx, color_counter) in enumerate(dc.items()):
        voxel_colors[i] = color_counter.most_common(1)[0][0]
        voxel_points[i][0] = voxel_idx[0] * voxel_size + voxel_size / 2
        voxel_points[i][1] = voxel_idx[1] * voxel_size + voxel_size / 2
        voxel_points[i][2] = voxel_idx[2] * voxel_size + voxel_size / 2

    pcd_down = o3d.geometry.PointCloud()
    pcd_down.points = o3d.utility.Vector3dVector(voxel_points)
    pcd_down.colors = o3d.utility.Vector3dVector(voxel_colors)
    return pcd_down


def custom_voxel_down(pcd, voxel_size):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    voxel_indices = np.floor(points / voxel_size).astype(int)

    dc = {}
    for i in range(len(points)):
        voxel_idx = tuple(voxel_indices[i])
        color_key = tuple(colors[i])
        if voxel_idx not in dc:
            dc[voxel_idx] = Counter()
        dc[voxel_idx][color_key] += 1

    voxel_points = np.empty((len(dc), 3))
    voxel_colors = np.empty((len(dc), 3))
    for i, (voxel_idx, color_counter) in enumerate(dc.items()):
        voxel_colors[i] = color_counter.most_common(1)[0][0]
        voxel_points[i][0] = voxel_idx[0] * voxel_size + voxel_size / 2
        voxel_points[i][1] = voxel_idx[1] * voxel_size + voxel_size / 2
        voxel_points[i][2] = voxel_idx[2] * voxel_size + voxel_size / 2

    pcd_down = o3d.geometry.PointCloud()
    pcd_down.points = o3d.utility.Vector3dVector(voxel_points)
    pcd_down.colors = o3d.utility.Vector3dVector(voxel_colors)
    return pcd_down


def preprocess_point_cloud(pcd, voxel_size, method='my'):
    # Do voxelization to reduce the number of points for less memory usage and speedup
    if method == 'my':
        # multi process
        pcd_down = custom_voxel_down_multiproc(pcd, voxel_size)
        # single process
        # pcd_down = custom_voxel_down(pcd, voxel_size)
    elif method == 'open3d':
        # ref: http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Voxel-downsampling
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    else:
        raise NotImplementedError

    # ref: http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
    # Compute normals and fpfh features for the global registration
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--voxel_size', type=float, default=0.01)
    parser.add_argument('--depth_unprojection', type=str, default='my', help='open3d or my')
    parser.add_argument('--voxel_down', type=str, default='my', help='open3d or my')
    parser.add_argument('--icp', type=str, default='my', help='open3d or my')
    parser.add_argument('--color_src', type=str, default='seg', help='rgb or seg')
    parser.add_argument('--save', action='store_true', default=False, help='save pcd')
    args = parser.parse_args()

    if args.color_src == 'seg' and args.voxel_down == 'open3d':
        raise ValueError('seg color_src only supports "my" voxel_down implementation')

    if args.data_root == None:
        if args.floor == 1:
            args.data_root = 'data_collection/first_floor/'
        elif args.floor == 2:
            args.data_root = 'data_collection/second_floor/'

    result_pcd = reconstruct(args)

    # Save
    if args.save:
        fname = f'f{args.floor}_vs-{args.voxel_size}_vd-{args.voxel_down}_icp-{args.icp}_clr-{args.color_src}.pcd'
        fpath = os.path.join('pcd', fname)
        o3d.io.write_point_cloud(fname, result_pcd)

    ceiling_y_threshold = 0.0 * GT_T_SCALE
    ceiling_mask = np.array(result_pcd.points)[:, 1] < ceiling_y_threshold
    cropped_pcd = result_pcd.select_by_index(np.where(ceiling_mask)[0])

    # # Visualize
    o3d.visualization.draw_geometries(
        [cropped_pcd],
    )
