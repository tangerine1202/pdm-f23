import argparse
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
                now_down, prev_down, trans_init, threshold=voxel_size, max_iter=50, method=args.icp)

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


def custom_voxel_down(pcd, voxel_size):
    # TODO: implement your own voxel down
    """
    Voxel downsampling uses a regular voxel grid to create a uniformly downsampled point cloud from an input point cloud.

    The algorithm operates in two steps:
    1. Points are bucketed into voxels.
    2. Each occupied voxel generates exactly one point by averaging all points inside.
    ref: http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Voxel-downsampling
    """
    # t0 = time.time()
    points = np.array(pcd.points)
    colors = np.array(pcd.colors)

    # FIXME: 0.03 - 40-60% / 0.01 - 5%
    # unique_t0 = time.time()
    unique_colors, color_ids = np.unique(colors, axis=0, return_inverse=True)
    # unique_t = time.time() - unique_t0

    # ignorble runtime
    xbins = np.arange(np.min(points[:, 0]), np.max(points[:, 0]), voxel_size)
    ybins = np.arange(np.min(points[:, 1]), np.max(points[:, 1]), voxel_size)
    zbins = np.arange(np.min(points[:, 2]), np.max(points[:, 2]), voxel_size)
    pts_xbin_idx = np.digitize(points[:, 0], xbins)
    pts_ybin_idx = np.digitize(points[:, 1], ybins)
    pts_zbin_idx = np.digitize(points[:, 2], zbins)

    # FIXME: 0.03 - 30-50% / 0.01 - 90%
    # loop_t0 = time.time()
    # bool_op_t = 0
    # where_t = 0
    # color_t = 0
    voxel_points = []
    voxel_colors = []
    for i in range(len(xbins)):
        xbin_mask = (pts_xbin_idx == i)
        if not np.any(xbin_mask):
            continue
        for j in range(len(ybins)):
            ybin_mask = (pts_ybin_idx == j)
            # bool_op_t0 = time.time()
            xybin_mask = (xbin_mask & ybin_mask)
            # bool_op_t += time.time() - bool_op_t0
            if not np.any(xybin_mask):
                continue
            for k in range(len(zbins)):
                zbin_mask = (pts_zbin_idx == k)

                # FIXME: 0.03 - 10% / 0.01 - 15%
                # bool_op_t0 = time.time()
                bin_mask = xybin_mask & zbin_mask
                # bool_op_t += time.time() - bool_op_t0
                if not np.any(bin_mask):
                    continue

                # FIXME: 0.03 - 4~% / 0.01 - 2-3%
                # where_t0 = time.time()
                voxel_idx = np.nonzero(bin_mask)[0]
                # where_t += time.time() - where_t0

                # FIXME: 0.03 - 5~% / 0.01 - 1-2%
                # take the most common color in the voxel
                # color_t0 = time.time()
                most_common_color_id = Counter(color_ids[voxel_idx]).most_common(1)[0][0]
                voxel_color = unique_colors[most_common_color_id]
                # color_t += time.time() - color_t0

                # put the voxel point at the center of the voxel
                voxel_point = np.array([xbins[i] + voxel_size / 2,
                                        ybins[j] + voxel_size / 2,
                                        zbins[k] + voxel_size / 2])

                voxel_points.append(voxel_point)
                voxel_colors.append(voxel_color)
    # loop_t = time.time() - loop_t0
    # total_t = time.time() - t0
    # print(f'unique_t:   {unique_t:.4f} {unique_t / total_t:.4f}%')
    # print(f'bool_op_t:  {bool_op_t:.4f} {bool_op_t / total_t:.4f}%')
    # print(f'where_t:    {where_t:.4f} {where_t / total_t:.4f}%')
    # print(f'color_t:    {color_t:.4f} {color_t / total_t:.4f}%')
    # print(f'loop_t:     {loop_t:.4f} {loop_t / total_t:.4f}%')
    # print(f'total_t:    {total_t:.4f} {total_t / total_t:.4f}%')

    pcd_down = o3d.geometry.PointCloud()
    pcd_down.points = o3d.utility.Vector3dVector(np.array(voxel_points))
    pcd_down.colors = o3d.utility.Vector3dVector(np.array(voxel_colors))
    return pcd_down


def preprocess_point_cloud(pcd, voxel_size, method='my'):
    # Do voxelization to reduce the number of points for less memory usage and speedup
    if method == 'my':
        pcd_down = custom_voxel_down(pcd, voxel_size)
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
    args = parser.parse_args()

    if args.data_root == None:
        if args.floor == 1:
            args.data_root = 'data_collection/first_floor/'
        elif args.floor == 2:
            args.data_root = 'data_collection/second_floor/'

    result_pcd = reconstruct(args)

    ceiling_y_threshold = 0.0 * GT_T_SCALE
    ceiling_mask = np.array(result_pcd.points)[:, 1] < ceiling_y_threshold
    cropped_pcd = result_pcd.select_by_index(np.where(ceiling_mask)[0])

    # Visualize
    o3d.visualization.draw_geometries(
        [cropped_pcd],
    )
