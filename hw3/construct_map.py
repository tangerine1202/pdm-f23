import json
import os
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

GT_T_SCALE = 0.1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcd_path', type=str, help='path of pcd file')
    parser.add_argument('--npy_path', type=str, help='path of npy file', default='semantic_3d_pointcloud/point.npy')
    parser.add_argument('--clr_path', type=str, help='path of color file', default='semantic_3d_pointcloud/color01.npy')
    parser.add_argument('--map_name', type=str, help='name of map image', default='map')
    parser.add_argument('--map_w_size', type=float, default=10.0, help='width size of map image')
    parser.add_argument('--save_dir', type=str, default='map', help='path to store cropped pcd')
    parser.add_argument('--ceiling_y', type=float, default=0.0, help='y threshold of ceiling')
    parser.add_argument('--floor_y', type=float, default=-1.2, help='y threshold of floor')
    parser.add_argument('--down_size', type=float, default=0.0, help='down sampling voxel size of pcd')
    parser.add_argument('--store_pcd', action='store_true', help='store cropped pcd')
    parser.add_argument('--viz', action='store_true', help='visualize cropped pcd')
    args = parser.parse_args()

    # read full pcd
    if args.pcd_path:
        pcd = o3d.io.read_point_cloud(args.pcd_path)
        ext = os.path.splitext(args.pcd_path)[-1]
        filename = os.path.basename(args.pcd_path)
    elif args.npy_path:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.load(args.npy_path) * 10000 / 255)
        if args.clr_path:
            pcd.colors = o3d.utility.Vector3dVector(np.load(args.clr_path))
        ext = os.path.splitext(args.npy_path)[-1]
        filename = os.path.basename(args.npy_path)
    
    print('bound', pcd.get_axis_aligned_bounding_box())

    # down sample
    if args.down_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=args.down_size)
    print('pcd size: ', np.array(pcd.points).shape)

    # crop ceiling and floor
    # ceiling_y_threshold = args.ceiling_y * GT_T_SCALE
    # floor_y_threshold = args.floor_y * GT_T_SCALE
    floor_y_threshold = args.floor_y
    ceiling_y_threshold = args.ceiling_y
    floor_mask = np.array(pcd.points)[:, 1] > floor_y_threshold
    ceiling_mask = np.array(pcd.points)[:, 1] < ceiling_y_threshold
    cropped_pcd = pcd.select_by_index(np.where(ceiling_mask & floor_mask)[0])
    print('cropped pcd size: ', np.array(cropped_pcd.points).shape)

    # store cropped pcd
    if args.store_pcd:
        o3d.io.write_point_cloud(os.path.join(args.save_dir, filename.replace(ext, '.pcd')), cropped_pcd)

    # map
    points = np.array(cropped_pcd.points)
    colors = np.array(cropped_pcd.colors) 
    # ensure higher points cover lower points
    order_args = np.argsort(points[:, 1])
    points = points[order_args]
    colors = colors[order_args]
    # use pcd z-axis as image x-axis, pcd x-axis as image y-axis
    min_x, max_x = np.min(points[:, 2]), np.max(points[:, 2])
    min_y, max_y = np.min(points[:, 0]), np.max(points[:, 0])
    wh_ratio = (max_y - min_y) / (max_x - min_x)
    args.map_w_size = args.map_w_size
    dot_size = 5
    fig = plt.figure(figsize=(args.map_w_size, args.map_w_size * wh_ratio))
    plt.scatter(points[:, 2], points[:, 0], c=colors, marker='o', edgecolors='none', lw=0, s=dot_size, alpha=1)
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(f'{args.map_name}.png'), bbox_inches='tight', pad_inches=0)

    # map config
    map_img = plt.imread(os.path.join(f'{args.map_name}.png'))[:, :, :3]
    map_mask = np.all(map_img == 255, axis=-1)
    ocp_pos = np.argwhere(map_mask == 0)
    # pixel coordinates
    top_px = np.max(ocp_pos[:, 0])
    bottom_px = np.min(ocp_pos[:, 0])
    left_px = np.min(ocp_pos[:, 1])
    right_px = np.max(ocp_pos[:, 1])
    # Habitat coordinates
    top_pcd = np.min(points[:, 0])
    bottom_pcd = np.max(points[:, 0])
    left_pcd = np.min(points[:, 2])
    right_pcd = np.max(points[:, 2])

    w_scale = (right_pcd - left_pcd) / (right_px - left_px)
    h_scale = (bottom_pcd - top_pcd) / (bottom_px - top_px)
    # origin of the Habitat coordinates in pixel coordinates
    x_center_px = (0 - left_pcd) * 1/w_scale + left_px
    y_center_px = (0 - top_pcd) * 1/h_scale + top_px

    map_cfg = {
        'w_px2pcd_scale': w_scale,
        'h_px2pcd_scale': h_scale,
        'x_center_px': x_center_px,
        'y_center_px': y_center_px,
    }
    json.dump(map_cfg, open(f'{args.map_name}.json', 'w'))


    # Visualize
    if args.viz:
        o3d.visualization.draw_geometries(
            [cropped_pcd],
        )
