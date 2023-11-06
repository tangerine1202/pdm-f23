import os
import argparse
import numpy as np
import open3d as o3d

GT_T_SCALE = 0.1

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path of pcd file')
args = parser.parse_args()

pcd = o3d.io.read_point_cloud(args.path)

ceiling_y_threshold = -1.0 * GT_T_SCALE
ceiling_mask = np.array(pcd.points)[:, 1] < ceiling_y_threshold
cropped_pcd = pcd.select_by_index(np.where(ceiling_mask)[0])

# # Visualize
o3d.visualization.draw_geometries(
    [cropped_pcd],
)
