import glob
import copy
from collections import Counter

import numpy as np
import open3d as o3d


width, height = 512, 512
fov = 90
fx = width / 2 / np.tan(np.deg2rad(fov / 2))
fy = height / 2 / np.tan(np.deg2rad(fov / 2))
cx = width / 2
cy = height / 2
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])
K_inv = np.array([[1 / fx, 0, - cx / fx],
                  [0, 1 / fy, - cy / fy],
                  [0, 0, 1]])
DEPTH_SCALE = 1
GT_T_SCALE = 1 / 10


def rotation_matrix_to_quaternion(R):
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]

    w = np.sqrt(1 + r11 + r22 + r33) / 2
    x = (r32 - r23) / (4 * w)
    y = (r13 - r31) / (4 * w)
    z = (r21 - r12) / (4 * w)

    return np.asarray([w, x, y, z])


def depth_image_to_point_cloud(rgb, depth, method='my'):
    # Get point cloud from rgb and depth image
    if method == 'my':
        # normalize
        colors = np.array(rgb).reshape(-1, 3) / 255  # (n, 3)
        z = np.array(depth).reshape(-1, 1) / 255 * DEPTH_SCALE  # (n, 1)

        # depth un-projection
        # NOTE: X = (z @ K_inv @ pixels) is slower than the following implementation
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        u = u.reshape(-1, 1)  # (n, 1)
        v = v.reshape(-1, 1)  # (n, 1)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pts = np.concatenate([x, y, z], axis=1)

        # wrap into open3d point cloud
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]]
        )
    elif method == 'open3d':
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, intrinsic_matrix=K)
        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth, depth_scale=DEPTH_SCALE, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, camera_intrinsic)
        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]]
        )
    else:
        raise NotImplementedError
    return pcd


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    # ref: http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def local_icp_algorithm(source_down, target_down, trans_init, threshold, method='my', max_iter=30, relative_fitness=1e-6, relative_rmse=1e-6):
    # ref: http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
    # Use Open3D ICP function to implement
    if method == 'my':
        result = my_local_icp_algorithm(
            source_down, target_down, threshold, trans_init,
            max_iter, relative_fitness, relative_rmse
        )
    elif method == 'open3d':
        result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iter,
                relative_fitness=relative_fitness,
                relative_rmse=relative_rmse)
        )
    else:
        raise NotImplementedError
    return result


def my_local_icp_algorithm(source_down, target_down, trans_init, threshold, max_iter=30, relative_fitness=1e-6, relative_rmse=1e-6):
    # ref slides: https://cs.gmu.edu/~kosecka/cs685/cs685-icp.pdf
    pts_src = np.array(source_down.points)  # n x 3
    pts_tgt = np.array(target_down.points)  # m x 3

    if len(pts_tgt) < 1000:
        max_depth = 0
    elif len(pts_tgt) < 3000:
        max_depth = 1
    elif len(pts_tgt) < 5000:
        max_depth = 2
    elif len(pts_tgt) < 10000:
        max_depth = 3
    else:
        max_depth = 4
    tgt_ids = np.arange(len(pts_tgt))
    tgt_octree = MyOcTree(pts_tgt, tgt_ids, max_depth=max_depth)

    # one-iter transformation
    T = trans_init
    R, t = T[:3, :3], T[:3, 3]
    # accumulated transformation
    T_accu = np.array(T)

    prev_fitness = 0
    prev_rmse = np.inf
    # for i in tqdm(range(max_iter)):
    for _ in range(max_iter):
        pts_src = (R @ pts_src.T).T + t  # n x 3

        # nearest matching
        corrs = my_find_nearest_neighbors_with_octree(pts_src, tgt_octree)

        # inlier selection
        inlier_pts_src = pts_src[corrs[:, 0]]
        inlier_pts_tgt = pts_tgt[corrs[:, 1]]
        inlier_pts_src, inlier_pts_tgt = my_inlier_selection(inlier_pts_src, inlier_pts_tgt, threshold)

        # transformation estimation
        R, t = my_compute_transformation(inlier_pts_src, inlier_pts_tgt)

        # update
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        T_accu = T @ T_accu

        # check convergence
        fitness = len(inlier_pts_src) / len(pts_src)
        inlier_rmse = my_compute_rmse(inlier_pts_src, inlier_pts_tgt)
        if (fitness - prev_fitness) < relative_fitness and (prev_rmse - inlier_rmse) < relative_rmse:
            break
        prev_rmse = inlier_rmse
        prev_fitness = fitness

    # wrap into open3d registration result
    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = T_accu
    result.fitness = fitness
    result.inlier_rmse = inlier_rmse
    return result


def my_find_nearest_neighbors_with_octree(src, octree):
    tgt_ids = octree.find_batch_nearest_neighbor(src)
    corrs = np.vstack([np.arange(src.shape[0]), tgt_ids]).T  # n x 2
    return corrs


def my_compute_rmse(src, dst):
    rmse = np.mean(np.linalg.norm(src - dst, axis=1))
    return rmse


def my_inlier_selection(src, tgt, threshold):
    dist = np.linalg.norm(src - tgt, axis=1)
    inlier_mask = dist < threshold
    if inlier_mask.sum() < 3:
        print('[Warning] inlier_mask.sum() < 3, please adjust the threshold')
        return src, tgt
    inlier_src = src[inlier_mask]
    inlier_tgt = tgt[inlier_mask]
    return inlier_src, inlier_tgt


def my_compute_transformation(src, tgt):
    # centroid (center of mass)
    centroid_src = np.mean(src, axis=0)  # 3
    centroid_tgt = np.mean(tgt, axis=0)  # 3
    centered_src = src - centroid_src  # n x 3
    centered_dst = tgt - centroid_tgt  # n x 3
    # SVD
    # NOTE: due to the matrix shape, the transpose is different from the formula
    W = centered_dst.T @ centered_src   # 3 x 3
    U, _, Vt = np.linalg.svd(W)
    # transform
    R = U @ Vt
    t = centroid_tgt - R @ centroid_src
    return R, t


class MyOcTreeNode():
    def __init__(self, min_corner, max_corner):
        self.min_corner = min_corner
        self.max_corner = max_corner
        self.points = np.empty((0, 3))
        self.ids = np.zeros((0), dtype=int)
        self.children = [None] * 8


class MyOcTree():
    def __init__(self, points, ids, max_depth=3):
        self.max_depth = max_depth
        self.root = MyOcTreeNode(np.min(points, axis=0), np.max(points, axis=0))
        self.build_octree(self.root, 0, points, ids)

    def build_octree(self, node, depth, points, ids):
        node.points = points.copy()
        node.ids = ids.copy()
        if depth < self.max_depth:
            self.subdivide(node)
            for child in node.children:
                min_corner, max_corner = child.min_corner, child.max_corner
                masks = np.all(points >= min_corner, axis=1) & np.all(points < max_corner, axis=1)
                child_points = points[masks]
                child_ids = ids[masks]
                self.build_octree(child, depth + 1, child_points, child_ids)

    def subdivide(self, node):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    min_corner = np.array([i, j, k]) * (node.max_corner - node.min_corner) / 2 + node.min_corner
                    max_corner = np.array([i + 1, j + 1, k + 1]) * (node.max_corner -
                                                                    node.min_corner) / 2 + node.min_corner
                    child_idx = i * 4 + j * 2 + k
                    child_node = MyOcTreeNode(min_corner, max_corner)
                    node.children[child_idx] = child_node

    def find_batch_nearest_neighbor(self, points):
        # initialize corresponding nodes with root
        corr_nodes = np.array([self.root] * points.shape[0])
        in_cuboid_masks = np.ones(points.shape[0], dtype=bool)
        self._find_batch_nearest_neighbor(self.root, 0, points, in_cuboid_masks, corr_nodes)

        tgt_ids = np.zeros(points.shape[0], dtype=int)
        # group points by corresponding nodes
        corr_nodes_counter = Counter(corr_nodes)
        for node_to_search in corr_nodes_counter.keys():
            # select points with the corresponding node
            corr_masks = (corr_nodes == node_to_search)
            corr_points = points[corr_masks]

            # find nearest neighbor
            dist = np.linalg.norm(
                np.expand_dims(corr_points, axis=1) - np.expand_dims(node_to_search.points, axis=0),
                axis=2
            )
            nearest_masks = np.argmin(dist, axis=1)
            nearest_ids = node_to_search.ids[nearest_masks]
            tgt_ids[corr_masks] = nearest_ids
        return tgt_ids

    def _find_batch_nearest_neighbor(self, node, depth, points, in_cuboid_masks, corr_nodes):
        in_cuboid_masks = in_cuboid_masks & (np.all(points >= node.min_corner, axis=1)
                                             & np.all(points < node.max_corner, axis=1))
        # update the correspondence node
        corr_nodes[in_cuboid_masks] = node
        if depth < self.max_depth:
            for child in node.children:
                if len(child.points) > 0:
                    self._find_batch_nearest_neighbor(child, depth + 1, points, in_cuboid_masks, corr_nodes)
