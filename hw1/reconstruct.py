import numpy as np
import open3d as o3d
import copy
import argparse
import glob
np.set_printoptions(suppress=True)

width, height = 512, 512
fov = 90
fx = width / 2 / np.tan(np.deg2rad(fov / 2))
fy = height / 2 / np.tan(np.deg2rad(fov / 2))
cx = width / 2
cy = height / 2
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])
# FIXME: what is the unit of depth_scale?
DEPTH_SCALE = 1
# scale ground truth translation to match the 3D scene
GT_T_SCALE = 1/10


def rotation_matrix_to_quaternion(R):
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]

    w = np.sqrt(1 + r11 + r22 + r33) / 2
    x = (r32 - r23) / (4 * w)
    y = (r13 - r31) / (4 * w)
    z = (r21 - r12) / (4 * w)

    return np.asarray([w, x, y, z])


def depth_image_to_point_cloud(rgb, depth):
    # ref: http://www.open3d.org/docs/release/tutorial/geometry/rgbd_image.html
    # Get point cloud from rgb and depth image
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
    return pcd


def preprocess_point_cloud(pcd, voxel_size):
    # ref: http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Voxel-downsampling
    # Do voxelization to reduce the number of points for less memory usage and speedup
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


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
    # fast global registration
    # distance_threshold = voxel_size * 0.5
    # result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
    #     source_down, target_down, source_fpfh, target_fpfh,
    #     o3d.pipelines.registration.FastGlobalRegistrationOption(
    #         maximum_correspondence_distance=distance_threshold))
    return result


def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    # ref: http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
    # Use Open3D ICP function to implement
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # Point-to-plane is more accurate
    # reg_p2l = o3d.pipelines.registration.registration_icp(
    #     source_down, target_down, threshold, trans_init,
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane()
    # )
    result = reg_p2p
    return result


def my_local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    # TODO: Write your own ICP function
    raise NotImplementedError
    return result


def reconstruct(args):
    # TODO: Return results
    data_root = args.data_root
    voxel_size = args.voxel_size

    if args.version == 'open3d':
        trans = local_icp_algorithm
    elif args.version == 'my_icp':
        trans = my_local_icp_algorithm
    else:
        raise NotImplementedError

    seq_len = len(glob.glob(f'{data_root}rgb/*.png'))

    T_0j = np.eye(4)
    poses = []
    merged_pcd = o3d.geometry.PointCloud()
    prev_down, prev_fpfh = None, None
    for i in range(1, seq_len+1):
        rgb_img = o3d.io.read_image(f'{data_root}rgb/{i}.png')
        dep_img = o3d.io.read_image(f'{data_root}depth/{i}.png')
        now_pcd = depth_image_to_point_cloud(rgb_img, dep_img)
        now_down, now_fpfh = preprocess_point_cloud(now_pcd, voxel_size)

        if prev_down is not None:
            # global registration
            reg_global = execute_global_registration(
                now_down, prev_down, now_fpfh, prev_fpfh, voxel_size)
            # local registration
            # FIXME: change to trans() later
            reg_p2p = local_icp_algorithm(
                now_down, prev_down, reg_global.transformation, threshold=voxel_size*0.4)

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
    pred_cam_poses = np.array(poses)
    return result_pcd, pred_cam_poses


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='my_icp', help='open3d or my_icp')
    parser.add_argument('--data_root', type=str, default='data_collection/first_floor/')
    parser.add_argument('--voxel_size', type=float, default=0.01)
    args = parser.parse_args()

    if args.floor == 1:
        args.data_root = "data_collection/first_floor/"
    elif args.floor == 2:
        args.data_root = "data_collection/second_floor/"

    # TODO: Output result point cloud and estimated camera pose
    '''
    Hint: Follow the steps on the spec
    '''
    result_pcd, pred_poses = reconstruct(args)

    # TODO: Calculate and print L2 distance
    '''
    Hint: Mean L2 distance = mean(norm(ground truth - estimated camera trajectory))
    '''
    # ground truth translation
    gt_poses = np.load(f'{args.data_root}GT_pose.npy')
    gt_ts = gt_poses[:, :3]
    gt_ts = gt_ts - gt_ts[0]
    # FIXME: why gt translation is 10 times larger than the 3D scene
    gt_ts = gt_ts * GT_T_SCALE
    # predicted translation
    pred_ts = pred_poses[:, :3]
    # L2 distance
    l2_distances = np.linalg.norm(gt_ts - pred_ts, axis=1)
    mean_l2_distance = np.mean(l2_distances)
    print("Mean L2 distance: ", mean_l2_distance)

    # TODO: Visualize result
    '''
    Hint: Should visualize
    1. Reconstructed point cloud
    2. Red line: estimated camera pose
    3. Black line: ground truth camera pose
    '''
    # ground truth trajectory
    gt_points = gt_ts
    gt_lines = [[i, i+1] for i in range(len(gt_points)-1)]
    gt_colors = [[0, 0, 0] for i in range(len(gt_lines))]
    gt_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(gt_points),
        lines=o3d.utility.Vector2iVector(gt_lines),
    )
    gt_lineset.colors = o3d.utility.Vector3dVector(gt_colors)

    # predicted trajectory
    pred_points = pred_ts
    pred_lines = [[i, i+1] for i in range(len(pred_points)-1)]
    pred_colors = [[1, 0, 0] for i in range(len(pred_lines))]
    pred_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pred_points),
        lines=o3d.utility.Vector2iVector(pred_lines),
    )
    pred_lineset.colors = o3d.utility.Vector3dVector(pred_colors)

    # Masking out the ceiling
    ceiling_y_threshold = 0.4 * GT_T_SCALE
    ceiling_mask = np.array(result_pcd.points)[:, 1] < ceiling_y_threshold
    cropped_pcd = result_pcd.select_by_index(np.where(ceiling_mask)[0])

    o3d.visualization.draw_geometries(
        [cropped_pcd, pred_lineset, gt_lineset],
    )
