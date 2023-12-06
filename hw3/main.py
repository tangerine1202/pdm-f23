import subprocess
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--goal-name', type=str, help='name of goal')
    parser.add_argument('--path-type', type=str, choices=['rrt', 'smooth'], help='type of path, options=[rrt, smooth]', default='smooth')
    parser.add_argument('--goal-thresh', type=int, help='threshold for goal region', default=3)
    args = parser.parse_args()

    if args.goal_name == 'cooktop':
        if args.goal_thresh < 10:
            print('[WARNING] recommend use goal_thresh >= 10 for cooktop')
            if input(f'continue with goal-thresh={args.goal_thresh} ? [y/n] ') != 'y':
                exit()

    map_cmd = 'python construct_map.py --npy_path semantic_3d_pointcloud/point.npy --clr_path semantic_3d_pointcloud/color01.npy'
    rrt_cmd = f'python generate_path.py --goal-name {args.goal_name} --goal-thresh {args.goal_thresh}'
    nav_cmd = f'python navigate.py --path-type {args.path_type} --record True'
    subprocess.call(map_cmd.split(' '), shell=False)
    subprocess.call(rrt_cmd.split(' '), shell=False)
    subprocess.call(nav_cmd.split(' '), shell=False)