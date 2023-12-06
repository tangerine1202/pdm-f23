import logging
import os 
import argparse
import json
import numpy as np
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

def read_png(path):
    return (plt.imread(path)[:,:,:3] * 255).astype(np.uint8)

def cv2_imshow(img, title='image'):
    if hasattr(img, 'shape') and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow(title, img)

def click_event(event, x, y, flags, params):
    global start_cfg
    if event == cv2.EVENT_LBUTTONDOWN:
        start_cfg = [y, x]
        logging.info(f'select start cfg {start_cfg}')
        tmp_img = img.copy()
        cv2.circle(tmp_img, (x, y), 3, (255, 0, 0), -1)
        cv2_imshow(tmp_img, title='start point')

def plot_edges(img, edges, type, opts):
    for edge in edges:
        x_from, y_from = edge[0]
        x_to, y_to = edge[1]
        clr = opts['rgb'][type]
        cv2.line (img, (y_from, x_from), (y_to, x_to), clr, opts['line_width'])
        cv2.circle(img, (y_from, x_from), opts['radius'], clr, -1)
        cv2.circle(img, (y_to, x_to), opts['radius'], clr, -1)
    return img

def save_masks(img, goal_map, occupancy_map):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('semantic map')
    plt.subplot(1, 3, 2)
    plt.imshow(goal_map, cmap='gray')
    plt.title('goal mask')
    plt.subplot(1, 3, 3)
    plt.imshow(occupancy_map, cmap='gray')
    plt.title('occupied mask')
    plt.savefig(f'masks.png')


class Tree:
    def __init__(self, start_cfg):
        self._nodes = [start_cfg]
        self._parent_ids = [0]

    def add_node(self, cfg, parent_id):
        self._nodes.append(cfg)
        self._parent_ids.append(parent_id)

    def get_path(self, goal_cfg):
        if not self.is_in_tree(goal_cfg):
            raise ValueError('goal_cfg not in tree')

        path = []
        curr_id = np.argwhere([np.array_equal(cfg, goal_cfg) for cfg in self._nodes])[0][0]
        curr_cfg = self._nodes[curr_id]
        while curr_id != 0:
            path.append(curr_cfg)
            curr_id = self._parent_ids[curr_id]
            curr_cfg = self._nodes[curr_id]
        path.reverse()
        return np.array(path)
    
    def get_nodes(self):
        return np.array(self._nodes)
    
    def get_edges(self):
        edges = []
        # NOTE: skip start node, since it has no parent
        for i in range(1, len(self._nodes)):
            edges.append([self._nodes[self._parent_ids[i]], self._nodes[i]])
        return np.array(edges)
    
    def is_in_tree(self, cfg):
        return np.any(cfg == self._nodes)
    

def sample_cfg(cfgs, goal_cfg, p_goal):
    if np.random.rand() < p_goal:
        q_rand = goal_cfg[np.random.randint(0, len(goal_cfg))]
    else:
        q_rand = cfgs[np.random.randint(0, len(cfgs))]
    return q_rand

def find_nearest_cfgs(node_cfgs, q_rand):
    dist = np.linalg.norm(node_cfgs - q_rand, axis=1)
    near_id = np.argmin(dist)
    q_near = node_cfgs[near_id]
    return q_near, near_id

def extend_q(q_near, q_rand, delta_q):
    dist = np.linalg.norm(q_rand - q_near)
    if dist < delta_q:
        q_new = q_rand
    else:
        q_new = q_near + (q_rand - q_near) * delta_q / dist
        # convert to a valid cfg
        q_new = q_new.astype(int)
    return q_new


def has_collision_free_path(q_near, q_new, ocp_map, n_steps):
    x_near, y_near = q_near
    x_new, y_new = q_new
    x_delta = x_new - x_near
    y_delta = y_new - y_near
    # NOTE: n_steps should >= delta_q to ensure no collision
    x_step = x_delta / n_steps
    y_step = y_delta / n_steps
    for i in range(n_steps):
        x = np.round(x_near + i * x_step).astype(int)
        y = np.round(y_near + i * y_step).astype(int)
        if ocp_map[x, y] == 1:
            return False
    return True

def is_valid(q_near, q_new, ocp_map, tree, n_steps=100):
    if np.any(q_new < 0) or np.any(q_new >= ocp_map.shape):
        raise ValueError('q_new out of bound')
    if ocp_map[q_new[0], q_new[1]] == 1:
        logging.debug('q_new is occupied')
        return False
    if tree.is_in_tree(q_new):
        logging.debug('q_new is in tree')
        return False
    if not has_collision_free_path(q_near, q_new, ocp_map, n_steps):
        logging.debug('no collision-free path')
        return False
    return True

def is_goal(cfg, tgt_cfgs):
    for tgt_cfg in tgt_cfgs:
        if np.array_equal(cfg, tgt_cfg):
            return True, cfg
    return False, None

def is_valid_start_cfg(cfg, ocp_map):
    if cfg is None:
        return False
    if ocp_map[cfg[0], cfg[1]] == 1:
        logging.critical('start cfg is occupied, please select another one')
        return False
    return True


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s [%(levelname)s] %(message)s', 
        datefmt='%H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument('--goal-name', type=str, 
                        help='target object name, options: [rack, cushion, lamp, stair, cooktop]', 
                        choices=['rack', 'cushion', 'lamp', 'stair', 'cooktop'],
                        default='lamp')
    parser.add_argument('--map-path', type=str, help='path to map image', default='map.png')
    parser.add_argument('--map-cfg-path', type=str, help='path to map cfg', default='map.json')
    parser.add_argument('--name2rgb_path', type=str, help='path to name2rgb json', default='name2rgb.json')
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument('--max-step', type=int, help='max step', default=10000)
    parser.add_argument('--p-goal', type=float, help='p_goal', default=0.5)
    parser.add_argument('--delta-q', type=float, help='delta_q', default=30)
    parser.add_argument('--goal-thresh', type=float, help='threshold for goal region', default=7)
    parser.add_argument('--collision-thresh', type=float, help='threshold for collision', default=3)
    args = parser.parse_args()

    if args.goal_name == 'cooktop':
        if args.goal_thresh < 10:
            logging.warning('goal_thresh should >= 10 for cooktop')
            # raise ValueError('goal_thresh should >= 10 for cooktop')

    np.random.seed(args.seed)
    name2rgb = json.load(open(args.name2rgb_path, 'r'))
    goal_rgb = np.array(name2rgb[args.goal_name])

    img = read_png(args.map_path)
    goal_map = np.all(img == goal_rgb, axis=-1).astype(np.uint8)
    occupancy_map = np.any(img != 255, axis=-1).astype(np.uint8)
    # enlarge goal region and collision region
    goal_map = cv2.dilate(goal_map, np.ones((args.goal_thresh, args.goal_thresh)))
    occupancy_map = cv2.dilate(occupancy_map, np.ones((args.collision_thresh, args.collision_thresh)))

    logging.info(f'map size {img.shape[:2]}')
    if goal_map.sum() == 0:
        raise ValueError('goal region is empty')


    # choose starting point
    start_cfg = None
    cv2.imshow('start point', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.setMouseCallback('start point', click_event)
    while not is_valid_start_cfg(start_cfg, occupancy_map):
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    logging.info(f'start_cfg {start_cfg}')


    # build configuration space
    goal_cfgs = np.argwhere(goal_map)
    # NOTE: tgt_cfgs should be valid
    cfgs = np.argwhere(occupancy_map == 0 | goal_map)
    save_masks(img, goal_map, occupancy_map==0 | goal_map)


    # RRT
    tree = Tree(start_cfg)
    found_path = False
    for total_step in tqdm(range(args.max_step)):
        q_rand = sample_cfg(cfgs, goal_cfgs, args.p_goal)
        q_near, q_near_id = find_nearest_cfgs(tree.get_nodes(), q_rand)
        q_new = extend_q(q_near, q_rand, args.delta_q)
        if is_valid(q_near, q_new, occupancy_map, tree, args.delta_q):
            tree.add_node(q_new, q_near_id)
            found_path, found_goal_cfg = is_goal(q_new, goal_cfgs)
            if found_path:
                break

    logging.info(f'using {total_step} steps')
    logging.info(f'tree size {len(tree.get_nodes())}')
    if found_path:
        paths = tree.get_path(found_goal_cfg)
        logging.info(f'found path size {len(paths)}')
        np.save('path.npy', paths)

        # optimize path
        improved_paths = [paths[0]]
        i = 0
        while i < len(paths) - 1:
            for j in range(i + 1, len(paths)):
                x_from, y_from = paths[i]
                x_to, y_to = paths[j]
                if not has_collision_free_path((x_from, y_from), (x_to, y_to), occupancy_map, args.delta_q):
                    j -= 1
                    break
            i = j
            improved_paths.append(paths[i])
            if i == len(paths) - 1:
                break
        logging.info(f'optimized path size {len(improved_paths)}')
        np.save('improved_path.npy', improved_paths)

    # viz
    plot_opts = {
        'rgb': { 
            'tree': (0, 0, 0), 
            'path':  (255, 0, 0), 
            'imp_path': (255, 155, 0),
            'start': (0, 0, 0),
            'goal': (0, 0, 0),
        },
        'line_width': 2,
        'radius': 3,
        'figsize': (8, 6),
    }
    plt.figure(figsize=plot_opts['figsize'])
    rrt_img = img.copy()
    rrt_img = plot_edges(rrt_img, tree.get_edges().tolist(), 'tree', plot_opts)

    if found_path:
        # log path
        # for i in range(len(paths) - 1):
        #     x_from, y_from = paths[i]
        #     x_to, y_to = paths[i+1]
        #     logging.info(f'({x_from}, {y_from}) -> ({x_to}, {y_to})')

        # plot path
        path_edges = [[paths[i], paths[i+1]] for i in range(len(paths) - 1)]
        rrt_img = plot_edges(rrt_img, path_edges, 'path', plot_opts)
        if improved_paths:
            improved_path_edges = [[improved_paths[i], improved_paths[i+1]] for i in range(len(improved_paths) - 1)]
            rrt_img = plot_edges(rrt_img, improved_path_edges, 'imp_path', plot_opts)
        cv2.putText(rrt_img, 's', (paths[0][1], paths[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, plot_opts['rgb']['start'], 2, cv2.LINE_AA)
        cv2.putText(rrt_img, 'e', (paths[-1][1], paths[-1][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, plot_opts['rgb']['goal'], 2, cv2.LINE_AA)

    plt.axis('equal')
    plt.tight_layout(pad=0)
    plt.imshow(rrt_img)
    plt.savefig('rrt.png', bbox_inches='tight', pad_inches=0)
