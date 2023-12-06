import json
import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import os
import sys
import argparse
import shutil
from scipy.spatial.transform import Rotation as scipy_R


# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
test_scene = "replica_v1/apartment_0/habitat/mesh_semantic.ply"

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
}

cam_extr = []

# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def transform_depth(image):
    depth_img = (image / 10 * 255).astype(np.uint8)
    return depth_img


def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img


def make_simple_cfg(settings, forward_step_size=0.25, turn_angle=10.0):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount = forward_step_size),
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount = turn_angle),
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount = turn_angle),
        )
    }

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # depth sensor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # semantic sensor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def navigateAndSee(action=""): #, data_root='data_collection/second_floor/'):
    global count
    observations = sim.step(action)
    # print("action: ", action)

    cv2.imshow("RGB", transform_rgb_bgr(observations["color_sensor"]))
    # cv2.imshow("depth", transform_depth(observations["depth_sensor"]))
    # cv2.imshow("semantic", transform_semantic(observations["semantic_sensor"]))
    agent_state = agent.get_state()
    sensor_state = agent_state.sensor_states['color_sensor']
    # print("Frame:", count)
    # print("camera pose: x y z rw rx ry rz")
    # print(sensor_state.position[0], sensor_state.position[1], sensor_state.position[2],
    #       sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)

    count += 1
    # cv2.imwrite(data_root + f"rgb/{count}.png", transform_rgb_bgr(observations["color_sensor"]))
    # cv2.imwrite(data_root + f"depth/{count}.png", transform_depth(observations["depth_sensor"]))
    # cv2.imwrite(data_root + f"semantic/{count}.png", transform_semantic(observations["semantic_sensor"]))
    # np.save(data_root + f"instance/{count}.npy", observations["semantic_sensor"])

    # cam_extr.append([sensor_state.position[0], sensor_state.position[1], sensor_state.position[2],
    #                 sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z])
    return sensor_state.position[0], sensor_state.position[1], sensor_state.position[2], sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z

def get_agent_pose():
    agent_state = agent.get_state()
    sensor_state = agent_state.sensor_states['color_sensor']
    t = np.array(sensor_state.position)
    q = scipy_R.from_quat(np.array([sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z, sensor_state.rotation.w]))
    r = q.as_euler('yxz', degrees=True)[0]
    return t, r

def get_relative_pose(src_t, src_r, tgt_t):
    rel_t = tgt_t - src_t
    rel_dist = np.linalg.norm(rel_t[[0, 2]])
    tgt_q = scipy_R.from_euler('y', np.arctan2(rel_t[0], rel_t[2]) + np.pi)
    tgt_r = tgt_q.as_euler('yxz', degrees=True)[0]
    if tgt_r > src_r:
        if tgt_r - src_r > 180:
            rel_r = tgt_r - src_r - 360
        else:
            rel_r = tgt_r - src_r
    else:
        if src_r - tgt_r > 180:
            rel_r = tgt_r - src_r + 360
        else:
            rel_r = tgt_r - src_r
    return rel_dist, rel_r

def get_next_action(rel_dist, rel_r, t_thresh=0.25, r_thresh=10):
    if np.abs(rel_r) > r_thresh:
        if rel_r > 0:
            return "turn_left"
        else:
            return "turn_right"
    elif rel_dist > t_thresh:
        return "move_forward"
    else:
        return "finish"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map-path', type=str, help='path to map image', default='map.png')
    parser.add_argument('--map-cfg-path', type=str, help='path to map cfg', default='map.json')
    parser.add_argument('--nav-path', type=str, help='path to path.npy', default='path.npy')
    parser.add_argument('--name2rgb_path', type=str, help='path to name2rgb json', default='name2rgb.json')
    parser.add_argument('--step-size', type=float, help='forward step size', default=0.2)
    parser.add_argument('--turn-angle', type=float, help='turn angle', default=10.0)
    parser.add_argument('--t-thresh', type=float, help='translation threshold', default=0.3)
    parser.add_argument('--r-thresh', type=float, help='rotation threshold', default=10.0)
    args = parser.parse_args()

    # Convert path to the Habitat coordinate system
    map_cfg = json.load(open(args.map_cfg_path, 'r'))
    px2pcd_scale = np.array([map_cfg['h_px2pcd_scale'], map_cfg['w_px2pcd_scale']])
    center_px = np.array([map_cfg['y_center_px'], map_cfg['x_center_px']])
    paths_in_px = np.load(args.nav_path)
    paths = (paths_in_px - center_px) * px2pcd_scale
    paths = np.concatenate((paths, np.zeros((paths.shape[0], 1))), axis=1)[:, [0, 2, 1]]

    cfg = make_simple_cfg(sim_settings, forward_step_size=args.step_size, turn_angle=args.turn_angle)
    sim = habitat_sim.Simulator(cfg)
    # initialize an agent
    agent = sim.initialize_agent(sim_settings["default_agent"])
    # Set agent state
    agent_state = habitat_sim.AgentState()
    # Set starting point
    agent_state.position = np.array(paths[0])  # agent in world space
    agent.set_state(agent_state)
    # obtain the default, discrete actions that an agent can perform
    # default action space contains 3 actions: move_forward, turn_left, and turn_right
    action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
    print("Discrete action space: ", action_names)


    # if args.floor == 1:
    #     data_root = "data_collection/first_floor/"
    # elif args.floor == 2:
    #     data_root = "data_collection/second_floor/"

    # if os.path.isdir(data_root):
    #     shutil.rmtree(data_root)  # WARNING: this line will delete whole directory with files

    # for sub_dir in ['rgb/', 'depth/', 'semantic/', 'instance/']:
    #     os.makedirs(data_root + sub_dir)

    count = 0
    action = "move_forward"
    navigateAndSee(action)

    FORWARD_KEY = "w"
    LEFT_KEY = "a"
    RIGHT_KEY = "d"
    FINISH = "f"
    print("#############################")
    print(" f for finish and quit the program")
    print("--- Manual Control ---")
    print(" w for go forward  ")
    print(" a for turn left  ")
    print(" d for turn right  ")
    print("--- Navigation by RRT ---")
    print(" j for RRT navigation")
    print("#############################")

    subgoal_idx = 1
    subgoal_t = paths[subgoal_idx]
    curr_t, curr_r = get_agent_pose()
    relative_dist, relative_r = get_relative_pose(curr_t, curr_r, subgoal_t)

    while True:
        keystroke = cv2.waitKey(0)

        if keystroke == ord(FINISH):
            print("action: FINISH")
            break
        # --- Manual Control ---
        elif keystroke == ord(FORWARD_KEY):
            action = "move_forward"
            navigateAndSee(action)
            print("action: FORWARD")
        elif keystroke == ord(LEFT_KEY):
            action = "turn_left"
            navigateAndSee(action)
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = "turn_right"
            navigateAndSee(action)
            print("action: RIGHT")
        # --- Navigation by RRT ---
        elif keystroke == ord('j'):
            curr_t, curr_r = get_agent_pose()
            relative_dist, relative_r = get_relative_pose(curr_t, curr_r, subgoal_t)
            action = get_next_action(relative_dist, relative_r, t_thresh=args.t_thresh, r_thresh=args.r_thresh)
            while action == 'finish':
                subgoal_idx += 1
                if subgoal_idx >= len(paths):
                    print("FINISH")
                    break
                subgoal_t = paths[subgoal_idx]
                curr_t, curr_r = get_agent_pose()
                relative_dist, relative_r = get_relative_pose(curr_t, curr_r, subgoal_t)
                action = get_next_action(relative_dist, relative_r, t_thresh=args.t_thresh, r_thresh=args.r_thresh)
            print("action: ", action)
            navigateAndSee(action)
        else:
            print("INVALID KEY")
            continue
        
        curr_t, curr_r = get_agent_pose()
        relative_dist, relative_r = get_relative_pose(curr_t, curr_r, subgoal_t)
        print('relative pose: ', relative_dist, relative_r)
        print(f'----- {subgoal_idx} / {len(paths) - 1} -----')
    


    # np.save(data_root + 'GT_pose.npy', np.asarray(cam_extr))
