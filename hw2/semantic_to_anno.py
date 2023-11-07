import os
import argparse
import json
import numpy as np
from PIL import Image
from scipy.io import loadmat

def load_scene_semantic_dict(dataset_path, scene):
    with open(os.path.join(dataset_path, scene, 'habitat', 'info_semantic.json'), 'r') as f:
        return json.load(f)

def fix_semantic_observation(instance_observation, scene_dict):
    # The labels of images collected by Habitat are instance ids
    # transfer instance to semantic
    instance_id_to_semantic_label_id = np.array(scene_dict["id_to_label"])
    semantic = instance_id_to_semantic_label_id[instance_observation]
    semantic_img = Image.new("L", (semantic.shape[1], semantic.shape[0]))
    semantic_img.putdata(semantic.flatten())
    return semantic_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inst_path', type=str, help='path to instance images dir')
    parser.add_argument('--output_path', type=str, default=None,
                          help='path to output dir, default will be anno/ at the same dir of rgb dir')
    parser.add_argument("--dataset", type=str, default='./replica_v1', help="Folder containing Replica dataset")
    parser.add_argument('--scene', type=str, default='apartment_0', help='scene name')
    args = parser.parse_args()

    scene_dict = load_scene_semantic_dict(args.dataset, args.scene)
    inst_dir = os.path.basename(os.path.normpath(args.inst_path))
    inst_path = args.inst_path
    output_path = args.output_path

    if output_path is None:
        output_path = inst_path.replace(inst_dir, 'anno')
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
            print(f'Created {output_path}')

    for fname in os.listdir(inst_path):
        fname, fext = os.path.splitext(fname)
        inst_fpath = os.path.join(inst_path, f'{fname}{fext}')
        anno_fpath = os.path.join(output_path, f'{fname}.png')
        inst = np.load(inst_fpath)
        anno_img = fix_semantic_observation(inst, scene_dict)
        # save semantic images
        anno_img.save(anno_fpath)
