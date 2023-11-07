# ref: https://hackmd.io/wNGlmMq2RC-lY3l8JhO4SA?view
import argparse
import os
import cv2
import json
from tqdm.auto import tqdm


def odgt(img_path, img_dir, anno_dir):
    seg_path = img_path.replace(img_dir, anno_dir)
    seg_path = seg_path.replace('.jpg', '.png')

    if os.path.exists(seg_path):
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        odgt_dic = {}
        odgt_dic["fpath_img"] = img_path
        odgt_dic["fpath_segm"] = seg_path
        odgt_dic["width"] = h
        odgt_dic["height"] = w
        return odgt_dic
    else:
        # print('the corresponded annotation does not exist')
        # print(img_path)
        return None


if __name__ == "__main__":
    """
    Assume the data directory is like:
    DATA_DIR ____
                |
                ___ annotations ______ modes[0]
                |                 |___ modes[1]
                |
                ___ images ______ modes[0]
                            |____ modes[1]
    # generated file
    saves[0]
    saves[1]
    """
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--data_dir', type=str, default='./data')
    argparse.add_argument('--modes', type=str, nargs='+', default=['train', 'val'])
    argparse.add_argument('--saves', type=str, nargs='+', default=['my_training.odgt', 'my_validation.odgt'])
    argparse.add_argument('--img_dir', type=str, default='images')
    argparse.add_argument('--anno_dir', type=str, default='annotations')
    args = argparse.parse_args()


    DATA_DIR = args.data_dir

    for i, mode in enumerate(args.modes):
        save = args.saves[i]
        dir_path = f"{DATA_DIR}/{args.img_dir}/{mode}"
        img_list = os.listdir(dir_path)
        img_list.sort()
        img_list = [os.path.join(dir_path, img) for img in img_list]

        with open(f'{DATA_DIR}/{save}', mode='wt', encoding='utf-8') as myodgt:
            for i, img in tqdm(enumerate(img_list)):
                a_odgt = odgt(img, args.img_dir, args.anno_dir)
                if a_odgt is not None:
                    myodgt.write(f'{json.dumps(a_odgt)}\n')
