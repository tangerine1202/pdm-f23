import os
import glob
import argparse

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from scipy.io import loadmat
from tqdm.auto import tqdm

from mit_semseg.config import cfg
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode
from mit_semseg.lib.utils import as_numpy
# from semantic_segmentation_pytorch.mit_semseg.config import cfg
# from semantic_segmentation_pytorch.mit_semseg.models import ModelBuilder, SegmentationModule
# from semantic_segmentation_pytorch.mit_semseg.utils import colorEncode
# from semantic_segmentation_pytorch.mit_semseg.lib.utils import as_numpy


colors = loadmat('./color101.mat')['colors']


def setup_model(cfg):
    print('weight:', cfg.MODEL.weights_encoder, cfg.MODEL.weights_decoder)
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.cuda()
    return segmentation_module


def get_img_resized_list(img, cfg):
    ori_height, ori_width = img.size

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    def round2nearest_multiple(x, p):
        return ((x - 1) // p + 1) * p

    def img_transform(img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = normalize(torch.from_numpy(img.copy()))
        return img

    img_resized_list = []
    for this_short_size in cfg.DATASET.imgSizes:
        # calculate target height and width
        scale = min(this_short_size / float(min(ori_height, ori_width)),
                    cfg.DATASET.imgMaxSize / float(max(ori_height, ori_width)))
        target_height, target_width = int(ori_height * scale), int(ori_width * scale)

        # to avoid rounding in network
        target_width = round2nearest_multiple(target_width, cfg.DATASET.padding_constant)
        target_height = round2nearest_multiple(target_height, cfg.DATASET.padding_constant)

        # resize images
        img_resized = img.resize((target_width, target_height), Image.BILINEAR)

        # image transform, to torch float tensor 3xHxW
        img_resized = img_transform(img_resized)
        img_resized = torch.unsqueeze(img_resized, 0)
        img_resized_list.append(img_resized)
    return img_resized_list


def inference(model, rgb_img, cfg):
    segSize = rgb_img.size
    img_resized_list = get_img_resized_list(rgb_img, cfg)

    with torch.no_grad():
        scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1]).cuda()
        for img in img_resized_list:
            feed_dict = {'img_data': img.cuda()}
            # forward pass
            scores_tmp = model(feed_dict, segSize=segSize)
            scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)
        _, pred = torch.max(scores, dim=1)
        pred = as_numpy(pred.squeeze(0).cpu())
    return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
    )
    parser.add_argument(
        "--cfg",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        '-f', '--floor',
        type=int,
        default=1
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default=None
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    if args.data_root == None:
        if args.floor == 1:
            args.data_root = './data_collection/first_floor/'
        elif args.floor == 2:
            args.data_root = './data_collection/second_floor/'

    # Prepare seg dir
    seg_dir_name = 'seg'
    seg_dir = f'{args.data_root}{seg_dir_name}/'
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir, exist_ok=True)

    model = setup_model(cfg)

    rgb_paths = glob.glob(f'{args.data_root}rgb/*.png')
    rgb_paths = sorted(rgb_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    for rgb_path in tqdm(rgb_paths):
        seg_path = rgb_path.replace('rgb', seg_dir_name)
        rgb = Image.open(rgb_path).convert('RGB')
        pred = inference(model, rgb, cfg)
        seg = colorEncode(pred, colors).astype(np.uint8)
        Image.fromarray(seg).save(seg_path)
