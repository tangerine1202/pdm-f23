# pdm-f23-hw2

NYCU Perception and Decision Making 2023 Fall

Spec: [Google Docs](https://drive.google.com/file/d/1LdzOZnM4sa_z1dcEKYHdXxHH_FsDKr_h/view?usp=sharing)

## Reconstruction

To run my reconstruction pipeline, please make sure semantic segmentation images are placed in `seg/` folder, located at the same directory as rgb images.

For example:
```
data_collection
|-- first_floor
|   |-- seg
|   |   |-- 000000.png
|   |   |-- ...
|   |-- rgb
|   |   |-- 000000.png
|   |   |-- ...
|   |-- ...
```

I have provided the `generate_seg.py` script to generate semantic segmentation images from rgb images.

```bash
# generate segmentation for reconstruction
# assume rgb images are placed at "data_collection/first_floor/rgb/"
python generate_seg.py --cfg ./ckpt/model_apartment_0/config.yaml -f 1 

# generate reconstruction with semantic segmentation
python reconstruction.py -f 1 --voxel_size 0.01 --color_src seg
# generate reconstruction with rgb image
python reconstruction.py -f 1 --voxel_size 0.01 --color_src rgb
```

## Training

**NOTE**: The modified semantic_segmentation_pytorch package is not included in the uploaded files, so the following commands may not work.

```bash
# install the customized semantic_segmentation_pytorch package
pip install -e semantic_segmentation_pytorch


# generate data for training
python data_generator_.py --output ./data --train_frames_per_room 100
python generate_odgt.py --data_dir ./data

# training
python semantic_segmentation_pytorch/train.py --cfg ./ckpt/model_apartment_0/config.yaml --gpus 0

# collet data from modified load.py in hw1 for evaluation
cd ../hw1
python load.py -f 1
# generate annotation for evaluation
python semantic_to_anno.py --inst_path data_collection/first_floor/instances
# generate odgt for evaluation
python generate_odgt --data_dir data_collection/first_floor --modes "" --save "my.odgt" --img_dir rgb --anno_dir anno

# eval
python semantic_segmentation_pytorch/eval_multipro.py --cfg ./ckpt/model_apartment_0/config.yaml --gpus 0 --val-odgt data_collection/first_floor/my.odgt
```

# SETUP (Original README)

## Preparation
In your original dpm-f23 directory, `git pull` to get new `hw2` directory. 

If you want to use [semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch), you may face the problem that the page of pretrain model are not accessible.
Please check the [issue](https://github.com/CSAILVision/semantic-segmentation-pytorch/issues/286) and follow the step here to download model checkpoints in the same way.

**1. ade20k-mobilenetv2dilated-c1_deepsup** 
```
wget http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-mobilenetv2dilated-c1_deepsup/encoder_epoch_20.pth 
wget http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-mobilenetv2dilated-c1_deepsup/decoder_epoch_20.pth
```

**2. ade20k-hrnetv2-c1** 
```
wget http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-hrnetv2-c1/encoder_epoch_30.pth 
wget http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-hrnetv2-c1/decoder_epoch_30.pth
```

**3. ade20k-resnet50dilated-ppm_deepsup** 
```
wget http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth 
wget http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth
```
