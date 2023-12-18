# pdm-f23-hw3
NYCU Perception and Decision Making 2023 Fall

Spec: [Google Docs](https://docs.google.com/document/d/10vEbFE372HeNocKmyQws_-5Dff27dhH60boNyF-QqMk/edit?usp=sharing)

## Preparation

1. Put dataset `replica_v1` under `src/`.
2. Download the `semantic_3d_pointcloud/` from the link in the spec, and put it under `src/`.
3. Install the dependencies `tqdm, scipy`. Other dependencies should be the same as HW2.
4. Run `python main.py --goal_name <goal_name> [--record <0,1>]` to construct map, generate path, navigate in Habitat.

## Usage

I split the process into 3 parts: construct map, generate path, navigate in Habitat. The `main.py` is just a wrapper to run these 3 parts in order.  You can also run them separately.

```shell
cd src/
# mapping
python construct_map.py --npy_path semantic_3d_pointcloud/point.npy --clr_path semantic_3d_pointcloud/color01.npy
# path generation
python generate_path.py --goal-name <goal_name>
# navigation
python navigate.py --path-type <path_type, rrt or smooth> --record <0, 1>
```

## Recording

The recording file will be saved in `src/record/`, both image and video.
