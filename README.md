## PlaTe: Visually-Grounded Planning with Transformers in Procedural Tasks

**[  📺 [Website](https://www.pair.toronto.edu/plate-planner/) | 🏗 [Github Repo](https://github.com/Jiankai-Sun/plate-pytorch) | 🎓 [Paper](https://arxiv.org/abs/2109.04869) ]**

## Requirements
- PyTorch >= 1.4
- [CrossTask](https://github.com/DmZhukov/CrossTask) Dataset

## Data Preparation
Follow this folder structure to prepare the dataset:
```
.
└── crosstask
    ├── crosstask_features
    └── crosstask_release
        ├── tasks_primary.txt
        ├── videos.csv
        └── videos_val.csv
```
The data root is set here [train_gpt.py](https://github.com/Jiankai-Sun/plate-pytorch/blob/main/train_gpt.py#L203).

## How to Run
```
conda create -f environment.yml
bash srun.sh
```
## Acknowledgement
We appreciate the following github repos a lot for their valuable code base implementations: [joaanna/something_else](https://github.com/joaanna/something_else), [karpathy/minGPT](https://github.com/karpathy/minGPT).

## Citation
```
@ARTICLE{PlaTe_RAL_2022,  
author={Sun, Jiankai and Huang, De-An and Lu, Bo and Liu, Yun-Hui and Zhou, Bolei and Garg, Animesh},  
journal={IEEE Robotics and Automation Letters},   
title={PlaTe: Visually-Grounded Planning With Transformers in Procedural Tasks},  
year={2022},  volume={7},  number={2},  pages={4924-4930},  
doi={10.1109/LRA.2022.3150855}}
```
