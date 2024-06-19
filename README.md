# Infant Posture Classification

Codes and experiments for the following paper: 

Xiaofei Huang, Shuangjun Liu, Michael Wan, Nihang Fu, David Li Pino, Bharath Modayur, Sarah Ostadabbas, "Appearance-Independent Pose-Based Posture Classification in Infants"

Contact: 

[Xiaofei Huang](xhuang@ece.neu.edu)

[Sarah Ostadabbas](ostadabbas@ece.neu.edu)

## Table of Contents
  * [Introduction](#introduction)
  * [what's New](#what's_new)
  * [Environment](#environment)
  * [Data preparation](#data-preparation)
  * [How to use](#how-to-use)
  * [Citation](#citation)
  * [Acknowledgments](#acknowledgments)
  * [License](#license)

## Introduction
Gross motor activities are one of the earliest observable signals of development in infants and automatic early screening for motor delays could improve infant
development in a wide spectrum of domains. With such applications in mind, we present a two-phase data efficient and privacy-preserving pose-based posture classification framework. Our pipeline first produces 2D or 3D poses using algorithms we previously developed, and then feeds those poses into a posture classification network, which predicts one of four infant posture classes.

## Environment
The code is developed using python 3.6 on Ubuntu 18.04.
1. Install pytorch = v1.7.0 with cuda 10.1 following [official instruction](https://pytorch.org/).

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
## Data Preparation
(1) 2D keypoints ground truth and 3D corrected keypoints comes from [SyRIP dataset](https://coe.northeastern.edu/Research/AClab/SyRIP/). 
(2) For 2D keypoints prediction, [FiDIP model](https://arxiv.org/abs/2010.06100) is applied to estimate infant 2D pose from image. 
(3) For 3D keypoints prediction, [HW-HuP-Infant mdoel for infant](https://arxiv.org/abs/2105.10996) is applied to estimate infant 3D pose and camera parameters from image.
(4) Due to MIMM is a private datset please contact Early Markers company to obtain.
(5) pretained 2D pose-based and 3D pose-based posture classification models on SyFRIP dataset are placed in `ckpts` folder.

## Training 2D pose-based model on SyRIP dataset
```
python train_kpts_syrip_4class.py \
    --test_pred /Root path of your saved 2D predicted pose data/SyRIP_2d_pred/keypoints_validate_infant_results_0.json \
    --train_anno /Root path of your saved 2D groundtruth pose data/SyRIP/annotations/train600/person_keypoints_train_infant.json \
    --test_anno /Root path of your saved 2D predicted pose data/SyRIP/annotations/validate100/person_keypoints_validate_infant.json \ 
    --dir ./outputs
```

## Training 3D pose-based model on SyRIP dataset
```
python train_kpts3d_syrip_4class.py \
     --train_kpt /Root path of your saved 3D groundtruth pose data/SyRIP/test100_train600_3d/train600/correct_3D_600.npy \     
     --train_imgname /Root path of your saved 3D groundtruth pose data/SyRIP/test100_train600_3d/train600/output_imgnames_600.npy \
     --val_kpt /Root path of your saved 3D predicted pose data/SyRIP/test100_train600_3d/validate100/output_pose_3D_100.npy \
     --val_imgname /Root path of your saved 3D predicted pose data/SyRIP/test100_train600_3d/validate100/output_imgnames_100.npy \
     --dir ./outputs
```

## To do
Here we only provide our 4-class (i.e. Supine, Prone, Sitting, and Standing) posture classification models. We are continue to expand to 5-class model, which includes Supine, Prone, Sitting, Standing, and All-Fours.

## Citation

If you use our code or models in your research, please cite with:
* FiDIP for 2D Infant Pose Estimation
```
@inproceedings{huang2021infant,
  title={Invariant Representation Learning for Infant Pose Estimation with Small Data},
  author={Huang, Xiaofei and Fu, Nihang and Liu, Shuangjun and Ostadabbas, Sarah},
  booktitle={IEEE International Conference on Automatic Face and Gesture Recognition (FG), 2021},
  month     = {December},
  year      = {2021}
}
```
* HW-HUP for 3D Infant Pose Estimation
```
@article{liu2021heuristic,
  title={Heuristic Weakly Supervised 3D Human Pose Estimation in Novel Contexts without Any 3D Pose Ground Truth},
  author={Liu, Shuangjun and Huang, Xiaofei and Fu, Nihang and Ostadabbas, Sarah},
  journal={arXiv preprint arXiv:2105.10996},
  year={2021}
}
```


## Acknowledgement
Thanks for the interactive 3D annotation tool to help us create 3D weakly groundtruth pose of SyRIP dataset.
* [Cascaded Deep Monocular 3D Human Pose Estimation With Evolutionary Training Data, Li, Shichao and Ke, Lei and Pratama, Kevin and Tai, Yu-Wing and Tang, Chi-Keung and Cheng, Kwang-Ting](https://github.com/Nicholasli1995/EvoSkeleton)


## License 
* This code is for non-commercial purpose only. 
* For further inquiry please contact: Augmented Cognition Lab at Northeastern University: http://www.northeastern.edu/ostadabbas/ 




