# TSformer-VO: an end-to-end Transformer-based model for monocular visual odometry

[![arXiv](https://img.shields.io/badge/cs.CV-arXiv%3A2305.06121-B31B1B.svg)](https://arxiv.org/abs/2305.06121)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/aofrancani/TSformer-VO/blob/main/LICENSE)

Code of the paper "[Transformer-based model for monocular visual odometry: a video understanding approach](https://arxiv.org/abs/2305.06121)"

<img src="tsformer-vo.jpg" width=600>


## Abstract
*Estimating the camera pose given images of a single camera is a traditional task in mobile robots and autonomous vehicles. This problem is called monocular visual odometry and it often relies on geometric approaches that require engineering effort for a specific scenario. Deep learning methods have shown to be generalizable after proper training and a considerable amount of available data. Transformer-based architectures have dominated the state-of-the-art in natural language processing and computer vision tasks, such as image and video understanding. In this work, we deal with the monocular visual odometry as a video understanding task to estimate the 6-DoF camera's pose. We contribute by presenting the TSformer-VO model based on spatio-temporal self-attention mechanisms to extract features from clips and estimate the motions in an end-to-end manner. Our approach achieved competitive state-of-the-art performance compared with geometry-based and deep learning-based methods on the KITTI visual odometry dataset, outperforming the DeepVO implementation highly accepted in the visual odometry community.*


## Contents
1. [Dataset](#1-dataset)
2. [Pre-trained models](#2-pre-trained-models)
3. [Setup](#3-setup)
4. [Usage](#4-usage)
5. [Evaluation](#5-evaluation)


## 1. Dataset
Download the [KITTI odometry dataset (grayscale).](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)

In this work, we use the `.jpg` format. You can convert the dataset to `.jpg` format with [png_to_jpg.py.](https://github.com/aofrancani/DPT-VO/blob/main/util/png_to_jpg.py)

Create a simbolic link (Windows) or a softlink (Linux) to the dataset in the `dataset` folder:

- On Windows:
```mklink /D <path_to_your_project>\TSformer-VO\data <path_to_your_downloaded_data>```
- On Linux: 
```ln -s <path_to_your_downloaded_data> <path_to_your_project>/TSformer-VO/data```

The data structure should be as follows:
```
|---TSformer-VO
    |---data
        |---sequences_jpg
            |---00
                |---image_0
                    |---000000.png
                    |---000001.png
                    |---...
                |---image_1
                    |...
                |---image_2
                    |---...
                |---image_3
                    |---...
            |---01
            |---...
		|---poses
			|---00.txt
			|---01.txt
			|---...
```

## 2. Pre-trained models



## 3. Setup
- Create a virtual environment using Anaconda and activate it:
```
conda create -n tsformer-vo python==3.8.0
conda activate tsformer-vo
```
- Install dependencies (with environment activated):
```
pip install -r requirements.txt
```

## 4. Usage



## 5. Evaluation
The evalutaion is done with the [KITTI odometry evaluation toolbox](https://github.com/Huangying-Zhan/kitti-odom-eval). Please go to the [evaluation repository](https://github.com/Huangying-Zhan/kitti-odom-eval) to see more details about the evaluation metrics and how to run the toolbox.


## Citation
Please cite our paper you find this research useful in your work:

```bibtex
@article{Francani2023,
  title={Transformer-based model for monocular visual odometry: a video understanding approach},
  author={Fran{\c{c}}ani, Andr{\'e} O and Maximo, Marcos ROA},
  journal={arXiv preprint arXiv:2305.06121},
  year={2023}
}
```

## References
Code adapted from [TimeSformer](https://github.com/facebookresearch/TimeSformer). Check out our previous work on monocular visual odometry: [DPT-VO](https://github.com/aofrancani/DPT-VO)

 
