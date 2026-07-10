# Active Domain Adaptation via Structural-Prior Warm-Start and Region Uncertainty for Cross-Modality Cardiac Image Segmentation

[![Paper](https://img.shields.io/badge/Paper-Link-blue)](https://doi.org/10.1016/j.bspc.2026.110775)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📢 News
- **[2026-06]** Our paper has been accepted! The complete training and testing code is now released.
# Overview

***
We propose an algorithm for cross-modality image segmentation named Active Domain Adaptation via Structural-Prior
Warm-Start and Region Uncertainty (WUADA). Our method first constructs a structural-prior during a warm-start phase to
stabilize the model's uncertainty estimation early in training. This facilitates a more accurate delineation of the
decision boundary and improves segmentation reliability. Building on this structural prior, a region-level uncertainty
strategy, which combines neighborhood smoothing with foreground gating, directs the annotation budget toward the most
informative foreground boundaries.

![Overall Framework.png](resources/Overall%20Framework.png)

# Usage

***

## 1. Prerequisites

- python 3.9.21
- torch 2.6.0+cu126
- torchvision 0.21.0+cu126

## 2. Step-by-step installation

    # Create conda environment
    conda create --name WUADA -y python=3.9
    conda activate WUADA
    
    # Installing the pip and dependencies for the fresh python
    conda install -y ipython pip

    # Installing required packages
    pip install -r requirements.txt

## 3. Data Preparation

- Download datasets [MS-CMESeg](https://zmiclab.github.io/zxh/0/mscmrseg19/data.html)
  and [MM-WHS](https://zmiclab.github.io/zxh/0/mmwhs/)
- The structure of the dataset should be as follows:
  ```
  ├── datasets/
  │   ├── MM-WHS/
  │   │   ├── ct/
  │   │   ├── mr/
  │   │   ├── generate_image_list.py
  │   │   └── generate_label_info.py
  │   └── MS-CMRSeg/
  │       ├── bSSFP/
  │       ├── LGE/
  │       ├── generate_image_list.py
  │       └── generate_label_info.py
  │
  └── originData/
      ├── MM-WHS/ 
      │   ├── ct_train/
      │   └── mr_train/
      ├── MS-CMRSeg/ 
      │   ├── bSSFP/
      │   └── LGE/
      ├── mm-whs.py
      └── ms_cmrseg.py
  ```
- The execution process of data preprocessing:

    1. **Organize Raw Data**

       Place the downloaded raw dataset into the `originData` directory.

    2. **Preprocess the MM-WHS Dataset**

       Run the `mm-whs.py` script to convert the original 3D `.nii.gz` images into 2D `.npy` slices. The key steps
       include:
        - **Resampling**: Resample all volumes to an isotropic voxel spacing of `1.0 mm`.
        - **ROI Cropping**: Crop a `256x256x256` region of interest (ROI) centered on the label's centroid.
        - **Slicing**: Convert 3D volumes into 2D slices along the coronal plane.
        - **Channel Stacking**: Stack three adjacent slices (`[t-1, t, t+1]`) to create a 3-channel input.
        - **Normalization**: Apply Z-score normalization to the foreground region.
        - **Splitting & Saving**: Save the processed slices as `.npy` files, splitting them into an 80%/20% train/test
          ratio.

    3. **Preprocess the MS-CMRSeg Dataset**

       Run the `ms_cmrseg.py` script for preprocessing.
        - **Note**: The process is identical to the MM-WHS pipeline, except the ROI size is set to `192x192x192`.

    4. **Generate Image List**

       Run the `datasets/generate_image_list.py` script to generate the list of image names required for training.

    5. **Calculate Class Weights**

       Run the `generate_label_info.py` script to calculate an importance score for each label to address class
       imbalance.

## 4. Illustration of a 4-fold Split

- MM-WHS CT<->MR (Split by case)

| fold | Train     | Val   | Test  |
|------|-----------|-------|-------|
| 1    | 5-16      | 1-4   | 17-20 |
| 2    | 1-4，9-16  | 5-8   | 17-20 |
| 3    | 1-8，13-16 | 9-12  | 17-20 |
| 4    | 1-12      | 13-16 | 17-20 |

- MS-CMRSeg bSSFP->LGE(Split by case)

| fold | Train      | Val   | Test |
|------|------------|-------|------|
| 1    | 16-45      | 6-15  | 1-5  |
| 2    | 6-15，26-45 | 16-25 | 1-5  |
| 3    | 6-25，36-45 | 26-35 | 1-5  |
| 4    | 6-35       | 36-45 | 1-5  |

## 5. Model Zoo

We will put our model checkpoints
here [[Google Drive](https://drive.google.com/drive/folders/1xE2yAw1KTx2-9CUIXYUCPPCvJrCnB8mV?usp=drive_link)] [[百度网盘](https://pan.baidu.com/s/1fpavQeGI0J6ncwrb3-A2tQ?pwd=1234)] (
提取码1234).

- CT->MR

| fold | budget       | mDSC (%) | mASSD (mm) | ckpt              |
|------|--------------|----------|------------|-------------------|
| 1    | 2 pixels     | 78.2     | 2.7        | ct2mr/pixel_2     |
| 2    | 5 pixels     | 81.0     | 2.1        | ct2mr/pixel_5     |
| 3    | 40 pixels    | 82.5     | 1.8        | ct2mr/pixel_40    |
| 1    | 0.03% region | 74.1     | 3.0        | ct2mr/region_0.03 |
| 2    | 1% region    | 85.3     | 1.6        | ct2mr/region_1    |
| 3    | 5% region    | 85.2     | 1.6        | ct2mr/region_5    |

- MR->CT

| fold | budget       | mDSC (%) | mASSD (mm) | ckpt              |
|------|--------------|----------|------------|-------------------|
| 1    | 2 pixels     | 89.0     | 1.4        | mr2ct/pixel_2     |
| 2    | 5 pixels     | 90.3     | 1.2        | mr2ct/pixel_5     |
| 3    | 40 pixels    | 91.9     | 1.1        | mr2ct/pixel_40    |
| 1    | 0.03% region | 89.5     | 1.5        | mr2ct/region_0.03 |
| 2    | 1% region    | 92.1     | 1.0        | mr2ct/region_1    |
| 3    | 5% region    | 92.7     | 0.9        | mr2ct/region_5    |

- bSSFP->LGE

| fold | budget       | mDSC (%) | mASSD (mm) | ckpt                  |
|------|--------------|----------|------------|-----------------------|
| 1    | 2 pixels     | 86.3     | 1.5        | bssfp2lge/pixel_2     |
| 2    | 5 pixels     | 87.0     | 1.5        | bssfp2lge/pixel_5     |
| 3    | 40 pixels    | 87.1     | 1.4        | bssfp2lge/pixel_40    |
| 1    | 0.05% region | 86.5     | 1.5        | bssfp2lge/region_0.05 |
| 2    | 1% region    | 87.8     | 1.3        | bssfp2lge/region_1    |
| 3    | 5% region    | 87.5     | 1.4        | bssfp2lge/region_5    |

- Source-free

| fold | budget       | mDSC (%) | mASSD (mm) | ckpt                    |
|------|--------------|----------|------------|-------------------------|
| 1    | 2 pixels     | 80.0     | 2.7        | source-free/pixel_2     |
| 2    | 5 pixels     | 82.4     | 2.1        | source-free/pixel_5     |
| 3    | 40 pixels    | 84.1     | 2.2        | source-free/pixel_40    |
| 1    | 0.03% region | 80.7     | 3.1        | source_free/region_0.03 |
| 2    | 1% region    | 88.0     | 1.6        | source_free/region_1    |
| 3    | 5% region    | 88.1     | 1.5        | source_free/region_5    |

## 6. Train and test

- The 'scripts. sh' under 'scripts' stores the scripts for running all the models in the appeal.

# Acknowledgements

***
This project is based on the open-source project: [RIPU](https://github.com/BIT-DA/RIPU). We thank their authors for
making the source  code publically available.