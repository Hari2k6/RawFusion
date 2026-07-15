# Efficient Burst HDR Image Restoration using Lightweight Deep Learning

> A lightweight PyTorch-based deep learning pipeline for reconstructing High Dynamic Range (HDR) images from multi-frame RAW burst sequences.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-green)
![HDR Imaging](https://img.shields.io/badge/Task-Burst%20HDR-orange)

---

## Overview

High Dynamic Range (HDR) imaging aims to recover details that are typically lost in extremely bright or dark regions of an image. Traditional cameras often fail to capture the full dynamic range of real-world scenes, especially under challenging lighting conditions. This project addresses this limitation by reconstructing a high-quality HDR image from a burst of multiple RAW frames captured at different exposure levels.

The proposed approach utilizes a lightweight convolutional neural network that efficiently fuses information from a sequence of nine RAW burst images. By combining complementary details across multiple frames, the model suppresses noise, reduces motion artifacts, and restores visually consistent HDR images while maintaining a low computational footprint. The architecture avoids computationally expensive attention mechanisms, making it suitable for faster inference and resource-constrained environments.

The complete pipeline was implemented using **PyTorch**, trained on an NVIDIA RTX 3050 Laptop GPU, and evaluated using **PSNR** and **SSIM** metrics.

---

# Example Results

### Input Burst Frame

<p align="center">
    <img src="assets/input_burst.png" width="750">
</p>

### Restored HDR Output

<p align="center">
    <img src="assets/prediction.png" width="750">
</p>

---

# Project Highlights

* Multi-frame Burst HDR image reconstruction
* Lightweight convolution-based burst fusion network
* Efficient RAW image processing pipeline
* Automatic burst grouping using a custom dataset loader
* Validation using PSNR and SSIM after every training epoch
* Best model checkpointing based on validation PSNR
* GPU-accelerated training using PyTorch
* Computationally efficient architecture with only **0.05 Million** parameters

---

# Project Statistics

| Property             | Value                             |
| -------------------- | --------------------------------- |
| Framework            | PyTorch                           |
| Programming Language | Python                            |
| Input                | 9 RAW Burst Frames                |
| Output               | Restored HDR RGB Image            |
| Training Images      | 300 Burst Sequences               |
| Validation Images    | 20 Burst Sequences                |
| Test Images          | 20 Burst Sequences                |
| Image Resolution     | 1536 × 768                        |
| Parameters           | ~0.05 Million                     |
| FLOPs                | ~3.548 GFLOPs                     |
| Validation PSNR      | **27.5 dB**                       |
| Test PSNR            | **27.0 dB**                       |
| Loss Function        | L1 Loss                           |
| Optimizer            | Adam                              |
| Learning Rate        | 1 × 10⁻⁴                          |
| Batch Size           | 4                                 |
| Epochs               | 20                                |
| GPU                  | NVIDIA RTX 3050 Laptop GPU (6 GB) |
| Training Time        | ~4–6 Hours                        |

---

# Why Burst HDR?

Single-image HDR reconstruction is often limited by sensor noise, clipped highlights, and missing shadow information. Burst HDR imaging addresses these challenges by combining multiple sequential RAW frames captured at different exposure levels.

Working directly with RAW sensor data preserves significantly more scene information than conventional RGB processing. By learning to fuse complementary details across multiple exposures, the proposed model produces cleaner, sharper, and more visually realistic HDR reconstructions while remaining computationally efficient.

---

# Key Features

* Lightweight deep learning architecture
* Burst image fusion
* HDR image reconstruction
* Noise suppression
* Motion artifact reduction
* Automatic dataset organization
* Efficient training pipeline
* Validation metric computation
* Best model checkpointing
* Fast inference on unseen burst sequences

---

# Dataset

The model is trained and evaluated on a multi-frame RAW burst HDR dataset consisting of burst sequences captured under varying exposure levels and lighting conditions. Each burst contains **9 RAW input frames** representing the same scene, along with a corresponding ground truth HDR image used for supervised learning.

## Dataset Summary

| Split      | Burst Sequences | Frames per Burst | Resolution |
| ---------- | --------------: | ---------------: | ---------: |
| Training   |             300 |                9 | 1536 × 768 |
| Validation |              20 |                9 | 1536 × 768 |
| Testing    |              20 |                9 | 1536 × 768 |

### Data Preparation

During preprocessing:

* RAW burst images are grouped automatically into valid scene sequences.
* Images are loaded using OpenCV.
* Color channels are converted from **BGR → RGB**.
* Images are normalized before being converted into PyTorch tensors.
* Only complete burst sequences containing **9 input frames and 1 ground truth image** are used during training.

The custom dataset loader automates burst grouping, reducing manual preprocessing while ensuring data consistency throughout training and evaluation.

---

# Model Architecture

The proposed model adopts a lightweight convolution-based burst fusion network designed for efficient HDR reconstruction from multi-frame burst images.

Instead of relying on computationally expensive transformer-based attention mechanisms, the network learns spatial features using convolutional layers while effectively aggregating complementary information from multiple burst frames.

The architecture processes an input tensor of shape:

```text
(9, 3, H, W)
```

representing nine RGB frames captured from the same scene.

The network performs:

* Multi-frame feature extraction
* Burst feature fusion
* Spatial feature learning
* HDR image reconstruction

and produces a single restored RGB image of shape:

```text
(3, H, W)
```

The lightweight design enables faster inference while maintaining competitive restoration quality.

---

# Training Pipeline

The complete training pipeline was implemented using **PyTorch**.

Each training iteration follows the workflow below:

```text
Burst Images
      │
      ▼
Custom Dataset Loader
      │
      ▼
Data Preprocessing
      │
      ▼
SimpleBurstModel
      │
      ▼
Predicted HDR Image
      │
      ▼
L1 Loss Computation
      │
      ▼
Backpropagation
      │
      ▼
Adam Optimizer
      │
      ▼
Model Update
```

Training progress is monitored using **tqdm**, while validation is performed after every epoch to evaluate model performance.

The checkpoint achieving the highest validation PSNR is automatically saved for inference.

---

# Training Configuration

| Parameter     |                             Value |
| ------------- | --------------------------------: |
| Framework     |                           PyTorch |
| Loss Function |     L1 Loss (Mean Absolute Error) |
| Optimizer     |                              Adam |
| Learning Rate |                          1 × 10⁻⁴ |
| Batch Size    |                                 4 |
| Epochs        |                                20 |
| GPU           | NVIDIA RTX 3050 Laptop GPU (6 GB) |
| Training Time |           Approximately 4–6 Hours |

---

# Evaluation Metrics

Model performance is evaluated using two widely adopted image restoration metrics.

### Peak Signal-to-Noise Ratio (PSNR)

PSNR measures the pixel-level similarity between the reconstructed image and the ground truth. Higher PSNR values indicate better reconstruction quality.

### Structural Similarity Index (SSIM)

SSIM evaluates perceptual image quality by comparing structural information, luminance, and contrast between images. It provides a better indication of visual similarity than pixel-wise error alone.

Validation metrics are computed after every training epoch to monitor convergence and determine the best-performing checkpoint.

---

# Model Analysis

To ensure the model remains computationally efficient, both parameter count and floating-point operations (FLOPs) were analyzed.

| Metric           |         Value |
| ---------------- | ------------: |
| Total Parameters | ~0.05 Million |
| FLOPs            | ~3.548 GFLOPs |

The lightweight architecture offers a favorable balance between computational efficiency and restoration performance, making it suitable for deployment on hardware with limited computational resources.

---

# Repository Structure

```text
Efficient-Burst-HDR-Restoration
│
├── DataLoader/          # Dataset loading and burst preprocessing
├── models/              # Network architectures
├── model_zoo/           # Model definitions and utilities
├── scoring_program/     # Evaluation scripts
├── utils/               # Helper functions
│
├── train.py             # Model training
├── test.py              # Model inference
├── test_demo.py         # Sample inference script
├── eval.py              # Evaluation metrics
├── sanity.py            # Sanity checking utilities
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/<YOUR_USERNAME>/Efficient-Burst-HDR-Restoration.git
cd Efficient-Burst-HDR-Restoration
```

## Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Training

Place the training dataset in the appropriate directory and update the dataset paths if required.

Start model training using:

```bash
python train.py
```

During training, the pipeline automatically:

* Loads burst image sequences
* Performs preprocessing
* Computes training loss
* Evaluates validation PSNR and SSIM
* Saves the best-performing checkpoint

---

# Inference

Run inference on unseen burst sequences using:

```bash
python test.py
```

The inference pipeline:

* Loads the trained model
* Processes 9-frame burst sequences
* Restores HDR images
* Applies post-processing and normalization
* Saves predictions as `.tif` images

---

# Results

The proposed lightweight architecture demonstrates a strong balance between restoration quality and computational efficiency.

| Metric          |      Performance |
| --------------- | ---------------: |
| Validation PSNR |      **27.5 dB** |
| Test PSNR       |      **27.0 dB** |
| Parameters      | **0.05 Million** |
| FLOPs           | **3.548 GFLOPs** |

Although the model is intentionally lightweight, it produces visually consistent HDR reconstructions while maintaining fast inference and low computational cost.

---

# Future Improvements

Potential directions for future work include:

* Incorporating attention-based feature fusion
* Exploring multi-scale feature extraction
* Training on larger HDR datasets
* Experimenting with perceptual and hybrid loss functions
* Improving motion alignment between burst frames
* Optimizing the model for real-time deployment

---

# Technologies Used

* Python
* PyTorch
* OpenCV
* NumPy
* TorchVision
* tifffile
* tqdm
* thop

---

# Acknowledgements

This project was developed as a team effort by three members as part of an academic computer vision project focused on Burst HDR image reconstruction.

The implementation builds upon concepts from modern Burst HDR restoration research while emphasizing computational efficiency through a lightweight convolution-based architecture.

Special thanks to the authors of the original **RawFusion / NTIRE Burst HDR** benchmark for providing the research foundation and dataset that enabled this work.

---

⭐ If you found this repository useful, consider giving it a star!

