# Comparison of Watershed and K-Means on the Basis of SSIM for Detection of Brain Tumours from MR Images

## Overview

This repository contains the implementation associated with the research study:

**"Comparison of Watershed and K-means on basis of SSIM for detection of brain tumours from MR images."**

The objective of this work is to compare two classical image segmentation techniques, **Watershed Segmentation** and **K-Means Clustering**, for the segmentation of brain tumours from Magnetic Resonance Imaging (MRI) scans. The quality of segmentation is evaluated using the **Structural Similarity Index Measure (SSIM)** by comparing the segmented outputs with manually annotated ground-truth images.

---

## Abstract

Brain tumour detection is a challenging task in medical image analysis and plays a crucial role in clinical diagnosis and treatment planning. Image segmentation serves as an essential preprocessing step for isolating tumour regions from MRI scans.

In this study, Watershed Segmentation and K-Means Clustering techniques are applied to brain MRI images and evaluated using the Structural Similarity Index Measure (SSIM). The experimental results demonstrate that Watershed Segmentation achieves higher structural similarity with the ground truth image compared to K-Means Segmentation, indicating superior segmentation performance for the considered dataset.

---

## Motivation

Manual analysis of MRI scans is time-consuming and subject to human variability. Automated image segmentation techniques can assist radiologists by providing preliminary tumour localization and region extraction.

This study investigates:

* The effectiveness of Watershed Segmentation for tumour region extraction.
* The effectiveness of K-Means Clustering for image segmentation.
* Comparative performance evaluation using SSIM.
* Similarity between algorithm-generated segmentation and manually segmented ground truth images.

---

## Methodology

### 1. Watershed Segmentation

Watershed Segmentation treats an image as a topographical surface where pixel intensities represent elevations.

The implemented pipeline consists of:

1. Image preprocessing
2. Morphological operations
3. Distance transform computation
4. Marker generation
5. Watershed transformation
6. Segmented image generation

### 2. K-Means Segmentation

K-Means is an unsupervised clustering algorithm that partitions image pixels into K clusters based on feature similarity.

The implemented workflow includes:

1. Pixel extraction and reshaping
2. Cluster initialization
3. Iterative cluster assignment
4. Centroid updating
5. Segmented image generation

### 3. Performance Evaluation using SSIM

The Structural Similarity Index Measure (SSIM) is employed to assess image similarity.

SSIM evaluates:

* Luminance similarity
* Contrast similarity
* Structural similarity

Higher SSIM values indicate greater similarity between the segmented image and the ground truth image.

---

## Repository Structure

```text
.
├── Abstract
├── README.md
├── Watershed ssim.py
├── seg.py
└── wk.py
```

### Code Description

| File                | Description                                                        |
| ------------------- | ------------------------------------------------------------------ |
| `Watershed ssim.py` | Watershed segmentation and SSIM comparison with ground truth       |
| `seg.py`            | K-Means segmentation and SSIM comparison with ground truth         |
| `wk.py`             | SSIM comparison between Watershed and K-Means segmentation outputs |
| `Abstract`          | Paper abstract                                                     |
| `README.md`         | Repository documentation                                           |

---

## Experimental Setup

### Software Requirements

* Python
* OpenCV
* NumPy
* Scikit-Image
* Scikit-Learn

### Installation

```bash
pip install opencv-python numpy scikit-image scikit-learn
```

---

## Experimental Results

The following SSIM values were obtained during experimentation:

| Comparison                                     | SSIM Score |
| ---------------------------------------------- | ---------- |
| Ground Truth vs Watershed Segmentation         | **0.8762** |
| Ground Truth vs K-Means Segmentation           | **0.7964** |
| Watershed Segmentation vs K-Means Segmentation | **0.8548** |

### Analysis

The Watershed Segmentation technique achieved the highest SSIM score when compared with the ground truth image. This indicates that the Watershed approach preserved structural information more effectively and produced segmentation results closer to manual annotations than the K-Means approach.

---

## Key Findings

* Watershed Segmentation outperformed K-Means Segmentation in terms of SSIM.
* Watershed-generated tumour boundaries were more consistent with the ground truth image.
* SSIM proved to be an effective metric for evaluating segmentation quality.
* Classical segmentation approaches can provide useful preliminary tumour extraction from MRI images.

---

## Reproducing the Experiments

### Watershed Segmentation Evaluation

```bash
python "Watershed ssim.py"
```

### K-Means Segmentation Evaluation

```bash
python seg.py
```

### Watershed vs K-Means Comparison

```bash
python wk.py
```

---

## Publication

**Hermehar P. S. Bedi, Gulchetan Singh, Gehna Sachdeva**

*Comparison of Watershed and K-means on basis of SSIM for detection of brain tumours from MR images.*

Published in:

**Computational Methods in Science and Technology (Volume 2)**

---

## Citation

```bibtex
@incollection{bedi2024watershed,
  title={Comparison of Watershed and K-means on basis of SSIM for detection of brain tumours from MR images},
  author={Bedi, Hermehar P. S. and Singh, Gulchetan and Sachdeva, Gehna},
  booktitle={Computational Methods in Science and Technology},
  year={2024}
}
```

---

## Authors

**Hermehar P. S. Bedi**
Chandigarh Engineering College (CGC), Landran, Mohali, India

**Gulchetan Singh**
Chandigarh University, Mohali, India

**Gehna Sachdeva**
Chandigarh University, Mohali, India

---

## License

This repository is provided for academic, educational, and research purposes. Please cite the associated publication when using the code or methodology in your work.
