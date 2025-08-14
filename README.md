# Human3DSeg

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2507.18655)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://your-project-page.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Human3DSEG: Part Segmentation of Human Meshes via Multi-View Human Parsing**  
> [James Dickens](https://github.com/JamesMcCullochDickens), Kamyar Hamad  
> 2025

![THuman2 1_Release_301-600_0303](https://github.com/user-attachments/assets/1ebf4995-3112-4162-bb53-0b1cad397b0f)


## Abstract
Human3DSeg is an open-source framework for 3D human body part segmentation that leverages multi-view human parsing techniques. This repository provides a complete pipeline for automatic annotation and segmentation of 3D human point clouds and meshes. The framework consists of two main components: a data preprocessing module that handles automatic 3D annotation through 2D projections, and a model training pipeline that implements a Point Transformer architecture for accurate point cloud segmentation. Human3DSeg achieves considerable performance on human body part segmentation tasks while requiring minimal manual annotation effort. The framework is designed to be modular, allowing researchers and practitioners to adapt individual components for their specific 3D human analysis tasks.

<!--## Key Features

- Feature 1
- Feature 2  
- Feature 3-->

## Repository Structure

This repository contains components for automatic 3D human point cloud annotation and segmentation, organized into two main parts:

```
Human3DSeg/
├── Data_Processing/      # Data preparation and preprocessing scripts
├── Model/                # Model training, evaluation, and inference code
├── data/                 
├── data_fps.py           
├── utils/                # Utility functions and helper modules
├── README.md             # Project documentation


Prepcoessed data is available at:
https://drive.google.com/drive/folders/1JU8f-IQLJxt3Gsuu4AEq61xD8vanAfXF?usp=sharing
```

## Installation

```bash
# Clone the repository
git clone git@github.com:kamyarothmanhamad/Human3DSegmentation.git
cd Human3DSEG

# Create conda environment
conda create -n human3dseg python=3.9
conda activate human3dseg

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation (`Data_Preprocessing/` folder)
The data preparation component provides tools and scripts for automatic 3D human point cloud annotation. This component:
- Operates independently from the training pipeline
- Includes comprehensive data preparation utilities
- Handles automatic annotation of 3D human point clouds
- Can be used as a standalone module for data preprocessing

```
Data_Preprocessing/
├── src/                   # Data loading utilities and scripts
│   ├── PointTransformerV3
│   ├── PyOpenGL
│   ├── Sapiens
│   ├── m2fp
│   └── Yolov8
├── data_processing/       # Evaluation scripts and metrics  
├── __init__.py 
```
To use m2fp as 2d segmentor, clone the official implementation into the src/ directory.
See https://github.com/soeaver/M2FP.

the Sapiens torchscript files is available here:
https://huggingface.co/facebook/sapiens-pose-1b-torchscript

**Usage for data preparation:**
```bash
cd Data_Preprocessing
python Data_Processing/data_processing.py 
```
---

### Training Pipeline (`Model/` folder)
The training pipeline processes the prepared data from the data preparation stage. This component:
- Works with output from the data preparation module
- Implements the main segmentation model training
- Includes evaluation and inference capabilities

```
Model/
├── Dataloaders/      # Data loading utilities and scripts
├── Evaluation/       # Evaluation scripts and metrics
├── Training/         # Training scripts and model definitions
├── __init__.py    
├── run.py            # Main entry point for running training/evaluation
```

To use Point Transformer v1 as a backbone, clone the official implementation into the `Model/` directory.

You can do this by running:
```bash
cd Model
git clone https://github.com/POSTECH-CVLab/point-transformer.git .
```

**Training:**
```bash
python Model/run.py pointtransformer_cihp_seg.yml human_seg_3d_cihp_PT.yml
```
<!--**Evaluation:**

### Evaluation
```bash
python 
```


## Dataset

TODO: Information about the dataset used, how to obtain it, and preprocessing steps.

## Results

| Method | Metric 1 | Metric 2 | Metric 3 |
|--------|----------|----------|----------|
| Ours   | **X.XX** | **X.XX** | **X.XX** |
| Method A | X.XX   | X.XX     | X.XX     |
| Method B | X.XX   | X.XX     | X.XX     |-->

## Citation
Please cite the following works:
```bibtex
@article{yang2023humanparsing,
  title={Deep Learning Technique for Human Parsing: A Survey and Outlook},
  author={Lu Yang and Wenhe Jia and Shan Li and Qing Song},
  journal={arXiv preprint arXiv:2301.00394},
  year={2023}
}

@article{khirodkar2024sapiens,
  title={Sapiens: Foundation for Human Vision Models},
  author={Khirodkar, Rawal and Bagautdinov, Timur and Martinez, Julieta and Zhaoen, Su and James, Austin and Selednik, Peter and Anderson, Stuart and Saito, Shunsuke},
  journal={arXiv preprint arXiv:2408.12569},
  year={2024}
}

@inproceedings{zhao2021point,
  title={Point transformer},
  author={Zhao, Hengshuang and Jiang, Li and Jia, Jiaya and Torr, Philip HS and Koltun, Vladlen},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={16259--16268},
  year={2021}
}
```

## License
[MIT](LICENSE) © 2025 James Dickens, Kamyar O. Hamad

## Acknowledgments

We thank the [THUman2.0 dataset](https://github.com/ytrock/THuman2.0-Dataset) for providing the human scans used in this work; please cite the dataset if you use it.


