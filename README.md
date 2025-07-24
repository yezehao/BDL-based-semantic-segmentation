# BDL based Sematic Segmentation
![GitHub License](https://img.shields.io/github/license/yezehao/BDL-based-semantic-segmentation)
![GitHub last commit](https://img.shields.io/github/last-commit/yezehao/BDL-based-semantic-segmentation)
![GitHub top language](https://img.shields.io/github/languages/top/yezehao/BDL-based-semantic-segmentation)

## Overview
This is a MSc individual project to enhance the visual detection capabilities of USVs by utilizing Bayesian SegNet. This project aims to enhance the reliability of USVs in complex and dynamic marine environments despite the scarcity of marine datasets.

## Achievement
Bayesian SegNet on [MaSTr1325 dataset](https://www.vicos.si/resources/mastr1325/)
![avater](https://github.com/yezehao/BDL-based-semantic-segmentation/blob/main/Deliverable/thesis/figures/MaSTr1325/BayesianSegNet-panel.png?raw=true)

Bayesian SegNet on [OASIs dataset](https://www.navlue.com/dataset)
![avater](https://github.com/yezehao/BDL-based-semantic-segmentation/blob/main/Deliverable/thesis/figures/OASIs/BayesianSegNet-usv-panel.png?raw=true)

## Gantt Chart
```mermaid
gantt
    dateFormat  YYYY-MM-DD
    title PROJ00114PG Bayesian deep learning based semantic segmentation for marine environments

    Mataining this Gantt Chart [Z. Ye] : crit, 2024-03-01, 2024-09-16

    section  Preliminary Preparation
    Literature Review: activate, 2024-03-01, 2024-07-01
    Outline Plan: done, crit, milestone, 2024-03-21, 0d
    Project Planning: done, 2024-03-21, 2024-04-30
    Project Initialization: done, crit, milestone, 2024-05-18, 0d 

    section Design & Development
    BNN on Simple Regression Problem: done, 2024-05-18, 2024-05-28
    WR - 20240528: done, milestone, 2024-05-28, 0d
    Relationship between BNN and Bayesian Inference: done, 2024-05-28, 2024-06-04
    WR - 20240604: done, milestone, 2024-06-04, 0d
    Bayesian CNN on simple image segmentation: activate, 2024-06-04, 14d
    Term 3 Progress Report: done, crit, milestone, 2024-06-10, 0d
    WR - 20240611: milestone, 2024-06-11, 0d
    WR - 20240618: milestone, 2024-06-18, 0d
    Bayesian CNN on MaSTr1325: 2024-06-18, 21d
    WR - 20240625: milestone, 2024-06-25, 0d
    WR - 20240702: milestone, 2024-07-02, 0d
    Term 4 Progress report: crit, milestone, 2024-07-08, 0d
    WR - 20240709: milestone, 2024-07-09, 0d


    section Testing
    Overall Testing and Parameter Adjustment: 2024-07-09, 2024-08-20
    Testing on Prediction Uncertainty: TPU, 2024-07-09, 15d
    WR - 20240716: milestone, 2024-07-16, 0d
    WR - 20240723: milestone, 2024-07-23, 0d
    Testing on Dataset Noise: TDN, after TPU, 15d
    WR - 20240730: milestone, 2024-07-30, 0d
    WR - 20240806: milestone, 2024-08-06, 0d
    Testing on Proir: TP, after TDN, 15d
    WR - 20240813: milestone, 2024-08-13, 0d
    WR - 20240820: milestone, 2024-08-20, 0d
    Testing Complement: crit, milestone, 2024-08-20, 0d

    section Thesis Writing
    Thesis Writing: 2024-08-09, 2024-09-02
    WR - 20240827: milestone, 2024-08-27, 0d
    Thesis Submission: crit, milestone, 2024-09-02, 0d
    Oral Examination Preparation : 2024-09-02, 2024-09-16
    WR - 20240903: milestone, 2024-09-03, 0d
    Oral Presentation Material Submission: crit, milestone, 2024-09-06, 0d
    Oral Examination: crit, milestone, 2024-09-16, 0d

```

## Dataset
The instructions of downloading datasets are illistruted in the [link](https://github.com/yezehao/BDL-based-semantic-segmentation/blob/main/Dataset/README.md).    
The corresponding `*.tar.gz` files of datasets are provided as well in [Dataset file](https://github.com/yezehao/BDL-based-semantic-segmentation/tree/main/Dataset).     

## Requirements
Conda environment ![conda](https://img.shields.io/badge/Anaconda-44A833.svg?style=flat&logo=Anaconda&logoColor=white)
```
conda create -n <your-env-name> python=3.10  
```
```
conda activate <your-env-name>
```
```
pip install -r requirements.txt
```

## Training & Testing
[None-Bayes-SegNet](https://github.com/yezehao/BDL-based-semantic-segmentation/tree/main/None-Bayes-SegNet)
```
python main.py --epoch 200 --arch segnet --batch_size 4 --dataset MaSTr1325 --action train&test
```

[Bayes-SegNet](https://github.com/yezehao/BDL-based-semantic-segmentation/tree/main/Bayes-SegNet)
```
python train.py --data-path MaSTr1325 --epoch 1000
```

## Performances
Performance on MaSTr1325 dataset
| **Architecture**   | **Pr (%)** | **Re (%)** | **F1 (%)** |
|--------------------|------------|------------|------------|
| Bayesian SegNet    | 81.2       | 97.8       | 87.8       |
| SegNet             | 79.9       | 87.5       | 81.3       |
| PSPNet             | 82.1       | 50.8       | 62.8       |
| U-Net              | 10.2       | 88.6       | 18.3       |

Performance on OASIs dataset
| **Architecture**    | **Evaluation Dataset** | **Type** | **Pr (%)** | **Re (%)** | **F1 (%)** |
|---------------------|------------------------|----------|------------|------------|------------|
| **Bayesian SegNet** | **OASIs**              | **1**    | 68.73      | 84.31      | 72.67      |
|                     |                        | **2**    | 47.27      | 95.74      | 64.36      |
|                     |                        | **3**    | 65.20      | 88.89      | 73.52      |
| **SegNet**          | **OASIs**              | **1**    | 28.27      | 65.89      | 33.93      |
|                     |                        | **2**    | 1.69       | 97.17      | 3.32       |
|                     |                        | **3**    | 10.09      | 98.97      | 17.82      |
| **SegNet**          | **SMD**                | --       | 31.2       | 76.3       | 44.3       |

## Citation
```
@article{ye2025bayesian,
  title={Bayesian deep learning based semantic segmentation for unmanned surface vehicles in uncertain marine environments},
  author={Ye, Zehao and Huang, Yanhong and Wu, Peng and Liu, Yuanchang},
  journal={Ocean Engineering},
  volume={339},
  pages={122065},
  year={2025},
  publisher={Elsevier}
}
```
