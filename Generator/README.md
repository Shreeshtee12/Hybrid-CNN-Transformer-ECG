# Hybrid CNN–Transformer ECG Classification
This repository contains the complete implementation of Hybrid CNN–Transformer Models for Multi-Label ECG Classification with Interpretability, a comprehensive investigation into deep learning architectures for automated electrocardiogram (ECG) interpretation, tested on the [PTB-XL ECG dataset](https://physionet.org/content/ptb-xl/1.0.1/). 

**Hybrid CNN–Transformer Models for Multi-Label ECG Classification with Interpretability**

## Overview
The project focuses on multi-label classification of ECG signals using deep learning models, including:
- CNN-based architectures
- Transformer-based architectures
- Hybrid CNN–Transformer models

## Research Goal
This work addresses automated cardiac diagnosis by developing novel hybrid architectures by achieving accurate classification while maintaining interpretability of model predictions.

Combine CNNs and Transformers synergistically to leverage both local morphological features and global temporal dependencies
Implement comprehensive interpretability mechanisms including attention visualization, Grad-CAM, SHAP, and Layer-Wise Relevance Propagation
Achieve state-of-the-art diagnostic performance across multiple cardiac abnormality categories
Provide clinically-actionable insights through explainable AI

## Cilincal Significance
Cardiovascular disease remains the leading cause of mortality worldwide. Automated ECG interpretation systems have the potential to:
-Reduce diagnostic delays in emergency and primary care settings
-Support clinical decision-making through AI-assisted pattern recognition
-Enable large-scale screening in resource-limited environments
-Provide continuous monitoring via wearable and portable devices


## Dataset
Experiments are conducted using the **PTB-XL ECG dataset**.
Due to licensing restrictions, the dataset itself is **not included** in this repository.

## Repository Structure
- `src/` – Model architectures and components
- `scripts/` – Training and inference scripts
- `notebooks/` – Experiment and analysis notebooks
- `requirements.txt` – Python dependencies

## Methods
- Multi-label ECG classification
- PyTorch-based training pipeline
- Evaluation using standard multi-label metrics
- Interpretability analysis of model predictions

## Status
This repository is part of an ongoing academic thesis project.
