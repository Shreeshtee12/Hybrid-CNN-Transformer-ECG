# PTB-XL ECG Classification Models

This repository contains multiple deep learning models developed and tested on the [PTB-XL ECG dataset](https://physionet.org/content/ptb-xl/1.0.1/). These models include CNNs, Transformer architectures, ResNet variants, LSTMs, and hybrids, designed to classify ECG signals based on diagnostic codes.

## Notebook Descriptions

| Notebook                        | Description |
|----------------------------------|-------------|
| `PTBXL_CNN.ipynb`               | A 1D convolutional neural network for multi-label classification using raw ECG signals. |
| `PTBXL_CNN_WandB.ipynb`         | CNN model integrated with [Weights & Biases](https://wandb.ai/) for experiment tracking. |
| `PTBXL_NN.ipynb`                | A fully connected neural network used as a baseline model. |
| `Transformer_CNN.ipynb`         | A hybrid model that combines 1D CNN layers with a Transformer for enhanced sequence learning. |
| `Transformer_OSC2.ipynb`        | Full pipeline for Transformer model, compatible with the OSC environment. |
| `ResNet1D_OSC.ipynb`            | A ResNet-style architecture adapted for 1D ECG signals. |
| `xResNet1D.ipynb`               | A deeper, improved ResNet variant for ECG classification with additional residual connections. |
| `xLSTM.ipynb`                   | A multi-layer LSTM model with optional CNN feature extraction front-end for temporal modeling. |
| `MacroROC_Comparison.ipynb`     | Aggregates predictions across all models and visualizes macro-average ROC curves. |


## Modular Model Implementations

All models are modularized in the `models/` directory and loaded dynamically using a central `model_selector.py`. Each file defines a `build_model()` function for easy integration into training scripts.

| Model Name        | File                    | Architecture Description |
|-------------------|-------------------------|---------------------------|
| `resnet1d`        | `resnet1d.py`           | Standard ResNet18-style 1D CNN for ECG signals |
| `xresnet1d`       | `xresnet1d.py`          | Deeper ResNet variant with flexible block depths |
| `transformer`     | `transformer.py`        | Pure Transformer encoder for sequence modeling |
| `cnn_transformer` | `cnn_transformer.py`    | Hybrid model combining 1D CNN layers with Transformer |
| `xlstm`           | `xlstm.py`              | Hybrid CNN + Bidirectional LSTM for temporal modeling |


## Features

- Multi-label classification using **SCP diagnostic codes**
- CNNs, ResNets, Transformers, LSTMs, and hybrid models
- Stratified training/testing splits with oversampling (AFIB handling)
- Macro-average **ROC-AUC** and per-label **ROC curves**
- Compatibility with **Ohio Supercomputer Center (OSC)**
- Visual model comparison with macro-AUC overlays

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/NicholasKanos/ptbxl-ecg-models.git
   cd ptbxl-ecg-models
   ```

2. **Setup Python environment**
   ```bash
   conda create -n ptbxl-env python=3.10
   conda activate ptbxl-env
   pip install -r requirements.txt
   ```

3. **Launch Jupyter**
   ```bash
   jupyter lab
   ```

## Requirements

Install from `requirements.txt`:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch torchvision wfdb pandas numpy scikit-learn matplotlib tqdm
```

## Dataset

- **PTB-XL ECG dataset**  
  10-second, 12-lead ECG recordings
- Auto-downloaded & extracted via `TransformerOSC.ipynb`

## Evaluation Metrics

- Macro-average **ROC-AUC** per model
- Per-class ROC curves
- Classification report (Precision, Recall, F1)
- Unified comparison in `MacroROC_Comparison.ipynb`

## Research Focus

This repository supports master's thesis work in:
- ECG classification using deep learning
- Comparison of temporal and attention-based architectures
- Healthcare AI for early diagnosis and risk prediction

## License

Licensed under the **MIT License**.

## Author

**Nicholas Kanos**  
Master of Computing and Information Systems, Youngstown State University  
**Focus**: ECG classification • Deep learning • Transformers • Healthcare AI
