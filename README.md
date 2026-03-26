# A Robust Patch-Based Hierarchical Framework for Image Forgery Detection and Classification

> **Team 39** — Final Year IAPR Research Project  
> Dheeraj KB · Nehan GRM · Harish Veylan P · Balaji AR  
> Guide: Sruthi CJ — Dept. of Computer Science & Engineering

## Overview

This project proposes a **Three-Stream Tokenization Vision Transformer (ViT)** that jointly fuses spatial, frequency (FFT), and noise-residual (SRM-style) patch representations via cross-attention for robust image forgery detection and classification.

### Hierarchical Strategy
1. **Stage 1** — Binary Detection: Authentic vs. Tampered
2. **Stage 2** — Forgery-Type Classification: Copy-Move / Splicing / Inpainting

## Project Structure

```
.
├── docs/                        # PPT, Report, Literature
├── notebooks/                   # Jupyter notebooks for EDA / experiments
├── src/
│   ├── extractors.py            # FFT, SRM noise-residual, patch extraction
│   ├── model.py                 # Three-Stream ViT architecture
│   ├── data_loader.py           # Dataset & DataLoader
│   ├── train.py                 # Training loop
│   ├── evaluate.py              # Evaluation metrics & Grad-CAM
│   └── download_datasets.py     # Dataset URLs & references
├── results/
│   ├── figures/                 # Plots, attention maps
│   └── checkpoints/             # Model weights
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd "Final Year IAPR Project"

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt
```

## Datasets (Online)

Datasets are **not stored locally** — they are accessed via online links during training.

Run the dataset reference script to see all sources:

```bash
python src/download_datasets.py
```

| Dataset    | Year | Images | Forgery Types                  | Source                 |
|------------|------|--------|--------------------------------|------------------------|
| CASIA v2.0 | 2013 | ~12K   | Copy-Move, Splicing            | Kaggle                 |
| IMD2020    | 2020 | ~2010  | Inpainting, Removal, Mixed     | UTIA (manual)          |
| Columbia   | 2004 | 363    | Splicing                       | Columbia DVMM (form)   |
| COVERAGE   | 2018 | 200    | Copy-Move                      | GitHub / OneDrive      |

## Training

```bash
python src/train.py --data_dir <path-or-url> --epochs 50 --batch_size 32
```

## Evaluation

```bash
python src/evaluate.py --data_dir <path-or-url> --checkpoint results/checkpoints/model_final.pth
```

## References

- Fridrich et al. (2003) — Copy-move forgery detection
- Bayar & Stamm (2016) — Deep learning for manipulation detection
- Dosovitskiy et al. (2021) — Vision Transformer (ViT)
- Fridrich & Kodovsky (2012) — SRM filters for steganalysis
- Wu et al. (2019) — ManTra-Net
- Selvaraju et al. (2017) — Grad-CAM
