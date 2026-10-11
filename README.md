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

---

## FTT v2: DINOv3 + LoRA Multi-Task Model

In addition to the baseline `ThreeStreamViT`, this repository implements **FTT v2** per `FTT_V2_IMPLEMENTATION_PLAN.md`. FTT v2 adds pixel-level localization (segmentation mask), multi-label forgery type classification, and forgery detection on top of a frozen DINOv3 ViT-S/16 backbone adapted via LoRA.

### Model Features
- **Backbone**: Frozen `facebook/dinov3-vits16-pretrain-lvd1689m` (or CPU `dummy` backbone for offline unit tests).
- **LoRA Adaptor**: Parameter-efficient fine-tuning on Q, K, V self-attention projections (<5% of backbone params).
- **Mask Decoder**: Conv decoder with edge-band dilation-erosion loss (`L_edge`).
- **Multi-Task Heads**: Image-level detection (`det_logit`) and multi-label type classification (`type_logits` across copy-move, splicing, inpainting).
- **Copy-Move Branch**: Explicit patch self-correlation with Chebyshev distance masking and mutual-match normalization.

### Running Tests
All tests run on CPU without requiring GPU or gated HF models:
```bash
python -m pytest tests -v
```

### Training
```bash
# Stage A (Mask decoder only baseline)
python -m src.ftt_v2.train --config configs/ftt_v2_pilot_stageA.yaml

# Stage B (LoRA + Mask decoder)
python -m src.ftt_v2.train --config configs/ftt_v2_pilot_stageB.yaml

# Stage C (LoRA + Detection head + Type head)
python -m src.ftt_v2.train --config configs/ftt_v2_pilot_stageC.yaml

# Stage D (LoRA + All heads + Copy-Move branch)
python -m src.ftt_v2.train --config configs/ftt_v2_pilot_stageD.yaml

# Main full run
python -m src.ftt_v2.train --config configs/ftt_v2_main.yaml
```

### Evaluation & Inference
```bash
# Evaluate on any manifest
python scripts/evaluate.py --model-dir outputs/ftt_v2/model --manifest data/manifests/val.csv --output-dir outputs/ftt_v2/eval

# Predict on an image
python scripts/infer.py --model-dir outputs/ftt_v2/model --image path/to/sample.jpg --output outputs/ftt_v2/preds
```

