# Robust Multi-Task Framework for Image Forgery Detection, Localization, and Classification (FTT v2)

> **Team 39** — Final Year IAPR Research Project  
> Dheeraj KB · Nehan GRM · Harish Veylan P · Balaji AR  
> Guide: Sruthi CJ — Dept. of Computer Science & Engineering

## Overview

**FTT v2 (Forensic Transformer v2)** is an end-to-end multi-task image forensics framework. It leverages a self-supervised **DINOv3 ViT-S/16** foundation vision transformer adapted via **LoRA** (Low-Rank Adaptation) and incorporates a specialized **Self-Correlation Copy-Move Branch** alongside convolutional localization and classification heads.

### Multi-Task Capabilities
1. **Pixel-Level Localization (`mask_logits`)**: High-resolution $(512 \times 512)$ segmentation mask locating forged pixels with boundary-aware edge loss.
2. **Image-Level Tamper Detection (`det_logit`)**: Binary decision logit determining Authentic vs. Forged.
3. **Forgery Type Classification (`type_logits`)**: Multi-label classification across `copy_move`, `splicing`, and `inpainting`.

---

## Architecture

```
                             Input Image (B, 3, 512, 512)
                                          │
                                          ▼
                         Frozen DINOv3 ViT-S/16 Backbone
                       (LoRA on Q, K, V Attention Projections)
                                ├── Layer 4 Hidden States ──┐
                                │                           ▼
                                │                 Copy-Move Correlation Branch
                                │                 • Patch Affinity Matrix (Chebyshev Masked)
                                │                 • Mutual Match Normalization
                                │                 • Top-8 Descriptors + Conv Head
                                │                           │
                                ▼                           ▼
                      Last Hidden State (Layer 12)     cm_feat (B, 32, 32, 32)
                                │                           │
             ┌──────────────────┼───────────────────────────┘
             │                  ▼
             │         Concatenate [F_last, cm_feat]
             │                  │
             │                  ▼
             │            Mask Decoder
             │         (Conv3x3 + Norm + ReLU -> Bilinear 16x)
             │                  │
             │                  ▼
             │         mask_logits (B, 1, 512, 512)
             │
             ├──> Detection Head (MLP on CLS + Patch Mean + Detached Mask/CM Stats) ──> det_logit (B,)
             │
             └──> Type Head (MLP on CLS + Patch Mean + Detached CM Stats) ──────────> type_logits (B, 3)
```

---

## Project Structure

```
.
├── configs/                         # Model and training YAML configurations
│   ├── ftt_v2_main.yaml             # Production run config (DINOv3 ViT-S/16 + all heads)
│   ├── ftt_v2_pilot_stageA.yaml     # Stage A: Decoder-only baseline
│   ├── ftt_v2_pilot_stageB.yaml     # Stage B: LoRA + Decoder
│   ├── ftt_v2_pilot_stageC.yaml     # Stage C: Multi-task (LoRA + Det + Type + Decoder)
│   └── ftt_v2_pilot_stageD.yaml     # Stage D: Full multi-task + Copy-Move branch
├── docs/                            # Research reports, review slides, design decisions
│   ├── DECISIONS.md                 # Architecture and design rationale
│   ├── OPEN_QUESTIONS.md            # Tracked questions and future research directions
│   ├── PROGRESS.md                  # Implementation phase audit and progress log
│   ├── REPO_AUDIT.md                # System audit and hardware specifications
│   ├── RESULTS.md                   # Evaluation results and metric benchmarks
│   ├── Team_39_Phase-2_Review-1_PPT.pdf
│   └── Team_39_Phase-2_Review-1_Report.pdf
├── scripts/                         # Manifest generation, inspection, and CLI tools
│   ├── build_manifest_casia.py      # Build dataset manifest for CASIA
│   ├── build_manifest_columbia.py   # Build dataset manifest for Columbia
│   ├── build_manifest_coverage.py   # Build dataset manifest for COVERAGE
│   ├── build_manifest_defacto.py    # Build dataset manifest for Defacto
│   ├── check_leakage.py             # Test for train/val data overlap via perceptual hash
│   ├── evaluate.py                  # CLI evaluation entrypoint
│   ├── infer.py                     # Single-image inference entrypoint
│   ├── make_inpaint_testset.py      # Synthesize test cases for inpainting
│   ├── shortcut_probe.py            # Diagnostic tool for dataset shortcuts
│   └── visualize_batch.py           # Batch visual inspection tool
├── src/
│   └── ftt_v2/                      # Core FTT v2 architecture package
│       ├── backbone.py              # DINOv3 ViT and CPU dummy backbone loader
│       ├── copy_move.py             # Explicit self-correlation copy-move branch
│       ├── forensic_model.py        # Unified ForensicModel nn.Module
│       ├── heads.py                 # Mask decoder, detection head, type head
│       ├── lora.py                  # PEFT LoRA adapter injection
│       ├── losses.py                # Boundary-aware edge loss + multi-task objective
│       ├── metrics.py               # Pixel F1, IoU, image AUC, mAP evaluation metrics
│       ├── outputs.py               # ModelOutput container dataclass
│       ├── train.py                 # Multi-task training engine
│       ├── evaluate.py              # Evaluation script
│       ├── infer.py                 # Inference and decision thresholding logic
│       ├── data/                    # Dataset, transforms, manifest parser, split engine
│       └── utils/                   # Checkpointing, seed utilities
├── tests/                           # Unit and integration test suite
├── requirements.txt                 # Project dependencies
└── FTT_V2_IMPLEMENTATION_PLAN.md    # Full specification and architecture design document
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Running Tests

All unit tests run on CPU without requiring GPU or gated Hugging Face model access:

```bash
python -m pytest tests -v
```

---

## Training

### Progressive Ablation Runs (Pilot Stages)
```bash
# Stage A: Mask decoder only baseline (frozen backbone)
python -m src.ftt_v2.train --config configs/ftt_v2_pilot_stageA.yaml

# Stage B: LoRA + Mask decoder
python -m src.ftt_v2.train --config configs/ftt_v2_pilot_stageB.yaml

# Stage C: Multi-task (LoRA + Detection head + Type head + Mask decoder)
python -m src.ftt_v2.train --config configs/ftt_v2_pilot_stageC.yaml

# Stage D: Full model (LoRA + All heads + Copy-Move branch)
python -m src.ftt_v2.train --config configs/ftt_v2_pilot_stageD.yaml
```

### Full Production Run
```bash
python -m src.ftt_v2.train --config configs/ftt_v2_main.yaml
```

---

## Evaluation & Inference

### Benchmark on a Manifest
```bash
python scripts/evaluate.py \
  --model-dir outputs/ftt_v2/model \
  --manifest data/manifests/val.csv \
  --output-dir outputs/ftt_v2/eval
```

### Predict on an Image
```bash
python scripts/infer.py \
  --model-dir outputs/ftt_v2/model \
  --image path/to/sample.jpg \
  --output outputs/ftt_v2/preds
```
