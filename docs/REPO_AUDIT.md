# Repo Audit — Phase 0

**Date:** 2026-10-11
**Branch:** `dkb/arch-fix`

---

## 1. Repository Map

| Component | File(s) | Notes |
|---|---|---|
| **Entrypoint (train)** | `src/train.py` | Argparse-based, stages 1+2 jointly, DataParallel + AMP support |
| **Entrypoint (eval)** | `src/evaluate.py` | Binary + type metrics, Grad-CAM |
| **Entrypoint (cross-eval)** | `src/cross_eval.py` | Cross-dataset generalization |
| **Model** | `src/model.py` | `ThreeStreamViT`: spatial+FFT+noise streams, cross-attn fusion, timm ViT-B encoder, detection (2-class) + type (3-class) heads |
| **Feature extractors** | `src/extractors.py` | FFT magnitude, SRM-style noise residual, patch extraction |
| **Data loader** | `src/data_loader.py` | `ForgeryDataset` (folder-per-class), `DefactoCopyMoveDataset`, train/val/test split, augmentation |
| **Dataset info** | `src/download_datasets.py` | Dataset URLs and references (CASIA v2, IMD2020, Columbia, COVERAGE) |
| **Docs** | `docs/` | Team PPT and report PDFs |
| **Notebooks** | `notebooks/` | Empty (`.gitkeep` only) |
| **Config** | `requirements.txt` | torch, torchvision, numpy, Pillow, scikit-learn, matplotlib, tqdm |

---

## 2. Installed Versions

| Package | Version |
|---|---|
| Python | 3.12.3 |
| torch | 2.9.1+cpu |
| transformers | 4.57.3 |
| numpy | 2.4.3 |
| Pillow | 11.3.0 |
| scikit-learn | 1.8.0 |
| opencv-python | 4.12.0.88 |
| pandas | 3.0.1 |
| PyYAML | 6.0.2 |
| peft | *(installing)* |
| timm | *(installing)* |
| imagehash | *(installing)* |
| torchvision | *(installing)* |

**GPU:** N/A on this local machine (CPU-only torch install). Target hardware per plan: 16 GB T4.

---

## 3. Existing Model Outputs

The existing `ThreeStreamViT` model outputs:

| Output | Shape | Description |
|---|---|---|
| `detection_logits` | `(B, 2)` | 2-class softmax (authentic vs tampered) |
| `classification_logits` | `(B, 3)` | 3-class softmax (copy_move, splicing, inpainting) |

**No localization / mask output exists.** The existing model has no pixel-level prediction capability.

The existing model uses:
- `timm` ViT-Base (224×224, patch 16, embed_dim 768)
- Three-stream tokenization (spatial, FFT, noise residual)
- Cross-attention fusion
- CLS-token-based heads (no mask decoder)

---

## 4. Existing Metrics Code

- `evaluate.py`: accuracy, classification_report, confusion_matrix (sklearn)
- No ROC-AUC, no pixel-level F1/IoU metrics
- Grad-CAM for visualization (targets transformer encoder last layer)

---

## 5. Existing Dataset Code

- `ForgeryDataset`: folder-per-class with label inference from folder names
- `DefactoCopyMoveDataset`: DEFACTO copy-move specific loader
- **No mask loading** — the existing datasets return `(image, binary_label, forgery_label)` only
- **No source-disjoint splitting** — uses random_split
- **No manifest-based pipeline**

---

## 6. FTT v2 Plan — What Must Be Added

Per the implementation plan, the new model (`ftt_v2`) requires:
1. DINOv3 ViT-S/16 backbone (frozen) + LoRA on Q/K/V
2. Conv mask decoder (pixel-level localization)
3. Detection head (image-level, sigmoid, not softmax)
4. Multi-label type head (3-class BCEWithLogits, not CrossEntropy)
5. Optional copy-move self-correlation branch
6. Manifest-based data pipeline with mask loading
7. Edge-aware segmentation loss
8. Comprehensive evaluation (pixel F1/IoU, ROC-AUC, robustness, cross-dataset)
9. Inference script with JSON output

---

## 7. Compatibility Notes

- The existing model is selected implicitly (it's the only one). FTT v2 will be a separate model path, selectable by config `model.name: ftt_v2`.
- Existing `src/model.py`, `src/train.py`, etc. will be **preserved** for backward compatibility. New FTT v2 code will go into `src/ftt_v2/` module hierarchy, with new entrypoints added alongside existing ones.
- A smoke test for the old model will be added per the plan.
