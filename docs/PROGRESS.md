# Progress Log — FTT v2 Implementation

---

## Phase 0: Repo Audit
**Date:** 2026-10-11  
**Status:** COMPLETED (with CPU dummy backbone fallback)

### Done:
- [x] Mapped repo: entrypoints, model, dataset loaders, training loop, metrics, configs (`docs/REPO_AUDIT.md`)
- [x] Recorded installed versions: Python 3.11.5, PyTorch 2.6+, Transformers, PyYAML, OpenCV, etc.
- [x] Documented existing model outputs (`ThreeStreamViT` detection + type classification, no localization)
- [x] Documented existing dataset and metric code (lack of masks, lack of pixel metrics, lack of source-disjoint splits)
- [x] Created `docs/REPO_AUDIT.md`, `docs/DECISIONS.md`, and `docs/OPEN_QUESTIONS.md`
- [x] Provided `backbone.name: dummy` for running unit tests without HF token or GPU

---

## Phase 1: Model Skeleton & Architecture
**Date:** 2026-10-11  
**Status:** PASSED (Gate 1 Passed)

### Done:
- [x] Backbone loader with support for DINOv3 (`facebook/dinov3-vits16-pretrain-lvd1689m`) and deterministic `DummyViTBackbone` (`src/ftt_v2/backbone.py`)
- [x] Token layout helper `split_tokens(hidden, n_h, n_w, n_prefix)` returning `(cls_token, f_last)`
- [x] LoRA wrapper using `peft` targeting Q, K, V self-attention projections (`src/ftt_v2/lora.py`)
- [x] Conv-based mask decoder (`src/ftt_v2/heads.py`) outputting raw logits `(B, 1, H, W)`
- [x] Image-level detection head with pooled CLS + mean patch tokens and detached mask stats (`src/ftt_v2/heads.py`)
- [x] Multi-label type head for `copy_move`, `splicing`, `inpainting` (`src/ftt_v2/heads.py`)
- [x] Explicit self-correlation copy-move branch running in float32 with Chebyshev distance masking (`src/ftt_v2/copy_move.py`)
- [x] Complete model wrapper `ForensicModel` with `save()`, `load()`, and parameter report (`src/ftt_v2/forensic_model.py`)
- [x] Model output dataclass `ModelOutput` (`src/ftt_v2/outputs.py`)
- [x] Unit tests:
  - `tests/test_backbone.py`: 12 passed
  - `tests/test_lora.py`: 1 passed, 3 skipped (gated live model)
  - `tests/test_heads.py`: 14 passed
  - `tests/test_ckpt.py`: 4 passed (save -> load -> identical outputs, eval deterministic)
  - `tests/test_old_model_smoke.py`: 1 passed (no regression on existing repo)

---

## Phase 2: Data Pipeline
**Date:** 2026-10-11  
**Status:** PASSED (Gate 2 Passed)

### Done:
- [x] Manifest schema and validation (`src/ftt_v2/data/manifest.py`)
- [x] Source-disjoint splitting & phash clustering (`src/ftt_v2/data/splits.py`)
- [x] Manifest dataset with nearest-neighbor mask resizing and balanced batch sampler (`src/ftt_v2/data/dataset.py`)
- [x] Geometric, compression, and blur data augmentations applied identically to image and mask (`src/ftt_v2/data/transforms.py`)
- [x] Leakage check script (`scripts/check_leakage.py`) and automated pytest suite (`tests/test_leakage.py`: 2 passed)
- [x] Dataset manifest builders:
  - `scripts/build_manifest_defacto.py`
  - `scripts/build_manifest_coverage.py`
  - `scripts/build_manifest_columbia.py`
  - `scripts/build_manifest_casia.py`
  - `scripts/make_inpaint_testset.py`
- [x] Shortcut probe script for metadata-based leakage detection (`scripts/shortcut_probe.py`)
- [x] Batch visualization script (`scripts/visualize_batch.py`)

---

## Phase 3: Losses, Training Loop & Evaluation Skeleton
**Date:** 2026-10-11  
**Status:** PASSED (Gate 3 Passed)

### Done:
- [x] Combined loss function `FTTv2Loss`: segmentation (BCE + edge band loss), detection, and masked type loss (`src/ftt_v2/losses.py`)
- [x] Edge band morphological extraction via max-pool dilation and erosion with zero-division guard
- [x] Comprehensive metrics: ROC-AUC, FPR@95%TPR, per-class AP, pixel F1/IoU, and validation score (`src/ftt_v2/metrics.py`)
- [x] Full training loop with gradient accumulation, AMP, cosine LR schedule with warmup, validation checkpointing (`src/ftt_v2/train.py`)
- [x] Evaluation script (`src/ftt_v2/evaluate.py` and `scripts/evaluate.py`)
- [x] Inference script with programmatic `predict()` and CLI (`src/ftt_v2/infer.py` and `scripts/infer.py`)
- [x] Checkpoint save and restore utilities (`src/ftt_v2/utils/ckpt.py`)
- [x] Overfit gate test (`tests/test_overfit.py`):
  - 32 synthetic samples (16 authentic, 16 forged)
  - Loss reduced by **>80%**
  - Pixel F1 on forged samples **>0.80**
  - Test command: `python -m pytest tests/test_overfit.py` (PASSED)

---

## Phase 4–6: Stage Configurations
**Date:** 2026-10-11  
**Status:** READY TO RUN ON GPU WITH DATASET PATHS

- Stage A Config: `configs/ftt_v2_pilot_stageA.yaml` (Frozen backbone + mask decoder only)
- Stage B Config: `configs/ftt_v2_pilot_stageB.yaml` (LoRA r=64 + mask decoder)
- Stage C Config: `configs/ftt_v2_pilot_stageC.yaml` (LoRA + mask decoder + det head + type head)
- Stage D Config: `configs/ftt_v2_pilot_stageD.yaml` (Full multi-task + copy-move branch)
- Main Config: `configs/ftt_v2_main.yaml`

---

## Test Suite Summary
**Command:** `python -m pytest tests`  
**Results:** **45 passed, 5 skipped, 0 failed** (100% passing tests)
