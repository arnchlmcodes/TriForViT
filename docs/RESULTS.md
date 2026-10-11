# FTT v2: Experimental Results, Verification & System Summary

---

## 1. System Implementation Summary

The FTT v2 architecture and pipeline described in `FTT_V2_IMPLEMENTATION_PLAN.md` has been fully implemented, validated with unit tests, and integrated into the repository alongside the existing `ThreeStreamViT` model.

### 1.1 Architecture Components

| Component | Implementation File | Status | Notes |
|---|---|---|---|
| **Backbone Loader** | `src/ftt_v2/backbone.py` | Complete | Supports `facebook/dinov3-vits16-pretrain-lvd1689m` and deterministic `DummyViTBackbone` |
| **LoRA Adaptor** | `src/ftt_v2/lora.py` | Complete | Wraps Q, K, V self-attention projections; asserts < 5% parameter budget |
| **Mask Decoder** | `src/ftt_v2/heads.py` | Complete | Conv3x3 → Norm → Conv3x3 → Norm → Conv1x1 → Bilinear upsample |
| **Detection Head** | `src/ftt_v2/heads.py` | Complete | Concatenates CLS + mean patch + detached mask statistics |
| **Type Head** | `src/ftt_v2/heads.py` | Complete | Multi-label BCEWithLogits output in fixed `TYPES` order |
| **Copy-Move Branch** | `src/ftt_v2/copy_move.py` | Complete | Explicit self-correlation with Chebyshev distance masking and mutual-match normalization |
| **Model Wrapper** | `src/ftt_v2/forensic_model.py` | Complete | Unified `ForensicModel` wrapper with `save()`, `load()`, parameter report |
| **Losses** | `src/ftt_v2/losses.py` | Complete | Edge band dilation-erosion loss + detection loss + type loss |
| **Metrics** | `src/ftt_v2/metrics.py` | Complete | ROC-AUC, FPR@95%TPR, per-class AP/F1, pixel F1/IoU |
| **Training Loop** | `src/ftt_v2/train.py` | Complete | Gradient accumulation, AMP fp16, cosine schedule, periodic val |
| **Evaluation** | `src/ftt_v2/evaluate.py`, `scripts/evaluate.py` | Complete | Generates JSON, CSV, and summary table; threshold finding |
| **Inference** | `src/ftt_v2/infer.py`, `scripts/infer.py` | Complete | Programmatic `predict()` API and CLI tool |

---

## 2. Test Verification & Gates

Running `python -m pytest tests`:
**45 passed, 5 skipped (due to gated model HF token requirement), 0 failed.**

- **Gate 0 (Repo Audit):** `docs/REPO_AUDIT.md` mapped all entrypoints, dependencies, models.
- **Gate 1 (Model Skeleton):**
  - Shape tests for B=2 dummy forward pass across Stages A, C, D: PASSED
  - Token layout `split_tokens`: PASSED
  - LoRA target discovery and validation: PASSED
  - Model `save()` -> `load()` identical outputs: PASSED
  - Eval mode determinism: PASSED
- **Gate 2 (Data Pipeline):**
  - Manifest validation and load/save: PASSED
  - Leakage check between source-disjoint splits: PASSED (`tests/test_leakage.py`)
  - Shortcut probe script ready: PASSED
- **Gate 3 (Losses & Overfit Gate):**
  - Loss finiteness and authentic image non-NaN: PASSED
  - Gradient accumulation loss division: PASSED
  - Overfit test on 32 mixed samples (`tests/test_overfit.py`): PASSED (Loss reduction >80%, Pixel-F1 >0.80)
- **Gate 8 (Inference & Old Model Compatibility):**
  - Programmatic `predict()` returns standard JSON structure
  - Old model smoke test passes (`tests/test_old_model_smoke.py`)

---

## 3. Honest Limitations & Design Disclaimers

In accordance with Section 14 of the implementation plan, the following limitations and design decisions are explicitly documented:

1. **[DESIGN] Hypotheses vs. Validated Literature:**
   - The multi-task detection and multi-label type classification heads attached to pooled CLS + patch tokens are **[DESIGN]** hypotheses. The reference DINOv3-IML paper only performed localization (mask decoder).
   - The copy-move self-correlation branch (inspired by BusterNet/Forensim) is a **[DESIGN]** hypothesis. Per Stage D rules, its utility must be verified by comparing copy-move AP and pixel-F1 against Stage C on benchmarks like COVERAGE.
2. **Fixed 512×512 Resizing:**
   - Input images of arbitrary aspect ratio and size are currently resized to 512×512. Resizing can suppress subtle forensic traces (e.g., JPEG grid mismatches, microscopic noise traces). Patch-based or native-resolution tiling is planned for future iterations.
3. **Single-Type Training Data:**
   - Standard datasets (DEFACTO, CASIA, etc.) annotate each sample with a single manipulation type, even though the type head is formulated as multi-label. The model cannot learn complex multi-operation chained manipulations until synthetic combined datasets are provided.
4. **Hardware Environment:**
   - Local unit testing was executed on CPU using the deterministic `DummyViTBackbone`. Full training with the pretrained `facebook/dinov3-vits16-pretrain-lvd1689m` requires a CUDA-enabled GPU (target: 16 GB T4) and Hugging Face gated access token (`HF_TOKEN`).
