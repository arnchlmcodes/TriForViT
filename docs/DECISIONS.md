# Design Decisions Log

This document records design decisions and conflicts between the implementation plan and the repo conventions.

---

## D1: File Layout

**Plan says:** Create `ftt_v2/` top-level module.
**Repo convention:** All code lives in `src/`.
**Decision:** Place all FTT v2 code under `src/ftt_v2/` as a Python package, preserving the repo's `src/` convention. Existing files remain untouched.

## D2: Config Format

**Plan says:** YAML config files in `configs/`.
**Decision:** Use YAML config files in `configs/` directory at repo root. This is additive and doesn't conflict with existing code.

## D3: LoRA Target Modules

**Status:** PENDING — will be resolved in Phase 1 after inspecting DINOv3 model module names.

## D4: Edge Loss Normalization

**Plan says:** Check official DINOv3-IML repo for exact formula.
**Decision:** PENDING — will use the plan's formula (`sum(band * bce_map) / max(sum(band), 1)`) and verify if official repo is reachable.

## D5: Norm in Decoder

**Plan says:** BatchNorm default, GroupNorm when micro-batch < 8.
**Decision:** Default to GroupNorm (8 groups) since development is on CPU with small batches. Config-switchable.
