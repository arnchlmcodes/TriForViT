# FTT v2: Implementation Plan
## DINOv3 + LoRA multi-task image forgery model (detect, classify type, localize)

**Audience:** an AI coding agent (Claude Code, Antigravity, Cursor, etc.) working inside the user's EXISTING repo.
**Goal:** add a new model that
1. decides **forged vs authentic**,
2. classifies the **forgery type(s)** among `copy_move`, `splicing`, `inpainting`,
3. **localizes** forged pixels with a mask.

---

## 0. Rules for the agent (read first, obey always)

1. Read this whole document before writing code. Execute phases in order. Every phase ends with a **GATE**. Do not start the next phase until the gate passes. If a gate fails twice after fixes, STOP and report exactly what you observed.
2. Tags used below:
   - **[PAPER]** = taken from the DINOv3-IML paper (arXiv 2604.16083, "DINOv3 Beats Specialized Detectors").
   - **[DESIGN]** = our own decision, NOT validated by a paper. Treat as a hypothesis to test.
   - **[VERIFY]** = an assumption about an external library or model. You MUST check it against the installed version (print shapes and module names, read source/docs). Never guess.
3. Do not break existing functionality. The existing model and training code must keep working. The new model is selected by a config flag (`model.name: ftt_v2`). Prefer additive changes. Do not delete or rewrite existing files unless a phase says so.
4. No hard-coded absolute paths, tokens or dataset locations. Use config files and environment variables. Never commit secrets (the Hugging Face token goes in `HF_TOKEN`).
5. Commit per phase: `phaseN: <summary>`. Write pytest tests as you go.
6. Seed everything (python, numpy, torch). Log seed and git commit hash with every run.
7. If this plan conflicts with the repo's conventions: follow the repo for style and structure, follow this plan for behavior. Record the conflict in `docs/DECISIONS.md` and continue.
8. Never claim a test passed without running it. Record the command and a result summary in `docs/PROGRESS.md` after each phase.
9. If something is unknown, write it in `docs/OPEN_QUESTIONS.md` and ask the user. Do not invent numbers, file names or labels.
10. Tests must not require the gated model or a GPU. Provide `backbone.name: dummy` (a tiny frozen random transformer with the same output layout) so unit tests run on CPU.

---

## 1. Task definition

Input: RGB image, any size. Version 1 resizes to **512x512** bilinear [PAPER does the same].

Outputs per image:

| name | shape | meaning |
|---|---|---|
| `mask_logits` | (B,1,512,512) | per-pixel forged logit |
| `det_logit` | (B,) | image-level forged logit |
| `type_logits` | (B,3) | multi-label logits, order `TYPES = ["copy_move","splicing","inpainting"]` |

- The type order is fixed everywhere in the code base. Define `TYPES` once and import it.
- Authentic image: mask all zero, `is_forged=0`, types `[0,0,0]`.
- Inference decision: forged iff `sigmoid(det_logit) >= tau_det`. Types are reported only when forged, and each type is reported when `sigmoid >= tau_type[k]`. Thresholds come from the validation set.
- Non-goals: face-swap or deepfake detection, whole-image AI-generated detection, video, adversarial robustness.

---

## 2. Phase 0: Repo audit (mandatory, no behavior changes)

Tasks:
1. Map the repo: entrypoints, model definitions, dataset loaders, training loop, loss, metrics, configs, requirements. Write `docs/REPO_AUDIT.md` with file paths for each.
2. Record versions: python, torch, torchvision, timm, transformers, peft, opencv, any augmentation library. Record GPU via `torch.cuda.get_device_name(0)` and free memory. Target hardware is a 16 GB T4.
3. Record what the existing model already outputs (detection, type, localization) and the existing metrics code. Reuse existing dataset or metric code ONLY if it satisfies Section 5 (masks, authentic images, source-disjoint splits). Otherwise add new modules; do not edit old ones.
4. Add missing dependencies to the requirements file (`peft`, a `transformers` version that supports DINOv3 [VERIFY minimum version], `scikit-learn`, `pandas`, `imagehash`). Pin the versions that work.
5. Hugging Face access. The model `facebook/dinov3-vits16-pretrain-lvd1689m` is **license-gated**. Check that loading works. If it does not, STOP and tell the user: accept the license on the model page and provide a token via `HF_TOKEN`.

**GATE 0:** `docs/REPO_AUDIT.md` exists, and a short script loads the DINOv3 backbone and runs a `(1,3,512,512)` dummy forward pass on the GPU without error.

---

## 3. Architecture

Notation: `d` = backbone embedding dim, `N = 1024` patches (32x32 grid for 512x512 input), `F_last` = last-layer patch tokens reshaped to `(B,d,32,32)`.

```
image -> DINOv3 ViT-S/16 (frozen) + LoRA on Q,K,V in every attention layer
           |-- last layer patch tokens -> F_last -----------------> Mask decoder -> mask_logits
           |-- CLS + mean patch tokens (+ mask stats) ------------> Detection head -> det_logit
           |-- CLS + mean patch tokens (+ copy-move scalars) -----> Type head -> type_logits
           '-- early layer patch tokens -> Copy-move branch -----> extra decoder channels + scalars
```

### 3.1 Backbone [PAPER: frozen DINOv3 ViT, last-layer dense features]
- Model: `facebook/dinov3-vits16-pretrain-lvd1689m` (ViT-S/16). Expected `d=384`, patch 16, depth 12 [VERIFY from the model config: `hidden_size`, `num_hidden_layers`, `patch_size`, number of register tokens].
- Load via `transformers.AutoModel` [VERIFY]. If it fails, use the official DINOv3 repo loader.
- Nothing in the code may hard-code 384. Read `d` from the config so ViT-B (d=768) is a config change.
- Input normalization: read mean and std from the model's `AutoImageProcessor` config [VERIFY]. Do NOT hard-code numbers.
- **Token layout:** the output sequence contains `[CLS] + [register tokens] + patch tokens` [VERIFY order and count]. Patch tokens are the LAST `(H/16)*(W/16)` tokens (1024 for 512x512). Write a helper `split_tokens(hidden, n_h, n_w)` returning `(cls, patches)`. Unit test: patch count == 1024, and patch `(r,c)` sits at index `r*32+c` after reshape.
- Apply a final LayerNorm to patch tokens if `last_hidden_state` is not already normalized [VERIFY; the paper applies LayerNorm].
- Request intermediate states (`output_hidden_states=True`) for the copy-move branch [VERIFY: whether `hidden_states[0]` is the embedding output and `hidden_states[i]` the output of block `i`].
- All backbone weights start with `requires_grad=False`.

### 3.2 LoRA [PAPER]
- Use `peft.LoraConfig`. Apply LoRA to the **Q, K, V projections of EVERY self-attention layer** and nothing else (not MLP, not the output projection).
- Defaults for ViT-S: `r=64`, `lora_alpha=128` [PAPER: ViT-S scored higher with r=64 than r=32 in its tables]. Provide `r=32, alpha=64` as an alternative config. `lora_dropout`: the paper does not say, so default `0.0` [DESIGN].
- Find target module names by printing `[n for n,_ in backbone.named_modules()]`. The layout may be separate Q/K/V linear layers or one fused `qkv` linear [VERIFY]. Target whichever exists. Write the names you found in `docs/DECISIONS.md`.
- Add runtime assertions right after wrapping:
  1. number of LoRA-wrapped modules == `3*depth` (separate Q/K/V) or `depth` (fused);
  2. every trainable backbone parameter name contains `lora_`;
  3. trainable backbone params > 0 and < 5% of backbone params;
  4. with LoRA enabled and untouched (B matrices initialize to zero), backbone outputs equal the plain frozen model's outputs (`allclose`). This proves LoRA starts as a no-op.
- Support `lora.enabled: false` (decoder-only baseline). Then the backbone has zero trainable parameters.
- Keep trainable weights in fp32. Use `torch.autocast` for fp16 compute. Do not call `.half()` on the model.

### 3.3 Heads

#### 3.3.1 Mask decoder [PAPER]
`Conv3x3(d -> d/2) + Norm + ReLU`, then `Conv3x3(d/2 -> d/4) + Norm + ReLU`, then `Conv1x1(d/4 -> 1)`, then bilinear upsample to 512x512 (`align_corners=False`). Output **logits** (no sigmoid inside the module; losses use `BCEWithLogits`, which is safe under mixed precision).
- Norm: the paper uses BatchNorm. Config `decoder.norm: bn|gn`, default `bn`. If the micro-batch is below 8, switch to `gn` (GroupNorm, 8 groups) [DESIGN]. BN statistics are unreliable with tiny batches.
- If the copy-move branch is on, the first conv takes `d + 32` input channels.

#### 3.3.2 Detection head [DESIGN]
- Pooled vector `z = concat(cls_token, mean(patch_tokens))` (size `2d`).
- Scalars from the mask branch: `s_max = max(mask_logits_at_32x32)` and `s_topk = mean of the top 1% logits`. Both are **detached** so the detection loss cannot distort localization. Config `det.use_mask_stats` (default true).
- MLP: `Linear(in -> 256) -> GELU -> Dropout(0.1) -> Linear(256 -> 1)`. Output `det_logit` of shape `(B,)`.

#### 3.3.3 Type head [DESIGN]
- **Multi-label, NOT softmax.** Same pooled features as the detection head (plus copy-move scalars if that branch is on) -> MLP -> 3 logits in `TYPES` order.
- Loss `BCEWithLogits`. Authentic images have all-zero targets. Rationale: one image can contain several manipulations, and authentic needs "none of the above".
- Optional per-class `pos_weight` from training frequencies (config).

#### 3.3.4 Copy-move branch (explicit self-correlation) [DESIGN, inspired by BusterNet and Forensim, NOT validated in our setting]
Why: copy-move means "this region matches another region in the same image". A conv decoder on last-layer features sees only local context, and the paper's ViT-S results on copy-move (COVERAGE) were much weaker than ViT-L's.

Run this module entirely in **float32 with autocast disabled**:
1. Take patch tokens from an EARLY layer (`cm.tap_layer`, default 4 of 12 [DESIGN; Forensim reportedly used the first four layers]). Reshape to `(B,N,d)`.
2. Project to 128 dims (`Linear`), then L2-normalize each token -> `Fn (B,N,128)`.
3. Affinity `A = Fn @ Fn^T / tau`, shape `(B,N,N)`. `tau` is learnable, init 0.1.
4. Mask out self and nearby patches: precompute a constant boolean `(N,N)` matrix `near[i,j] = Chebyshev grid distance(i,j) <= cm.min_dist` (default 3 patches = 48 px). Set `A[near] = -1e4`. Use a finite value, never `-inf` (avoids NaN in softmax).
5. Mutual-match normalization: `P = sqrt( softmax(A, dim=-1) * softmax(A, dim=-2) )`.
6. Per-patch descriptor = top-k values of `P` along the last dim (`cm.topk`, default 8) -> `(B,N,k)` -> reshape `(B,k,32,32)`.
7. Small head: `Conv3x3(k -> 32) + ReLU -> Conv3x3(32 -> 32) + ReLU` gives `cm_feat (B,32,32,32)`; `Conv1x1(32 -> 1)` gives `cm_map_logits (B,1,32,32)`.
8. Use `cm_feat` as extra input channels for the mask decoder (concatenate with `F_last`). Use `max(cm_map_logits)` and the top-1% mean as extra scalars for the detection and type heads.

**CRITICAL:** do NOT pixel-supervise `cm_map_logits` with the ground-truth mask. The match map lights up BOTH the source region and the pasted region, but the GT mask usually marks only the pasted one [VERIFY per dataset]. Supervising it with the mask would punish correct matches. Let it learn only through the downstream decoder, detection and type losses.

Memory: `A` is `(B,1024,1024)`, fine at 512x512. If the input size grows, `N` grows quadratically. Assert `N <= 4096` or implement chunked top-k.

### 3.4 Model wrapper
- `ForensicModel.forward(images) -> ModelOutput` (a dataclass with `mask_logits, det_logit, type_logits, cm_map_logits (optional)`). Training and evaluation code must only use these field names.
- `model.trainable_parameters()` returns LoRA + heads only. `model.param_report()` prints counts for frozen backbone, LoRA and each head.
- `save(path)` stores LoRA adapter weights + head weights + the full config as JSON + git hash. `load(path)` rebuilds the model from that config. Never pickle the whole model.

---

## 4. Losses

`L_total = L_seg + lambda_det * L_det + lambda_type * L_type` (defaults `lambda_det = 1.0`, `lambda_type = 1.0` [DESIGN]).

- **`L_seg = L_bce + lambda_edge * L_edge`** [PAPER, `lambda_edge = 20`].
  - `L_bce`: `BCEWithLogits` over all pixels, mean. Computed for ALL images. Authentic images have all-zero masks, which teaches the decoder to stay quiet.
  - `L_edge`: weighted BCE with weight 1 inside a band around the GT mask boundary and 0 elsewhere [PAPER: band width 7 px]. Build the band with max-pool: `dil = maxpool(mask, k=7, s=1, p=3)`, `ero = -maxpool(-mask, k=7, s=1, p=3)`, `band = dil - ero`. Then `L_edge = sum(band * bce_map) / max(sum(band), 1)`. For authentic images the band is empty and the `max(...,1)` guard prevents a divide-by-zero NaN.
  - [VERIFY] The paper's exact normalization of the edge term is not fully specified. If the official repo (github.com/Irennnne/DINOv3-IML) is reachable, compare; otherwise keep the formula above and note it in `docs/DECISIONS.md`.
- **`L_det`** = `BCEWithLogits(det_logit, is_forged)`.
- **`L_type`** = `BCEWithLogits(type_logits, type_targets)`, masked by `type_valid` (rows whose type is unknown are excluded), mean over valid entries.
- **Gradient accumulation:** divide the loss by `accum_steps` before `backward`. Unit test: 4 micro-batches of 2 give the same gradients as 1 batch of 8 within tolerance (use eval mode and a model without BN or dropout for this test).
- Never apply `sigmoid` before a `*WithLogits` loss.

Required unit tests: edge band on a toy square mask has the expected area; an authentic (all-zero) mask gives finite loss and zero edge loss; all losses are finite on random inputs.

---

## 5. Data pipeline

### 5.1 Datasets
- **Train and validation:** DEFACTO (copy-move, splicing, inpainting, with masks; derived from MS-COCO) plus **authentic negatives from MS-COCO**. DEFACTO has no authentic images.
- **Test A (in-domain):** held-out DEFACTO + COCO, source-disjoint from training.
- **Test B (cross-dataset, real benchmarks):** COVERAGE, CoMoFoD, Columbia, IMD2020, NIST16, CASIA v2. **Never train on any Test B dataset.**
- **Test C (modern inpainting):** a small custom set (about 500 forged + 500 authentic) generated later with an open inpainting model. Test only.
- Sizes: pilot train = 2k per type + 3k authentic; main train = 10k per type + 15k authentic; validation = 10% of train (source-disjoint); Test A = 2k per type + 6k authentic.
- The user supplies dataset locations via config. This plan does NOT download datasets. Ask the user for paths at the start of Phase 2.

### 5.2 Manifest
One CSV per split at `data/manifests/{train,val,test_A,test_B_<name>,test_C}.csv` with columns:
`image_id, path, mask_path, is_forged, copy_move, splicing, inpainting, type_valid, source_id, dataset, width, height, file_format`

- Authentic: `mask_path` empty (the loader creates a zero mask), types `0,0,0`, `type_valid=1`.
- Forged: use exactly the labels the dataset provides. If the type is unknown, set `type_valid=0` so the row is excluded from `L_type` only.
- Write one `scripts/build_manifest_<dataset>.py` per dataset. Do not guess labels: Columbia = splicing only; COVERAGE and CoMoFoD = copy-move only; for CASIA v2 the filenames may encode same-source vs different-source (`Tp_S` / `Tp_D`) [VERIFY by inspecting files and the README before using]; NIST16 types come from its metadata [VERIFY].

### 5.3 Splitting (leakage guard)
- Split by `source_id`, never by file. All variants derived from one COCO image go to the same split. [VERIFY whether DEFACTO filenames encode the COCO image id.] If source ids cannot be recovered, group near-duplicates by perceptual hash (`imagehash.phash`, Hamming distance <= 6) [DESIGN] and split by group.
- Authentic COCO negatives must exclude every COCO image used as a DEFACTO source, in any split.
- `scripts/check_leakage.py`: asserts that the `source_id` intersection between train, val and test is empty, and prints counts per split by type. Run it as a pytest test.

### 5.4 Shortcut probe (mandatory)
Forged and authentic images come from different pipelines. A model can learn file format or compression instead of forgery.
- `scripts/shortcut_probe.py`: train logistic regression (or a small gradient-boosting model) on METADATA ONLY (width, height, aspect ratio, file bytes per pixel, file extension, mean JPEG quantization-table value via PIL `im.quantization`, EXIF presence) to predict `is_forged` on train. Report AUC on val.
- If AUC > 0.65 the data is biased. Fix it BEFORE training: re-encode authentic images through the same save step, format and quality as forged ones, and match the resolution distribution. Re-run until AUC <= 0.65. Log the AUC in `docs/PROGRESS.md`.

### 5.5 Dataset class and augmentation
- Returns a dict: `image` (3,512,512 float, normalized with the backbone processor's mean/std), `mask` (1,512,512 float in {0,1}), `is_forged`, `types` (3), `type_valid` (3), `source_id`, `dataset`.
- Resize image bilinear. Resize mask **nearest**, then binarize. Never interpolate masks bilinearly.
- Training augmentation (same geometric transform applied to image and mask): horizontal flip; random-resized-crop with scale 0.6 to 1.0 then resize to 512 (p=0.5); JPEG recompression with quality U(50,100) (p=0.3); Gaussian blur with kernel in {3,5,7} (p=0.2); Gaussian noise (p=0.2). No heavy color jitter. Validation and test: resize + normalize only.
- Apply every augmentation independently of the label. An augmentation applied to only one class creates a shortcut.
- Class-balanced sampler (config): each batch should contain authentic and all three types in roughly equal proportion during the pilot.
- Optional (Phase 7): online synthetic forgeries (crop-paste copy-move with random scale and rotation, splice, `cv2.inpaint` of random regions) built from authentic images with exact masks, at most 20% of each batch.

**GATE 2:** manifests built; leakage check passes; shortcut-probe AUC <= 0.65; `scripts/visualize_batch.py` saves 16 samples (image, mask overlay, labels) to `outputs/debug/batch_vis.png`, and the user can confirm that masks line up with the forged regions.

---

## 6. Training

### 6.1 Defaults ([PAPER] unless tagged)
- Optimizer AdamW, learning rate 3e-4 for all trainable parameters (LoRA + heads), cosine schedule, 5 warmup epochs, 100 epochs on the main run (pilot: 20 epochs), input 512x512.
- Evaluate every 4 epochs and keep the best checkpoint by validation score.
- Weight decay 0.01 and gradient clip 1.0 [DESIGN; the paper does not specify].
- The paper used an effective batch of 240 via gradient accumulation. Our dataset is far smaller, so 240 gives very few optimizer steps per epoch. Default effective batch = **64** for the pilot [DESIGN]; try 240 on the main run if it trains stably. Micro-batch comes from GPU memory (start at 8 on a 16 GB T4).
- Precision: `torch.autocast(dtype=torch.float16)` + `GradScaler` (T4 has no fast bf16). The copy-move branch stays fp32. If NaNs appear, disable autocast to confirm the cause, then locate the op.
- Validation score used for checkpoint selection [DESIGN]: mean of (pixel-F1 on forged images, image-level AUC, macro-F1 of types). Log each term separately too.
- Checkpoints: `best.pt` and `last.pt` each evaluation. Resume restores model, optimizer, scheduler, scaler, epoch and RNG states.

### 6.2 Stages (one config per stage; run in order; compare)
| Stage | Config | Purpose |
|---|---|---|
| A | `lora.enabled=false`, mask decoder only | How well do frozen DINOv3 features alone localize? [DESIGN; the paper did not test this] |
| B | A + LoRA (r=64) | The paper recipe |
| C | B + detection head + type head | Multi-task |
| D | C + copy-move branch | Keep ONLY if it helps copy-move without hurting the rest |
| E (optional) | Larger backbone (ViT-B) or more data | Only after A to D are done |

Stage rules:
- B must beat A by a clear margin. If it does not, investigate (learning rate, data, bugs) before moving on.
- C must not lower pixel-F1 by more than 1 point versus B.
- D is kept only if copy-move AP and pixel-F1 on copy-move images improve (including on COVERAGE and CoMoFoD at the end) and other metrics do not degrade.

### 6.3 Never do
- Never fully fine-tune the backbone (the paper found it unstable with limited data). At train start, assert that no non-LoRA backbone parameter has `requires_grad=True`.
- Never choose thresholds, checkpoints or hyperparameters using Test B or Test C data.

---

## 7. Evaluation (`scripts/evaluate.py`)

Runs on any manifest. Outputs JSON + CSV + a printed table.

- **Image-level** (all images including authentic): ROC-AUC of the detection score; precision, recall and F1 at `tau_det`; false-positive rate at 95% true-positive rate. Do not use accuracy as the headline metric (class-balance dependent).
- **Type-level** (forged images with `type_valid`): per-class average precision and F1 at `tau_type`; a 3x3 confusion matrix on single-type images (argmax); and the rate of "detected as forged but wrong type".
- **Pixel-level** (forged images only): F1 and IoU at threshold 0.5. Report best-threshold F1 separately, labelled optimistic. Authentic images are excluded from pixel F1 (empty masks make it undefined), but report the mean fraction of pixels predicted forged on authentic images (false-alarm area).
- **Thresholds:** `tau_det`, `tau_type[k]`, `tau_pix` are chosen on VALIDATION only (for example, maximize F1 or fix FPR at 5%), saved to `thresholds.json` next to the checkpoint, then applied unchanged to every test set.
- **Robustness suite** (1000-image subset of Test A): Gaussian blur kernel {3,7,11,15,19,23}; JPEG quality {100,90,80,70,60,50}; Gaussian noise variance {3,7,11,15,19,23} [PAPER used these ranges]. Report pixel-F1 and image AUC per setting. The paper found blur by far the most damaging, so expect that.
- **Cross-dataset (Test B):** per dataset, report only the metrics that make sense (COVERAGE is copy-move only, so no type confusion matrix there).
- **Test C (modern inpainting):** image-level AUC and pixel-F1, reported separately from everything else.
- Always report: parameter counts (frozen / LoRA / heads), training time, peak GPU memory, median inference latency over 100 images.

---

## 8. Inference (`scripts/infer.py` and `predict(image_path)`)

Returns JSON:
```
{"forged": bool, "p_forged": float,
 "type_probs": {"copy_move": p, "splicing": p, "inpainting": p},
 "predicted_types": [...], "mask_png": path, "overlay_png": path}
```
- If not forged: `predicted_types = []` and an empty mask.
- Resize to 512 exactly as in training. Large images are resized in v1; native-resolution tiling is future work (document it).
- Always `model.eval()`, `torch.no_grad()`, LoRA dropout off.

---

## 9. Pitfalls and how to handle them

| ID | Pitfall | Symptom | Guard / handling |
|---|---|---|---|
| P1 | Wrong patch-token slicing (CLS and register tokens mixed in) | Shifted or garbled masks, shape errors | `split_tokens` helper + unit test (Section 3.1) |
| P2 | LoRA silently not applied, or applied to wrong modules | Stage B equals Stage A, or trainable count is 0 or huge | The four LoRA assertions (Section 3.2) |
| P3 | Backbone accidentally unfrozen (full fine-tuning) | Huge trainable count, unstable training | Assertion at train start; `param_report()` printed in the log |
| P4 | Gated model or old `transformers` | Load error | Gate 0 check; clear message telling the user to accept the license |
| P5 | Wrong input normalization | Slow convergence, weak features | Read mean/std from the processor config; test that they are read |
| P6 | fp16 overflow or NaN (affinity softmax, losses) | NaN loss | fp32 + autocast off in the copy-move branch; finite `-1e4` masking; `GradScaler`; NaN check each step; `set_detect_anomaly` only for debugging |
| P7 | BatchNorm with tiny micro-batch | Noisy or unstable validation | `decoder.norm: gn` option; default to `gn` when micro-batch < 8 |
| P8 | Format or compression shortcut (authentic vs forged pipelines) | Near-perfect validation detection but poor Test B results | Shortcut probe (Section 5.4); re-encode authentic images; always run cross-dataset tests |
| P9 | Source leakage (same COCO source in train and val/test) | Inflated validation scores | Split by `source_id` or perceptual-hash groups; `check_leakage.py` as a test |
| P10 | Class imbalance (types, authentic vs forged) | One type dominates; type head always outputs "no" | Balanced sampler; per-class `pos_weight`; per-class thresholds |
| P11 | Mask misalignment (different geometry on image vs mask, bilinear mask resize) | Low pixel-F1 despite good-looking predictions | Same geometric transform for both; nearest for masks; visual check at Gate 2 |
| P12 | Edge loss NaN on authentic images | NaN loss | `max(sum(band),1)` guard; unit test |
| P13 | Supervising the copy-move map with a target-only mask | Branch learns to ignore source regions | Never pixel-supervise `cm_map` (Section 3.3.4) |
| P14 | Repetitive textures (windows, tiles, foliage) create false copy-move matches | False alarms on authentic images | Near-neighbor mask; mutual-match normalization; include such authentic images as hard negatives; report FPR on them |
| P15 | Copy-move with scale or rotation | Missed matches | Synthetic copy-move augmentation with random scale and rotation [DESIGN]; ViT features are only partly invariant |
| P16 | Threshold or checkpoint chosen on test data | Optimistic results | Validation only; `thresholds.json` produced from validation |
| P17 | Pixel metrics averaged over authentic images | Undefined or inflated F1 | Forged-only pixel metrics + separate false-alarm area |
| P18 | Gradient accumulation bug | Wrong effective learning rate | Divide loss by `accum_steps`; unit test (Section 4) |
| P19 | Checkpoint reload mismatch (adapter config lost) | Different outputs after load | Save config JSON; test save -> load -> identical outputs |
| P20 | Train/eval mode mix-ups (dropout, BN) | Non-deterministic evaluation | Call `model.eval()`; test that two evaluation passes give equal outputs |
| P21 | OOM on a 16 GB T4 | CUDA out-of-memory | Lower micro-batch; gradient checkpointing (with PEFT it may need `enable_input_require_grads` [VERIFY]); never change image size silently |
| P22 | Resizing erases fine forensic traces | Weak splicing or inpainting detection | Known limitation. Keep train and test resizing identical. Later experiment: random native-resolution crops |
| P23 | Blur and heavy compression fragility | Large drop under blur | Robustness suite; blur and JPEG augmentation |
| P24 | DEFACTO inpainting predates diffusion inpainting | Fails on modern inpainting | Test C; later add generated data to training |
| P25 | Training labels are single-type, but the type head is multi-label | Cannot learn images with two manipulations | State this limitation in the README; later add synthetic chained forgeries; do not claim multi-type ability until tested |
| P26 | Types reported on authentic images | Contradictory outputs | Authentic images have zero type targets; inference suppresses types when not forged |
| P27 | Regression of the existing model | Old pipeline breaks | Keep old code path; add a smoke test for it |
| P28 | Non-determinism | Irreproducible runs | Seeds + logged commit hash |
| P29 | Label ambiguity (removal vs inpainting, splice plus retouch) | Type confusion that is not a model error | Log the confusion matrix; do not over-tune on it |
| P30 | Overfitting a small pilot set | Great train metrics, weak validation | Judge by validation only; weight decay; early stopping by best validation score |
| P31 | Slow data loading (large PNG or TIFF files) | GPU idle | `num_workers`, `persistent_workers`, pin memory; profile one epoch |
| P32 | Secrets in the repo | Leaked token | `HF_TOKEN` env var only; add a pre-commit check |

---

## 10. Phases and gates

Report format after each phase in `docs/PROGRESS.md`: what was done, commands run, results, issues, decisions.

| Phase | Work | GATE (must pass) |
|---|---|---|
| 0 | Repo audit (Section 2) | Gate 0 in Section 2 |
| 1 | Model skeleton: backbone loader (incl. `dummy`), LoRA, mask decoder, wrapper, save/load | Unit tests pass on CPU with `dummy`: output shapes for B=2 (`mask_logits (2,1,512,512)`); `split_tokens`; LoRA assertions; LoRA no-op test; save->load identical outputs. On GPU with the real backbone: one forward/backward on a batch of 4 without error |
| 2 | Data pipeline (Section 5) | Gate 2 in Section 5 |
| 3 | Losses + training loop + evaluation skeleton | Overfit test: 32 samples (mixed forged and authentic), 300 steps, loss falls by at least 80% and pixel-F1 on those 32 is at least 0.8. Gradient-accumulation test and checkpoint save/resume test pass |
| 4 | Stage A and Stage B pilot runs | No NaNs; B beats A on validation; results table in `docs/PROGRESS.md` |
| 5 | Stage C (detection + type heads) | Localization drop <= 1 point vs B; image-level AUC and per-type AP reported; types suppressed on non-forged outputs |
| 6 | Stage D (copy-move branch) | Unit tests: diagonal and near-neighbor entries masked; output finite with fp16 inputs; shapes correct. Keep-or-drop decision recorded using the Stage rules in 6.2 |
| 7 | Full evaluation: Test A, Test B, robustness suite, optional Test C generator (`scripts/make_inpaint_testset.py`) | `thresholds.json` from validation only; results tables written |
| 8 | Inference script, README, `docs/RESULTS.md` | `predict()` works on 5 sample images; RESULTS.md lists metrics, limitations and failed experiments |

---

## 11. Proposed layout (adapt names to the repo after Phase 0)

```
ftt_v2/
  models/   backbone.py  lora.py  heads.py  copy_move.py  forensic_model.py  outputs.py
  data/     manifest.py  dataset.py  transforms.py  splits.py
  losses.py  metrics.py  train.py  evaluate.py  infer.py
  utils/    seed.py  ckpt.py
configs/    ftt_v2_pilot_stageA.yaml ... stageD.yaml, ftt_v2_main.yaml
scripts/    build_manifest_<dataset>.py  check_leakage.py  shortcut_probe.py
            visualize_batch.py  make_inpaint_testset.py
tests/      test_backbone.py  test_lora.py  test_heads.py  test_losses.py
            test_leakage.py  test_ckpt.py  test_old_model_smoke.py
docs/       REPO_AUDIT.md  DECISIONS.md  OPEN_QUESTIONS.md  PROGRESS.md  RESULTS.md
```

---

## 12. Default config (starting point)

```yaml
model: {name: ftt_v2, image_size: 512, types: [copy_move, splicing, inpainting]}
backbone: {name: dinov3_vits16, hf_id: facebook/dinov3-vits16-pretrain-lvd1689m, frozen: true}
lora: {enabled: true, r: 64, alpha: 128, dropout: 0.0, targets: qkv_all_layers}
decoder: {norm: bn}            # use gn if micro_batch < 8
det: {enabled: false, use_mask_stats: true}      # on from Stage C
type_head: {enabled: false, pos_weight: null}    # on from Stage C
cm: {enabled: false, tap_layer: 4, proj_dim: 128, min_dist: 3, topk: 8, tau_init: 0.1}  # on in Stage D
loss: {lambda_edge: 20, edge_band_px: 7, lambda_det: 1.0, lambda_type: 1.0}
train: {epochs: 20, warmup_epochs: 5, lr: 3.0e-4, weight_decay: 0.01, grad_clip: 1.0,
        micro_batch: 8, effective_batch: 64, amp: fp16, eval_every: 4, seed: 42}
data: {manifest_dir: data/manifests, balanced_sampler: true, num_workers: 4}
```

---

## 13. Open items the agent must resolve or ask about

1. Exact module names for Q/K/V in the installed DINOv3 implementation.
2. Number and position of register tokens; the `hidden_states` indexing convention.
3. Whether DEFACTO filenames encode the source COCO image id.
4. Whether DEFACTO copy-move masks include the source region or only the pasted region.
5. CASIA v2 type labels (`Tp_S` / `Tp_D`) and NIST16 manipulation metadata.
6. The official repo's exact edge-loss normalization.
7. Dataset paths and HF token availability (ask the user).

## 14. Definition of done

- All gates 0 to 8 pass and are logged in `docs/PROGRESS.md`.
- Stage A to D comparison table exists, with the keep-or-drop decision for the copy-move branch.
- Evaluation covers Test A, Test B, the robustness suite and (if built) Test C, with thresholds chosen on validation only.
- The old model still runs (smoke test passes).
- `docs/RESULTS.md` states limitations honestly, including: single-type training labels, untested modern inpainting if Test C was not built, fixed 512x512 resizing, and which pieces are [DESIGN] hypotheses rather than validated results.
