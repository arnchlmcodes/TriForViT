# Open Questions

Items the agent cannot resolve without user input or external verification.

---

## OQ1: Hugging Face Token for DINOv3

The model `facebook/dinov3-vits16-pretrain-lvd1689m` is license-gated. Need user to:
1. Accept the license on the model page
2. Provide HF token via `HF_TOKEN` environment variable

**Status:** PENDING — user notified, awaiting login.

## OQ2: Dataset Paths

The plan does NOT download datasets. Need user-provided paths for:
- DEFACTO (copy-move, splicing, inpainting)
- MS-COCO authentic images
- COVERAGE, CoMoFoD, Columbia, IMD2020, NIST16, CASIA v2 (for test sets)

**Status:** PENDING — will ask user at Phase 2.

## OQ3: DEFACTO Filename Encoding

Does the DEFACTO filename encode the source COCO image ID?
Needed for source-disjoint splitting.

**Status:** PENDING — will inspect files when datasets are provided.

## OQ4: DEFACTO Copy-Move Masks

Do DEFACTO copy-move masks include the source region or only the pasted region?
Critical for the copy-move branch (Section 3.3.4 of plan).

**Status:** PENDING — will verify per dataset.

## OQ5: CASIA v2 Type Labels

Do `Tp_S` / `Tp_D` prefixes in CASIA v2 filenames encode same-source vs different-source?
Need to inspect files and README.

**Status:** PENDING.

## OQ6: NIST16 Manipulation Metadata

How are manipulation types encoded in NIST16 metadata?

**Status:** PENDING.
