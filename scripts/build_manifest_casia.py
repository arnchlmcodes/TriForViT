"""
build_manifest_casia.py — Build manifest CSV for CASIA v2 benchmark (Test B).

CASIA v2 structure:
  - Au/ : Authentic images
  - Tp/ : Tampered images
    * Tp_S_* : Same-source tampering (copy-move)
    * Tp_D_* : Different-source tampering (splicing)
Ground-truth masks (CASIA v2 Groundtruth folder).

Usage:
    python scripts/build_manifest_casia.py \
        --data-dir path/to/CASIA2 \
        --mask-dir path/to/CASIA2_Groundtruth \
        --output data/manifests/test_B_casia2.csv
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from PIL import Image

from src.ftt_v2.data.manifest import save_manifest


def main():
    parser = argparse.ArgumentParser(description="Build CASIA v2 benchmark manifest")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to CASIA v2 root (Au and Tp folders)")
    parser.add_argument("--mask-dir", type=str, default="", help="Path to CASIA v2 groundtruth masks")
    parser.add_argument("--output", type=str, default="data/manifests/test_B_casia2.csv")
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    mask_root = Path(args.mask_dir) if args.mask_dir else None
    records = []

    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    # 1. Authentic images (Au)
    au_dir = data_root / "Au" if (data_root / "Au").exists() else None
    if au_dir:
        for p in au_dir.rglob("*"):
            if p.suffix.lower() in exts:
                try:
                    with Image.open(p) as im:
                        w, h = im.size
                        fmt = im.format or p.suffix.lstrip(".").upper()
                except Exception:
                    w, h, fmt = 512, 512, p.suffix.lstrip(".").upper()

                rec = {
                    "image_id": f"casia2_au_{p.stem}",
                    "path": str(p.resolve()),
                    "mask_path": "",
                    "is_forged": 0,
                    "copy_move": 0,
                    "splicing": 0,
                    "inpainting": 0,
                    "type_valid": 1,
                    "source_id": f"casia2_{p.stem}",
                    "dataset": "casia2",
                    "width": w,
                    "height": h,
                    "file_format": fmt,
                }
                records.append(rec)

    # 2. Tampered images (Tp)
    tp_dir = data_root / "Tp" if (data_root / "Tp").exists() else None
    if tp_dir:
        for p in tp_dir.rglob("*"):
            if p.suffix.lower() in exts:
                stem = p.stem
                is_cm = "tp_s" in stem.lower()
                is_sp = "tp_d" in stem.lower()

                mask_path = ""
                if mask_root and mask_root.exists():
                    for m_ext in [".png", ".jpg", ".tif", ".bmp"]:
                        cand = mask_root / f"{stem}_gt{m_ext}"
                        if cand.exists():
                            mask_path = str(cand.resolve())
                            break
                        cand2 = mask_root / f"{stem}{m_ext}"
                        if cand2.exists():
                            mask_path = str(cand2.resolve())
                            break

                try:
                    with Image.open(p) as im:
                        w, h = im.size
                        fmt = im.format or p.suffix.lstrip(".").upper()
                except Exception:
                    w, h, fmt = 512, 512, p.suffix.lstrip(".").upper()

                rec = {
                    "image_id": f"casia2_tp_{stem}",
                    "path": str(p.resolve()),
                    "mask_path": mask_path,
                    "is_forged": 1,
                    "copy_move": 1 if is_cm else 0,
                    "splicing": 1 if is_sp else 0,
                    "inpainting": 0,
                    "type_valid": 1 if (is_cm or is_sp) else 0,
                    "source_id": f"casia2_{stem}",
                    "dataset": "casia2",
                    "width": w,
                    "height": h,
                    "file_format": fmt,
                }
                records.append(rec)

    df = pd.DataFrame(records)
    print(f"Collected {len(df)} samples ({df['is_forged'].sum()} tampered, {(df['is_forged']==0).sum()} authentic)")
    save_manifest(df, args.output)
    print(f"Saved manifest to {args.output}")


if __name__ == "__main__":
    main()
