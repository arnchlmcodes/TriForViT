"""
build_manifest_coverage.py — Build manifest CSV for COVERAGE benchmark (Test B).

COVERAGE is copy-move only (100 forged images, 100 authentic images).
Masks mark forged regions.

Usage:
    python scripts/build_manifest_coverage.py \
        --data-dir path/to/coverage \
        --output data/manifests/test_B_coverage.csv
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
    parser = argparse.ArgumentParser(description="Build COVERAGE benchmark manifest")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to COVERAGE root")
    parser.add_argument("--output", type=str, default="data/manifests/test_B_coverage.csv")
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    records = []

    # COVERAGE directory layout typically contains image/ and mask/
    img_dir = data_root / "image" if (data_root / "image").exists() else data_root
    mask_dir = data_root / "mask" if (data_root / "mask").exists() else data_root

    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    images = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts and "mask" not in p.stem.lower()]

    for p in images:
        stem = p.stem
        # Check if authentic (COVERAGE often prefixes or suffixes authentic vs tampered)
        is_tampered = "forged" in stem.lower() or "tampered" in stem.lower() or stem.endswith("t")
        if not is_tampered and (stem.endswith("copy") or stem.isdigit()):
            is_tampered = True

        mask_path = ""
        if is_tampered and mask_dir.exists():
            for m_ext in [".tif", ".png", ".jpg"]:
                cand = mask_dir / f"{stem}{m_ext}"
                if cand.exists():
                    mask_path = str(cand.resolve())
                    break
                cand_forged = mask_dir / f"{stem}forged{m_ext}"
                if cand_forged.exists():
                    mask_path = str(cand_forged.resolve())
                    break

        try:
            with Image.open(p) as im:
                w, h = im.size
                fmt = im.format or p.suffix.lstrip(".").upper()
        except Exception:
            w, h, fmt = 512, 512, p.suffix.lstrip(".").upper()

        rec = {
            "image_id": f"coverage_{stem}",
            "path": str(p.resolve()),
            "mask_path": mask_path,
            "is_forged": 1 if is_tampered else 0,
            "copy_move": 1 if is_tampered else 0,
            "splicing": 0,
            "inpainting": 0,
            "type_valid": 1,
            "source_id": f"coverage_{stem.rstrip('t').rstrip('forged')}",
            "dataset": "coverage",
            "width": w,
            "height": h,
            "file_format": fmt,
        }
        records.append(rec)

    df = pd.DataFrame(records)
    print(f"Collected {len(df)} samples ({df['is_forged'].sum()} forged, {(df['is_forged']==0).sum()} authentic)")
    save_manifest(df, args.output)
    print(f"Saved manifest to {args.output}")


if __name__ == "__main__":
    main()
