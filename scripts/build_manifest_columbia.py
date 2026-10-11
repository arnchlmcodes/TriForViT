"""
build_manifest_columbia.py — Build manifest CSV for Columbia benchmark (Test B).

Columbia contains uncompressed image splicing (180 spliced images, 183 authentic).
Labels: splicing only.

Usage:
    python scripts/build_manifest_columbia.py \
        --data-dir path/to/columbia \
        --output data/manifests/test_B_columbia.csv
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
    parser = argparse.ArgumentParser(description="Build Columbia benchmark manifest")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to Columbia dataset root")
    parser.add_argument("--output", type=str, default="data/manifests/test_B_columbia.csv")
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    records = []

    exts = {".tif", ".tiff", ".png", ".bmp", ".jpg"}
    all_files = [p for p in data_root.rglob("*") if p.suffix.lower() in exts and "edgemask" not in p.stem.lower()]

    for p in all_files:
        stem = p.stem.lower()
        # In Columbia: 'authentic' or 'auth' in name or path means authentic, 'splice' or 'spliced' means spliced
        is_spliced = "splice" in stem or "sp" in stem or "splice" in str(p.parent).lower()

        mask_path = ""
        if is_spliced:
            # Look for ground truth mask
            for m_ext in [".bmp", ".png", ".tif"]:
                cand = p.parent / f"{p.stem}_edgemask{m_ext}"
                if cand.exists():
                    mask_path = str(cand.resolve())
                    break
                cand_mask = p.parent / f"{p.stem}_mask{m_ext}"
                if cand_mask.exists():
                    mask_path = str(cand_mask.resolve())
                    break

        try:
            with Image.open(p) as im:
                w, h = im.size
                fmt = im.format or p.suffix.lstrip(".").upper()
        except Exception:
            w, h, fmt = 512, 512, p.suffix.lstrip(".").upper()

        rec = {
            "image_id": f"columbia_{p.stem}",
            "path": str(p.resolve()),
            "mask_path": mask_path,
            "is_forged": 1 if is_spliced else 0,
            "copy_move": 0,
            "splicing": 1 if is_spliced else 0,
            "inpainting": 0,
            "type_valid": 1,
            "source_id": f"columbia_{p.stem}",
            "dataset": "columbia",
            "width": w,
            "height": h,
            "file_format": fmt,
        }
        records.append(rec)

    df = pd.DataFrame(records)
    print(f"Collected {len(df)} samples ({df['is_forged'].sum()} spliced, {(df['is_forged']==0).sum()} authentic)")
    save_manifest(df, args.output)
    print(f"Saved manifest to {args.output}")


if __name__ == "__main__":
    main()
