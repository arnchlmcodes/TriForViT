"""
build_manifest_defacto.py — Build manifest CSV for DEFACTO + COCO authentic images.

DEFACTO contains:
  - copymove_img / copymove_annotations (masks)
  - splicing_img / splicing_annotations (masks)
  - inpainting_img / inpainting_annotations (masks)

Plus COCO authentic negatives (unmanipulated images).
Extracts COCO image ID from filenames to use as source_id for source-disjoint splits.

Usage:
    python scripts/build_manifest_defacto.py \
        --defacto-dir path/to/defacto \
        --coco-dir path/to/coco \
        --output-dir data/manifests
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from PIL import Image

from src.ftt_v2.data.manifest import MANIFEST_COLUMNS, save_manifest
from src.ftt_v2.data.splits import split_by_source_id


def extract_coco_id(filename: str) -> str:
    """Extract COCO image id from DEFACTO filename or return filename stem."""
    # Matches patterns like COCO_train2014_000000123456 or 123456
    m = re.search(r"(\d{6,12})", filename)
    if m:
        return m.group(1)
    return Path(filename).stem


def scan_defacto_category(
    img_dir: Path,
    mask_dir: Path,
    category: str,
) -> List[dict]:
    """Scan a DEFACTO category folder and pair images with their ground-truth masks."""
    records = []
    if not img_dir.exists():
        print(f"[WARN] Directory not found: {img_dir}")
        return records

    # Supported extensions
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    img_files = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]

    for img_path in img_files:
        stem = img_path.stem
        # Look for matching mask
        mask_path = None
        if mask_dir.exists():
            for m_ext in [".png", ".jpg", ".tif"]:
                cand = mask_dir / f"{stem}{m_ext}"
                if cand.exists():
                    mask_path = str(cand)
                    break
                # Some DEFACTO masks have _mask suffix
                cand_mask = mask_dir / f"{stem}_mask{m_ext}"
                if cand_mask.exists():
                    mask_path = str(cand_mask)
                    break

        try:
            with Image.open(img_path) as im:
                w, h = im.size
                fmt = im.format or img_path.suffix.lstrip(".").upper()
        except Exception:
            w, h, fmt = 512, 512, img_path.suffix.lstrip(".").upper()

        coco_id = extract_coco_id(img_path.name)

        rec = {
            "image_id": f"defacto_{category}_{stem}",
            "path": str(img_path.resolve()),
            "mask_path": mask_path or "",
            "is_forged": 1,
            "copy_move": 1 if category == "copy_move" else 0,
            "splicing": 1 if category == "splicing" else 0,
            "inpainting": 1 if category == "inpainting" else 0,
            "type_valid": 1,
            "source_id": f"coco_{coco_id}",
            "dataset": f"defacto_{category}",
            "width": w,
            "height": h,
            "file_format": fmt,
        }
        records.append(rec)

    return records


def scan_coco_authentics(
    coco_dir: Path,
    exclude_source_ids: set,
    limit: Optional[int] = None,
) -> List[dict]:
    """Scan COCO directory for authentic negative images, excluding any used in DEFACTO."""
    records = []
    if not coco_dir.exists():
        print(f"[WARN] COCO directory not found: {coco_dir}")
        return records

    exts = {".jpg", ".jpeg", ".png"}
    files = [p for p in coco_dir.rglob("*") if p.suffix.lower() in exts]

    for p in files:
        coco_id = extract_coco_id(p.name)
        source_id = f"coco_{coco_id}"
        if source_id in exclude_source_ids:
            continue  # Leakage prevention!

        try:
            with Image.open(p) as im:
                w, h = im.size
                fmt = im.format or p.suffix.lstrip(".").upper()
        except Exception:
            w, h, fmt = 512, 512, p.suffix.lstrip(".").upper()

        rec = {
            "image_id": f"coco_auth_{p.stem}",
            "path": str(p.resolve()),
            "mask_path": "",  # Empty mask for authentic
            "is_forged": 0,
            "copy_move": 0,
            "splicing": 0,
            "inpainting": 0,
            "type_valid": 1,
            "source_id": source_id,
            "dataset": "coco_authentic",
            "width": w,
            "height": h,
            "file_format": fmt,
        }
        records.append(rec)
        if limit and len(records) >= limit:
            break

    return records


def main():
    parser = argparse.ArgumentParser(description="Build DEFACTO + COCO manifests")
    parser.add_argument("--defacto-dir", type=str, required=True, help="Path to DEFACTO root")
    parser.add_argument("--coco-dir", type=str, required=True, help="Path to COCO authentic images")
    parser.add_argument("--output-dir", type=str, default="data/manifests")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    defacto_root = Path(args.defacto_dir)
    coco_root = Path(args.coco_dir)
    out_dir = Path(args.output_dir)

    all_records = []

    # 1. DEFACTO Copy-Move
    cm_records = scan_defacto_category(
        defacto_root / "copymove_img",
        defacto_root / "copymove_annotations",
        "copy_move",
    )
    all_records.extend(cm_records)
    print(f"Loaded {len(cm_records)} DEFACTO copy-move samples")

    # 2. DEFACTO Splicing
    sp_records = scan_defacto_category(
        defacto_root / "splicing_img",
        defacto_root / "splicing_annotations",
        "splicing",
    )
    all_records.extend(sp_records)
    print(f"Loaded {len(sp_records)} DEFACTO splicing samples")

    # 3. DEFACTO Inpainting
    ip_records = scan_defacto_category(
        defacto_root / "inpainting_img",
        defacto_root / "inpainting_annotations",
        "inpainting",
    )
    all_records.extend(ip_records)
    print(f"Loaded {len(ip_records)} DEFACTO inpainting samples")

    # 4. Source IDs used by DEFACTO
    defacto_sources = {r["source_id"] for r in all_records}

    # 5. COCO authentic negatives (source-disjoint from DEFACTO)
    auth_records = scan_coco_authentics(coco_root, exclude_source_ids=defacto_sources)
    all_records.extend(auth_records)
    print(f"Loaded {len(auth_records)} authentic COCO samples (disjoint from DEFACTO)")

    df = pd.DataFrame(all_records)
    if len(df) == 0:
        print("[ERROR] No records collected. Check directories.")
        sys.exit(1)

    # Split by source_id
    splits = split_by_source_id(
        df,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_df in splits.items():
        save_path = out_dir / f"{name}.csv"
        save_manifest(split_df, save_path)
        print(f"Saved {name} split ({len(split_df)} samples) to {save_path}")


if __name__ == "__main__":
    main()
