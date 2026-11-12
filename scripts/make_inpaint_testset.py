"""
make_inpaint_testset.py — Generate Test C benchmark (inpainting evaluation).

Generates inpainting forgeries from authentic images using random masks
and inpainting (via OpenCV inpaint or diffusion), saving exact ground-truth masks.
Creates data/manifests/test_C.csv.

Usage:
    python scripts/make_inpaint_testset.py \
        --input-dir path/to/authentic_images \
        --output-dir data/test_C \
        --num-samples 500
"""

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.ftt_v2.data.manifest import save_manifest


def generate_random_mask(height: int, width: int, rng: random.Random) -> np.ndarray:
    """Generate a random binary mask containing rectangles, ellipses, and brush strokes."""
    mask = np.zeros((height, width), dtype=np.uint8)

    # 1. Random rectangles
    num_shapes = rng.randint(1, 3)
    for _ in range(num_shapes):
        shape_type = rng.choice(["rect", "ellipse", "stroke"])
        if shape_type == "rect":
            x1 = rng.randint(0, width - 40)
            y1 = rng.randint(0, height - 40)
            w = rng.randint(30, min(150, width - x1))
            h = rng.randint(30, min(150, height - y1))
            cv2.rectangle(mask, (x1, y1), (x1 + w, y1 + h), 255, -1)
        elif shape_type == "ellipse":
            cx = rng.randint(40, width - 40)
            cy = rng.randint(40, height - 40)
            axes = (rng.randint(20, 80), rng.randint(20, 80))
            angle = rng.randint(0, 180)
            cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)
        else:
            # Stroke
            num_pts = rng.randint(3, 8)
            pts = [(rng.randint(20, width - 20), rng.randint(20, height - 20)) for _ in range(num_pts)]
            thickness = rng.randint(15, 45)
            for i in range(len(pts) - 1):
                cv2.line(mask, pts[i], pts[i + 1], 255, thickness)

    return mask


def main():
    parser = argparse.ArgumentParser(description="Generate inpainting Test C set")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory of authentic images")
    parser.add_argument("--output-dir", type=str, default="data/test_C", help="Output directory for generated dataset")
    parser.add_argument("--manifest-path", type=str, default="data/manifests/test_C.csv")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of forged + authentic pairs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_img_dir = out_dir / "images"
    out_mask_dir = out_dir / "masks"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_imgs = [p for p in in_dir.rglob("*") if p.suffix.lower() in exts]

    if not all_imgs:
        print(f"[ERROR] No images found in {in_dir}")
        sys.exit(1)

    rng.shuffle(all_imgs)
    selected = all_imgs[: args.num_samples]

    records = []

    for idx, img_path in enumerate(selected):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        h, w, c = img_bgr.shape

        # 1. Authentic record
        auth_name = f"testC_auth_{idx:05d}.png"
        auth_save_path = out_img_dir / auth_name
        cv2.imwrite(str(auth_save_path), img_bgr)

        records.append({
            "image_id": f"testC_auth_{idx:05d}",
            "path": str(auth_save_path.resolve()),
            "mask_path": "",
            "is_forged": 0,
            "copy_move": 0,
            "splicing": 0,
            "inpainting": 0,
            "type_valid": 1,
            "source_id": f"testC_src_{idx:05d}",
            "dataset": "test_C_inpainting",
            "width": w,
            "height": h,
            "file_format": "PNG",
        })

        # 2. Inpainted forged record
        mask = generate_random_mask(h, w, rng)
        # Apply Telea or Navier-Stokes inpainting
        method = cv2.INPAINT_TELEA if (idx % 2 == 0) else cv2.INPAINT_NS
        inpainted_bgr = cv2.inpaint(img_bgr, mask, 3, method)

        forged_name = f"testC_forged_{idx:05d}.png"
        mask_name = f"testC_mask_{idx:05d}.png"
        forged_save_path = out_img_dir / forged_name
        mask_save_path = out_mask_dir / mask_name

        cv2.imwrite(str(forged_save_path), inpainted_bgr)
        cv2.imwrite(str(mask_save_path), mask)

        records.append({
            "image_id": f"testC_forged_{idx:05d}",
            "path": str(forged_save_path.resolve()),
            "mask_path": str(mask_save_path.resolve()),
            "is_forged": 1,
            "copy_move": 0,
            "splicing": 0,
            "inpainting": 1,
            "type_valid": 1,
            "source_id": f"testC_src_{idx:05d}",
            "dataset": "test_C_inpainting",
            "width": w,
            "height": h,
            "file_format": "PNG",
        })

    df = pd.DataFrame(records)
    save_manifest(df, args.manifest_path)
    print(f"Generated {len(df)} samples ({args.manifest_path})")


if __name__ == "__main__":
    main()
