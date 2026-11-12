"""
kaggle_preflight.py — Kaggle Pre-Training Diagnostic and Verification Script.

Runs comprehensive checks on:
  1. Python & Hardware environment (CUDA, GPU name, VRAM).
  2. Critical package dependencies.
  3. Directory permissions (writable /kaggle/working).
  4. Configuration file validity.
  5. Hugging Face authentication & DINOv3 model accessibility.
  6. Dataset manifests & image/mask readability (if available).
  7. Synthetic end-to-end forward/backward execution on available accelerator.

Usage:
    python scripts/kaggle_preflight.py --config configs/ftt_v2_kaggle.yaml
"""

from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


class PreflightChecker:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.results: List[Tuple[str, str, str]] = []  # (Category, Status, Details)
        self.cfg = {}

    def log(self, category: str, status: str, details: str):
        self.results.append((category, status, details))
        status_colored = {
            "PASS": f"[PASS]",
            "FAIL": f"[FAIL]",
            "WARN": f"[WARN]",
            "SKIP": f"[SKIP]",
        }.get(status, f"[{status}]")
        print(f"{status_colored:<8} | {category:<25} | {details}")

    def check_environment(self):
        print("\n--- 1. Environment & Hardware ---")
        py_ver = sys.version.split()[0]
        if sys.version_info >= (3, 10):
            self.log("Python Version", "PASS", f"Python {py_ver}")
        else:
            self.log("Python Version", "WARN", f"Python {py_ver} (Recommend >= 3.10)")

        try:
            import torch
            if torch.cuda.is_available():
                n_gpus = torch.cuda.device_count()
                gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.log(
                    "CUDA Acceleration",
                    "PASS",
                    f"CUDA {torch.version.cuda} | GPUs: {n_gpus} ({', '.join(gpu_names)}) | VRAM: {vram_gb:.1f} GB",
                )
                if vram_gb < 14.0:
                    self.log(
                        "GPU Memory Sizing",
                        "WARN",
                        f"{vram_gb:.1f} GB VRAM detected. ViT 512x512 recommends >= 15 GB (e.g. T4).",
                    )
                else:
                    self.log("GPU Memory Sizing", "PASS", f"{vram_gb:.1f} GB VRAM is sufficient for T4 recipes.")
            else:
                self.log(
                    "CUDA Acceleration",
                    "WARN",
                    "No CUDA GPU detected! CPU-only mode will be extremely slow for ViT 512x512.",
                )
        except Exception as e:
            self.log("CUDA Acceleration", "FAIL", f"Failed importing torch: {e}")

    def check_dependencies(self):
        print("\n--- 2. Dependency Verification ---")
        req_packages = [
            ("torch", "torch"),
            ("torchvision", "torchvision"),
            ("transformers", "transformers"),
            ("peft", "peft"),
            ("yaml", "PyYAML"),
            ("pandas", "pandas"),
            ("imagehash", "imagehash"),
            ("sklearn", "scikit-learn"),
            ("cv2", "opencv-python"),
            ("PIL", "Pillow"),
            ("tqdm", "tqdm"),
        ]
        all_ok = True
        for mod_name, pkg_label in req_packages:
            try:
                mod = __import__(mod_name)
                ver = getattr(mod, "__version__", "installed")
                self.log(f"Pkg: {pkg_label}", "PASS", f"v{ver}")
            except ImportError:
                self.log(f"Pkg: {pkg_label}", "FAIL", "Missing! Run: pip install " + pkg_label)
                all_ok = False
        return all_ok

    def check_filesystem(self):
        print("\n--- 3. Filesystem & Working Directory ---")
        # Check Kaggle working dir or local dir
        paths_to_test = [Path("/kaggle/working"), Path("outputs"), Path(".")]
        writable_found = False
        for p in paths_to_test:
            try:
                p.mkdir(parents=True, exist_ok=True)
                test_file = p / ".preflight_write_test.tmp"
                test_file.write_text("ok")
                test_file.unlink()
                self.log(f"Writability: {p}", "PASS", "Directory is writable")
                writable_found = True
                break
            except Exception:
                continue
        if not writable_found:
            self.log("Writability", "FAIL", "Cannot find a writable working directory!")

    def check_configuration(self):
        print("\n--- 4. Configuration Check ---")
        if not self.config_path.exists():
            self.log("Config File", "FAIL", f"Config path {self.config_path} does not exist!")
            return False

        try:
            import yaml
            with open(self.config_path) as f:
                self.cfg = yaml.safe_load(f)
            self.log("Config File", "PASS", f"Loaded {self.config_path.name}")

            # Verify key sections
            for sec in ["model", "backbone", "lora", "decoder", "loss", "train", "data"]:
                if sec in self.cfg:
                    self.log(f"Config section: {sec}", "PASS", "Present")
                else:
                    self.log(f"Config section: {sec}", "FAIL", "Missing section!")
            return True
        except Exception as e:
            self.log("Config File", "FAIL", f"Error parsing YAML: {e}")
            return False

    def check_huggingface_auth(self):
        print("\n--- 5. Hugging Face & DINOv3 Access ---")
        token = os.environ.get("HF_TOKEN", None)
        backbone_cfg = self.cfg.get("backbone", {})
        bb_name = backbone_cfg.get("name", "dinov3_vits16")
        hf_id = backbone_cfg.get("hf_id", "facebook/dinov3-vits16-pretrain-lvd1689m")

        if bb_name == "dummy":
            self.log("DINOv3 Auth", "SKIP", "Using dummy backbone (HF token not required)")
            return

        if token:
            masked = token[:4] + "..." + token[-4:] if len(token) > 8 else "***"
            self.log("HF_TOKEN Found", "PASS", f"Token configured: {masked}")
        else:
            self.log(
                "HF_TOKEN Found",
                "WARN",
                "HF_TOKEN environment variable not set. Loading gated DINOv3 models may fail.",
            )

        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            info = api.model_info(hf_id)
            self.log("Model Existence", "PASS", f"Model '{hf_id}' verified on Hugging Face (gated: {info.gated})")
        except Exception as e:
            err = str(e)
            if "401" in err or "403" in err or "gated" in err.lower():
                self.log(
                    "DINOv3 Model Access",
                    "WARN",
                    f"Repository '{hf_id}' is gated. Ensure you accepted the license at https://huggingface.co/{hf_id}",
                )
            else:
                self.log("DINOv3 Model Access", "WARN", f"Could not verify '{hf_id}' online: {e}")

    def check_dataset_manifests(self):
        print("\n--- 6. Dataset Manifests & Image Access ---")
        data_cfg = self.cfg.get("data", {})
        manifest_dir = Path(data_cfg.get("manifest_dir", "data/manifests"))
        image_root = data_cfg.get("image_root", None)

        train_csv = manifest_dir / "train.csv"
        val_csv = manifest_dir / "val.csv"

        if not train_csv.exists() or not val_csv.exists():
            self.log(
                "Dataset Manifests",
                "SKIP",
                f"Manifests not found at {manifest_dir} (Generate with scripts/build_manifest_*.py before training)",
            )
            return

        try:
            import pandas as pd
            from PIL import Image
            from src.ftt_v2.data.manifest import validate_manifest

            df_train = pd.read_csv(train_csv)
            errs = validate_manifest(df_train)
            if errs:
                self.log("Train Manifest Schema", "FAIL", f"Schema errors: {errs[:2]}")
            else:
                self.log("Train Manifest Schema", "PASS", f"{len(df_train)} rows conforming to FTT v2 schema")

            # Check sample images
            from src.ftt_v2.data.dataset import ForensicDataset
            ds = ForensicDataset(str(train_csv), image_root=image_root)
            sample = ds[0]
            self.log(
                "Dataset Sample Read",
                "PASS",
                f"Sample 0 loaded: image {tuple(sample['image'].shape)}, mask {tuple(sample['mask'].shape)}, forged={sample['is_forged'].item()}",
            )
        except Exception as e:
            self.log("Dataset Loading", "FAIL", f"Error testing dataset loading: {e}")

    def check_model_forward_backward(self):
        print("\n--- 7. End-to-End Synthetic Execution ---")
        try:
            import torch
            from src.ftt_v2.forensic_model import ForensicModel
            from src.ftt_v2.losses import FTTv2Loss

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Create test config with dummy backbone to verify pipeline mechanics safely
            test_cfg = dict(self.cfg)
            test_cfg["backbone"] = {"name": "dummy", "embed_dim": 384, "depth": 2, "patch_size": 16}
            test_cfg["model"] = {"name": "ftt_v2", "image_size": 512, "types": ["copy_move", "splicing", "inpainting"]}

            model = ForensicModel(test_cfg).to(device)
            loss_fn = FTTv2Loss().to(device)
            optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)

            # Dummy input
            dummy_img = torch.randn(2, 3, 512, 512, device=device)
            dummy_mask = torch.zeros(2, 1, 512, 512, device=device)
            dummy_is_f = torch.tensor([0.0, 1.0], device=device)
            dummy_types = torch.zeros(2, 3, device=device)
            dummy_valid = torch.ones(2, 3, device=device)

            optimizer.zero_grad()
            out = model(dummy_img)

            self.log(
                "Forward Pass Output",
                "PASS",
                f"mask_logits: {tuple(out.mask_logits.shape)}, det: {tuple(out.det_logit.shape)}, type: {tuple(out.type_logits.shape)}",
            )

            losses = loss_fn(
                out.mask_logits, dummy_mask,
                out.det_logit, dummy_is_f,
                out.type_logits, dummy_types, dummy_valid,
            )
            loss = losses["total"]
            if torch.isfinite(loss):
                self.log("Loss Computation", "PASS", f"Total loss: {loss.item():.4f} (Finite)")
            else:
                self.log("Loss Computation", "FAIL", "Loss is NaN or Inf!")

            loss.backward()
            optimizer.step()
            self.log("Backward & Optimizer", "PASS", "Gradients propagated and optimizer updated successfully")

        except Exception as e:
            self.log("Synthetic Execution", "FAIL", f"Failed smoke forward/backward: {e}")

    def print_summary(self):
        print("\n" + "=" * 70)
        print("KAGGLE PREFLIGHT AUDIT SUMMARY")
        print("=" * 70)
        passes = sum(1 for _, s, _ in self.results if s == "PASS")
        fails = sum(1 for _, s, _ in self.results if s == "FAIL")
        warns = sum(1 for _, s, _ in self.results if s == "WARN")
        skips = sum(1 for _, s, _ in self.results if s == "SKIP")

        print(f"Total Checks: {len(self.results)} | PASS: {passes} | WARN: {warns} | SKIP: {skips} | FAIL: {fails}\n")

        if fails > 0:
            print("[VERDICT] FAIL — Issues detected above that will prevent training. Fix before running.")
            sys.exit(1)
        elif warns > 0:
            print("[VERDICT] PASS WITH WARNINGS — Code and environment are valid. Review warnings (e.g. HF_TOKEN or GPU).")
        else:
            print("[VERDICT] PASS — Ready for training!")


def main():
    parser = argparse.ArgumentParser(description="Kaggle Preflight Verification")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ftt_v2_kaggle.yaml",
        help="Path to YAML configuration",
    )
    args = parser.parse_args()

    checker = PreflightChecker(args.config)
    checker.check_environment()
    checker.check_dependencies()
    checker.check_filesystem()
    if checker.check_configuration():
        checker.check_huggingface_auth()
        checker.check_dataset_manifests()
        checker.check_model_forward_backward()
    checker.print_summary()


if __name__ == "__main__":
    main()
