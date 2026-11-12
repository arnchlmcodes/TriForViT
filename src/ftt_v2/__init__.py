"""
FTT v2: DINOv3 + LoRA multi-task image forgery model.

Detect, classify type, and localize forged regions.
"""

import os
os.environ.setdefault("USE_TF", "0")

# Forgery type constants — used everywhere in the codebase.
# The order is fixed and must not change.
TYPES = ["copy_move", "splicing", "inpainting"]
