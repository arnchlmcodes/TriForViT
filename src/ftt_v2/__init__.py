"""
FTT v2: DINOv3 + LoRA multi-task image forgery model.

Detect, classify type, and localize forged regions.
"""

# Forgery type constants — used everywhere in the codebase.
# The order is fixed and must not change.
TYPES = ["copy_move", "splicing", "inpainting"]
