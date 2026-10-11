"""
FTT v2 data pipeline: manifest-based dataset with mask loading and augmentation.
"""

from .dataset import BalancedBatchSampler, ManifestDataset
from .manifest import MANIFEST_COLUMNS, load_manifest, save_manifest, validate_manifest
from .splits import group_by_phash, split_by_source_id
from .transforms import get_eval_transforms, get_train_transforms

__all__ = [
    "BalancedBatchSampler",
    "ManifestDataset",
    "MANIFEST_COLUMNS",
    "load_manifest",
    "save_manifest",
    "validate_manifest",
    "group_by_phash",
    "split_by_source_id",
    "get_train_transforms",
    "get_eval_transforms",
]
