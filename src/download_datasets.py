"""
datasets.py - Online Dataset Links & References

This file lists the online sources for all datasets used in the project.
Datasets are loaded directly from these sources during training (e.g. via
Kaggle API, HuggingFace, or direct URL streaming).

No local data/ directory is used. All dataset access is online.
"""

# ---------------------------------------------------------------------------
# Dataset URLs and References
# ---------------------------------------------------------------------------

DATASETS = {
    "CASIA_v2": {
        "description": "CASIA v2.0 Image Tampering Detection Dataset (~12K images)",
        "forgery_types": ["Copy-Move", "Splicing"],
        "year": 2013,
        "kaggle_slug": "sophatvathana/casia-dataset",
        "kaggle_url": "https://www.kaggle.com/datasets/sophatvathana/casia-dataset",
        "usage": "Primary training dataset",
    },
    "IMD2020": {
        "description": "IMD2020 Large-Scale Annotated Dataset for Manipulated Images (~2010 images)",
        "forgery_types": ["Inpainting", "Removal", "Mixed"],
        "year": 2020,
        "url": "https://cmp.felk.cvut.cz/~novozdarek/imd2020/",
        "usage": "Training for inpainting and modern edits",
    },
    "Columbia": {
        "description": "Columbia Uncompressed Image Splicing Detection Dataset (363 images)",
        "forgery_types": ["Splicing"],
        "year": 2004,
        "url": "https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/",
        "note": "Requires academic registration form",
        "usage": "Cross-dataset evaluation (high-quality images)",
    },
    "COVERAGE": {
        "description": "COVERAGE Copy-Move Forgery Database (200 images)",
        "forgery_types": ["Copy-Move"],
        "year": 2018,
        "github_url": "https://github.com/wenbihan/coverage",
        "usage": "Cross-dataset evaluation (realistic scenarios)",
    },
}


def print_dataset_info():
    """Print a summary of all datasets and their online sources."""
    print("=" * 65)
    print("  Dataset References — Three-Stream ViT")
    print("=" * 65)

    for name, info in DATASETS.items():
        print(f"\n  [{name}]")
        print(f"    {info['description']}")
        print(f"    Forgery types : {', '.join(info['forgery_types'])}")
        print(f"    Purpose       : {info['usage']}")
        if "kaggle_url" in info:
            print(f"    Kaggle        : {info['kaggle_url']}")
        if "url" in info:
            print(f"    URL           : {info['url']}")
        if "github_url" in info:
            print(f"    GitHub        : {info['github_url']}")
        if "note" in info:
            print(f"    ⚠ Note        : {info['note']}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    print_dataset_info()
