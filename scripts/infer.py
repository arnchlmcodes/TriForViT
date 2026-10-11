"""
scripts/infer.py — CLI entrypoint for FTT v2 inference.
Delegates to src.ftt_v2.infer.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.ftt_v2.infer import main

if __name__ == "__main__":
    main()
