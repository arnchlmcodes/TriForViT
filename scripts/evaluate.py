"""
scripts/evaluate.py — CLI entrypoint for FTT v2 evaluation.
Delegates to src.ftt_v2.evaluate.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.ftt_v2.evaluate import main

if __name__ == "__main__":
    main()
