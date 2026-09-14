"""
Ensures the repo root is on sys.path so `from src.xxx import yyy` works
when running `pytest` from the repository root, without needing an
editable install.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
