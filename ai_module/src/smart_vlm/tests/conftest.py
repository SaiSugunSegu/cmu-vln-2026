"""Make smart_vlm and its captioner dependency importable outside a colcon install.

smart_vlm.category1_utils re-exports from captioner.text_utils (both stdlib-only),
so the sibling package has to be on sys.path for the pure-python tests too.
"""
import os
import sys

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "captioner"))
