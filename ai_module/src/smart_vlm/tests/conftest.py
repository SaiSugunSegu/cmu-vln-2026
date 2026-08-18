"""Make smart_vlm and its captioner / language_planner deps importable outside colcon.

smart_vlm.numerical_utils re-exports from captioner.text_utils and pulls the extract
system prompt from language_planner.prompts, so both sibling packages have to be on
sys.path for the pure-python tests too.
"""
import os
import sys

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "captioner"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "language_planner"))
