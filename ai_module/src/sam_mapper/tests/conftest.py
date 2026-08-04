"""Make the package importable when pytest runs from the repo root.

sam_mapper is an ament_python package; outside a sourced colcon install the
package dir is not on sys.path, so every test used to repeat this insert.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
