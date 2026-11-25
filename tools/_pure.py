"""Import RoomHeat's pure-Python modules without Home Assistant.

Importing `custom_components.roomheat` would run its __init__ (which imports HA).
Registering the directory as a bare package skips that, while relative imports
between the pure modules (sampling, thermal, dataset, ...) still work.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "custom_components" / "roomheat"
_NAME = "roomheat_pure"


def load(module: str):
    if _NAME not in sys.modules:
        pkg = types.ModuleType(_NAME)
        pkg.__path__ = [str(PACKAGE_DIR)]
        sys.modules[_NAME] = pkg
    return importlib.import_module(f"{_NAME}.{module}")
