from __future__ import annotations

import importlib
from typing import Any


def import_from_path(path: str) -> Any:
    """Import a symbol from a string like 'module.sub:ClassName'."""
    if ":" not in path:
        raise ValueError(f"Invalid import path '{path}'. Expected 'module:Symbol'.")
    mod, sym = path.split(":", 1)
    module = importlib.import_module(mod)
    return getattr(module, sym)
