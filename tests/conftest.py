"""Shared pytest fixtures — mock heavy dependencies before import."""

import sys
import types
from pathlib import Path

# Ensure ltx_core is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "ltx-core" / "src"))

# Ensure ltx_pipelines is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "ltx-pipelines" / "src"))

# Stub out missing third-party packages BEFORE any test collection.
# These must be registered at module level (not in a fixture) because
# Python imports ltx_pipelines during collection, and media_io.py has
# top-level `import av` / `import OpenImageIO` that would fail otherwise.
_STUBS = [
    "av", "av.codec", "av.container", "av.stream", "av.video",
    "av.audio", "av.filter", "av.data",
    "OpenImageIO", "OpenImageIO.ImageSpec", "OpenImageIO.ImageInput",
]
for _name in _STUBS:
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)
