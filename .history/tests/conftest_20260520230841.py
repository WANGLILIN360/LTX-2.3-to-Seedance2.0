"""Shared pytest fixtures — mock heavy dependencies before import."""

import sys
from pathlib import Path

import pytest


# Ensure ltx_core is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "ltx-core" / "src"))

# Ensure ltx_pipelines is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "ltx-pipelines" / "src"))


@pytest.fixture(autouse=True, scope="session")
def _stub_missing_deps():
    """Stub out missing third-party packages that block import.

    ltx_pipelines has deep dependency chains (av, OpenImageIO, etc.)
    that are not needed for unit-testing core logic. We pre-register
    stub modules so the imports succeed without installing them.
    """
    stubs = [
        "av",
        "av.codec",
        "av.container",
        "av.stream",
        "av.video",
        "av.audio",
        "av.filter",
        "av.data",
        "OpenImageIO",
        "OpenImageIO.ImageSpec",
        "OpenImageIO.ImageInput",
    ]
    for name in stubs:
        if name not in sys.modules:
            sys.modules[name] = type(sys)(name)
