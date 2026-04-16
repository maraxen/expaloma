"""Pytest configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def model_eqx_path(repo_root: Path) -> Path:
    """Path to bundled v0.0.8 ``.eqx`` in the source tree."""
    p = repo_root / "src" / "expaloma" / "weights" / "espaloma_v0_0_8.eqx"
    if not p.is_file():
        pytest.skip(f"Bundled weights missing: {p}")
    return p
