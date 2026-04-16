"""Filesystem paths to package data (bundled weights)."""

from __future__ import annotations

from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent


def bundled_eqx_path() -> Path:
    """Path to the shipped v0.0.8 Equinox checkpoint (``espaloma_v0_0_8.eqx``)."""
    return _PKG_ROOT / "weights" / "espaloma_v0_0_8.eqx"
