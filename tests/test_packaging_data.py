"""Bundled weights are present in the package (source tree and wheel)."""

from __future__ import annotations

from importlib.resources import files

from expaloma.paths import bundled_eqx_path


def test_bundled_eqx_exists_on_disk():
    p = bundled_eqx_path()
    assert p.is_file(), f"missing {p}"
    assert p.stat().st_size > 0


def test_bundled_eqx_visible_via_importlib_resources():
    ref = files("expaloma") / "weights" / "espaloma_v0_0_8.eqx"
    assert ref.is_file()
