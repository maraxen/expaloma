#!/usr/bin/env python3
"""
Convert ``.espaloma_charge_model.pt`` (PyTorch + DGL) to Equinox leaf serialization.

Requires PyTorch in the environment (not a runtime dependency of ``expaloma``).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path


def _install_import_shims(repo_root: Path) -> None:
    stub = repo_root / "tests" / "stubs"
    if stub.is_dir() and str(stub) not in sys.path:
        sys.path.insert(0, str(stub))
    models_path = repo_root / "references" / "espaloma-charge" / "espaloma_charge" / "models.py"
    spec = importlib.util.spec_from_file_location("espaloma_charge.models", models_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    pkg = types.ModuleType("espaloma_charge")
    pkg.models = mod
    sys.modules["espaloma_charge"] = pkg
    sys.modules["espaloma_charge.models"] = mod


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PyTorch espaloma_charge weights to Equinox .eqx")
    parser.add_argument("pytorch_model", type=Path, help="Path to .espaloma_charge_model.pt")
    parser.add_argument("output", type=Path, help="Output path (.eqx)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    try:
        import torch
    except ImportError as e:
        raise SystemExit("PyTorch is required for this script. Install with: uv pip install torch") from e

    import equinox as eqx

    from expaloma.nn.model import EspalomaModel, model_from_torch_state_dict

    _install_import_shims(repo_root)

    m = torch.load(args.pytorch_model, map_location="cpu", weights_only=False)
    sd = m.state_dict()
    model = model_from_torch_state_dict({k: sd[k] for k in sd.keys()})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(str(args.output), model)
    print(f"Wrote Equinox weights to {args.output}")


if __name__ == "__main__":
    main()
