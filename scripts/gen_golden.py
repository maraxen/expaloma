#!/usr/bin/env python3
"""Generate golden charge vectors for regression tests (JAX path)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-eqx",
        type=Path,
        default=None,
        help="Converted .eqx weights (default: bundled src/expaloma/weights/espaloma_v0_0_8.eqx)",
    )
    parser.add_argument(
        "--smiles",
        type=str,
        default="CC(=O)Oc1ccccc1C(=O)O",
        help="SMILES for golden output",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tests/data/espaloma_golden/aspirin_charges.npy"),
        help="Output .npy path",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(repo / "src"))

    from expaloma.infer import charges_for_smiles

    model_eqx = args.model_eqx or (repo / "src" / "expaloma" / "weights" / "espaloma_v0_0_8.eqx")
    q = charges_for_smiles(args.smiles, model_eqx)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, q)
    print(f"Wrote {args.out} shape={q.shape}")


if __name__ == "__main__":
    main()
