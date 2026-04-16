#!/usr/bin/env python3
"""
Time charge inference after JIT warmup (not run in default CI — noisy on shared runners).

Uses ``jax.block_until_ready`` on the model output before timing stops, so XLA work is complete.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from rdkit import Chem


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Expaloma charge inference (JAX)")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Equinox .eqx path (default: bundled v0.0.8)",
    )
    parser.add_argument(
        "--smiles",
        type=str,
        default="CC(=O)Oc1ccccc1C(=O)O",
        help="SMILES string to benchmark",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs (JIT compile)")
    parser.add_argument("--repeats", type=int, default=20, help="Timed runs after warmup")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))

    from expaloma.featurize import from_rdkit_mol, graph_to_jax
    from expaloma.nn.model import load_eqx
    from expaloma.paths import bundled_eqx_path

    weights = args.weights if args.weights is not None else bundled_eqx_path()
    smiles = args.smiles

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise SystemExit(f"Invalid SMILES: {smiles!r}")
    tc = float(Chem.GetFormalCharge(mol))
    gf = from_rdkit_mol(mol)
    x, send, recv, qref = graph_to_jax(gf)
    n = x.shape[0]
    seg = jnp.zeros((n,), dtype=jnp.int32)

    model = load_eqx(weights)

    def run_once() -> None:
        q = model(x, send, recv, seg, 1, tc, q_ref=qref)
        jax.block_until_ready(q)

    for _ in range(args.warmup):
        run_once()

    times: list[float] = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        run_once()
        times.append(time.perf_counter() - t0)

    p50 = statistics.median(times)
    p95 = statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times)
    print(f"smiles={smiles!r} repeats={args.repeats}")
    print(f"jax={jax.__version__} jaxlib backend devices={jax.devices()}")
    print(f"time_s: median={p50:.6f} p95~={p95:.6f} min={min(times):.6f} max={max(times):.6f}")


if __name__ == "__main__":
    main()
