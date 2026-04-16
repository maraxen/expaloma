# expaloma

Native [JAX](https://github.com/jax-ml/jax)/[Equinox](https://github.com/patrick-kidger/equinox) partial-charge inference, ported from [espaloma_charge](https://github.com/choderalab/espaloma_charge) so downstream simulators can avoid PyTorch + DGL at runtime.

## Install

```bash
uv sync
# optional dev (pytest)
uv sync --extra dev
```

## Inference (bundled weights)

The public **v0.0.8** checkpoint is shipped as Equinox bytes at `src/expaloma/weights/espaloma_v0_0_8.eqx` (also included in wheels). Provenance and SHA256 hashes are in [`src/expaloma/weights/README.md`](src/expaloma/weights/README.md).

```bash
uv run expaloma infer "CC(=O)Oc1ccccc1C(=O)O"
# or pass a custom checkpoint:
uv run expaloma infer "CCO" --weights path/to/custom.eqx
```

From Python:

```python
from expaloma.infer import charges_for_smiles

q = charges_for_smiles("CCO")  # uses bundled .eqx by default
```

## Convert PyTorch `model.pt` to `.eqx` (offline)

Conversion uses PyTorch only in the script environment (not a runtime dependency of `expaloma`):

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv run python scripts/convert_weights.py path/to/model.pt out.eqx
```

## Tests

```bash
uv sync --extra dev
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
curl -fsSL -o /tmp/espaloma_model.pt \
  "https://github.com/choderalab/espaloma_charge/releases/download/v0.0.8/model.pt"
uv run pytest tests/ -v
```

CI downloads the same `model.pt` for **JAX vs PyTorch reference** parity checks; golden vectors and bundled `.eqx` cover regression without relying on that download for basic JAX tests.

## Benchmark (local)

[`scripts/benchmark_infer.py`](scripts/benchmark_infer.py) runs warmup + timed repeats with `jax.block_until_ready`. Timing is environment-dependent and **not** part of default CI.

```bash
uv run python scripts/benchmark_infer.py --repeats 50
```

## Attribution

The original [espaloma_charge](https://github.com/choderalab/espaloma_charge) project is MIT-licensed (see `references/espaloma-charge/LICENSE`). This port bundles weights derived from the published **v0.0.8** release; retain upstream notices in distributions.
