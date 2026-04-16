# expaloma

Native [JAX](https://github.com/jax-ml/jax)/[Equinox](https://github.com/patrick-kidger/equinox) partial-charge inference, ported from [espaloma_charge](https://github.com/choderalab/espaloma_charge) so downstream simulators can avoid PyTorch + DGL at runtime.

## Install

```bash
uv sync
# optional dev (pytest)
uv sync --extra dev
```

### Typer CLI

The package exposes a **`typer`** application ([`src/expaloma/cli.py`](src/expaloma/cli.py)) registered as the **`expaloma`** console script in [`pyproject.toml`](pyproject.toml) (`[project.scripts]`). Commands:

- **`expaloma infer SMILES`** — partial charges (bundled `.eqx` by default).
- **`expaloma convert-weights MODEL.pt OUT.eqx`** — offline conversion (needs a **git checkout** with `scripts/` and submodules; not available from a PyPI-only install).
- **`expaloma --version`** — print the installed package version.

### PyPI and `uv tool`

After you [publish](#ci-and-releases) to PyPI:

```bash
pip install expaloma
expaloma infer "CCO"
```

Install as a **uv tool** (isolated env with the `expaloma` console script on `PATH`):

```bash
uv tool install expaloma
expaloma --version
# one-shot without installing:
uvx --from expaloma expaloma infer "CCO"
```

### CI and releases

| Workflow | Purpose |
|----------|---------|
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | Lint/tests on push/PR to `main` |
| [`.github/workflows/publish.yml`](.github/workflows/publish.yml) | Build with `uv build` and upload to **PyPI** on **GitHub Release** (OIDC trusted publishing) |

Configure **trusted publishing** on PyPI for this repo/workflow, and add a GitHub **Environment** named `pypi` if you use environment protection rules. Tag releases (e.g. `v0.1.0`) and publish a **GitHub Release** to trigger the workflow (or run it manually via **workflow_dispatch**).

### Branch protection (`gh` CLI)

GitHub’s **rulesets** API is the supported approach. After CI is green, you can require the **`test`** check on `main` via the web UI (**Settings → Rules → Rulesets**), or use the API. Examples:

```bash
# Repo visibility / basics (requires gh auth)
gh repo view maraxen/expaloma

# Many teams configure rulesets in the UI so required checks match exactly (e.g. "test" from ci.yml).
# To require pull requests before merging, use Settings → Rulesets → Add rule → Target: main.
```

For automation-heavy setups, create a ruleset with the REST API (`gh api repos/{owner}/{repo}/rulesets`) using JSON from **Settings → Rulesets → View JSON**, or start from [GitHub’s ruleset documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository). The **`gh`** CLI does not yet offer a single stable `gh ruleset create` for all options; the UI or API remains the reliable path.

## Inference (bundled weights)

The public **v0.0.8** checkpoint is shipped as Equinox bytes at `src/expaloma/weights/espaloma_v0_0_8.eqx` (also included in wheels). Provenance and SHA256 hashes are in [`src/expaloma/weights/README.md`](src/expaloma/weights/README.md).

```bash
uv run expaloma infer "CC(=O)Oc1ccccc1C(=O)O"
# or pass a custom checkpoint:
uv run expaloma infer "CCO" --weights path/to/custom.eqx
```

From Python:

```python
from expaloma.infer import charges_for_rdkit_mol, charges_for_smiles
from rdkit import Chem

q = charges_for_smiles("CCO")  # uses bundled .eqx by default
q = charges_for_rdkit_mol(Chem.MolFromSmiles("CCO"))  # same atom order as RDKit
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
