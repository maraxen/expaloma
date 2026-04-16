# Expaloma: Native JAX/Equinox Port Handoff

Welcome to **Expaloma**, the standalone JAX/Equinox port of the espaloma-charge partial charge inference model.
This document contains the comprehensive technical handoff for finishing the development inside this `expaloma` repository.

## Overview
The goal is to eliminate the heavyweight dependencies of PyTorch and DGL within the downstream `prolix/proxide` physics simulator. JAX will cleanly JIT-compile the structure processing and neural inference.

## Technical Scope

### 1. The GNN Architecture (Equinox)
The original PyTorch `espaloma_charge` model architecture uses staggered sequence of Linear, GraphConv (Legacy DGL), Tanh, and BatchNormalization.

**Expected File: `src/expaloma/nn/layers.py`**
- Implement an Equinox module mapping to `dgl.nn.GraphConv` / `SAGEConv`.
- Equation for the simple GNN message passing on the homogenous graph representation. Use `jax.lax.scan` for larger iterative updates if needed, though statically unrolled `for` loops across the `n_layers = 2 or 3` might be enough.
- You will be passing node features $X$ and adjacency matrices $A$.
- For batched and padded inputs, $A$ will need to be masked appropriately.

### 2. The Analytical QEq Solver
When predicting charges, `espaloma_charge` predicts an electronegativity $e$ and hardness $s$ per atom, then uses a Lagrange Multiplier to analytically compute partial charges $q$ that sum exactly to the `total_charge` of the molecule.

**Expected File: `src/expaloma/nn/qeq.py`**
- Port `espaloma_charge.models.ChargeEquilibrium.forward` to `jax.numpy`.
- Compute $s^{-1}$ and $es^{-1}$. Use `jax.ops.segment_sum` to compute values per-molecule across batches! This is how you handle the message reduction that `dgl`'s `sum_nodes` previously did.
- QEq Equation:
  $$ q_i^* = -e_i s_i^{-1} + s_i^{-1} \frac{Q + \sum_i e_i s_i^{-1}}{\sum_j s_j^{-1}} $$

### 3. Weight Extraction & Conversion Script
The PyTorch `espaloma_charge` weights are provided in a `.pt` file generated from a DGL Model. You must map these dictionary entries to the JAX/Equinox tree logic.

**Expected File: `scripts/convert_weights.py`**
- Read `.espaloma_charge_model.pt` via PyTorch (as an offline asset).
- Translate dictionary keys (e.g., `_sequential.f_in.0.weight` which corresponds to Linear to the corresponding Equinox `nn.Linear` properties).
- Translate `dgl.nn.GraphConv` layer parameters which might use `weight` (for feature transformation) and `bias`.
- Save as `weights.eqx` or `weights.npz` using Equinox's serialisation.

### 4. Bucketed Padding for `vmap`/`jit` Performance
Since JAX requires static shapes for XLA JIT compilation, you cannot directly pass variable-sized graphs.

**Expected File: `src/expaloma/padding.py`**
- Implement bucketing strategies: e.g. pad atoms to `N=64, 128, 256, 512`.
- Apply a boolean atom-mask indicating whether an atom is "real" or "padding".
- Ensure your `segment_sum` and message passing routines multiply by the mask.

### 5. AtomMapNum Index Safety
The Proxide wrapper passes RDKit molecules. Original atom indices must be preserved exactly in the final array.
In your data loading step `from_rdkit_mol`: Ensure the output JAX structures maintain the original 0-indexed RDKit Atom Order. `proxide` will further verify with `AtomMapNum`.

### 6. Testing, CI, and Equivalency Check
It is crucial that the JAX port exactly replicates the PyTorch outputs to within floating-point precision, while building a robust foundation of unit tests.
- **Reference Submodule:** The original repository has been added as a git submodule in `references/espaloma-charge`, pinned to commit `21a0182f9619c5f5e3130a66a67c338b87fd6c87`.
- **Equivalency Tests**: You must write CI tests in `tests/test_equivalency.py` that load SMILES strings, run them through the PyTorch module imported directly from `references/espaloma-charge`, and compare the numerical exactness of the output to the new Equinox modules. Add GitHub Actions or integration scripts to automate this parity check.
- **Unit Tests**: Add targeted native unit tests for the JAX layers (especially the padding/bucketing logic and QEq solver) without relying on PyTorch at all.

### 7. The Typer CLI
The scaffold now includes a basic `cli.py`!
- The CLI provides the `infer` and `convert-weights` endpoints.
- Plumb these commands to use the actual `expaloma.nn.model` logic and the weight conversion workflow.

## Acceptance Criteria
- [ ] `expaloma convert-weights model.pt out.eqx` outputs properly structured Equinox state.
- [ ] `expaloma infer "CC(=O)Oc1ccccc1C(=O)O"` produces identical partial charges (within `1e-5` float tolerance) as `espaloma_charge` (regression tests).
- [ ] PyTorch, DGL are NOT imported inside `src/expaloma/*` (only allowed in scripts/convert_weights).
