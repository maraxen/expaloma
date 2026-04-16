"""High-level inference API."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from rdkit import Chem

from expaloma.featurize import from_rdkit_mol, graph_to_jax
from expaloma.nn.model import EspalomaModel, load_eqx


def charges_for_smiles(
    smiles: str,
    weights: Path | str | None = None,
    *,
    total_charge: float | None = None,
) -> np.ndarray:
    """
    ESPALOMA partial charges for a SMILES string (implicit hydrogens per RDKit).

    ``weights`` defaults to the bundled v0.0.8 ``.eqx`` (see ``expaloma.paths.bundled_eqx_path``).

    ``total_charge`` defaults to RDKit formal charge (``espaloma_charge.app.charge`` parity).
    """
    if weights is None:
        from expaloma.paths import bundled_eqx_path

        weights = bundled_eqx_path()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    if total_charge is None:
        total_charge = float(Chem.GetFormalCharge(mol))

    gf = from_rdkit_mol(mol)
    x, send, recv, qref = graph_to_jax(gf)
    n = x.shape[0]
    seg = jnp.zeros((n,), dtype=jnp.int32)

    model: EspalomaModel = load_eqx(weights)
    q = model(x, send, recv, seg, 1, total_charge, q_ref=qref)
    return np.asarray(q, dtype=np.float64).reshape(-1)
