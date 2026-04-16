"""Numerical parity: JAX vs PyTorch reference forward (no DGL runtime)."""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

# Curated SMILES for JAX vs torch reference (requires ``/tmp/espaloma_model.pt``).
PARITY_SMILES = (
    "CCO",
    "c1ccccc1",
    "CC(=O)[O-]",
    "CC(=O)Oc1ccccc1C(=O)O",
)


def test_jax_matches_torch_reference(model_eqx_path: Path):
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(ROOT / "tests" / "stubs"))
    from rdkit import Chem

    from expaloma.featurize import from_rdkit_mol, graph_to_jax
    from expaloma.infer import charges_for_smiles
    from expaloma.nn.model import load_eqx
    sys.path.insert(0, str(ROOT / "tests"))
    from torch_reference_forward import load_torch_model, torch_forward

    pt = Path("/tmp/espaloma_model.pt")
    if not pt.is_file():
        pytest.skip("Need /tmp/espaloma_model.pt")

    torch.set_grad_enabled(False)
    m = load_torch_model(str(pt))
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    gf = from_rdkit_mol(mol)
    x, send, recv, qref = graph_to_jax(gf)
    h0t = torch.tensor(np.array(x))
    st = torch.tensor(np.array(send), dtype=torch.long)
    rt = torch.tensor(np.array(recv), dtype=torch.long)
    qreft = torch.tensor(np.array(qref), dtype=torch.float32)
    tc = float(Chem.GetFormalCharge(mol))
    q_pt = torch_forward(m, h0t, st, rt, tc, qreft).numpy()

    jax_m = load_eqx(model_eqx_path)
    n = x.shape[0]
    seg = jnp.zeros((n,), dtype=jnp.int32)
    q_jx = np.array(
        jax_m(jnp.asarray(x), jnp.asarray(send), jnp.asarray(recv), seg, 1, tc, q_ref=qref)
    ).flatten()

    assert np.max(np.abs(q_pt - q_jx)) < 1e-5

    q_api = charges_for_smiles("CC(=O)Oc1ccccc1C(=O)O", model_eqx_path)
    assert np.max(np.abs(q_api - q_jx)) < 1e-6

    q_default = charges_for_smiles("CC(=O)Oc1ccccc1C(=O)O")
    assert np.max(np.abs(q_default - q_jx)) < 1e-6


@pytest.mark.parametrize("smiles", PARITY_SMILES)
def test_jax_matches_torch_parametric(smiles: str, model_eqx_path: Path):
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(ROOT / "tests" / "stubs"))
    from rdkit import Chem

    from expaloma.featurize import from_rdkit_mol, graph_to_jax
    from expaloma.nn.model import load_eqx
    sys.path.insert(0, str(ROOT / "tests"))
    from torch_reference_forward import load_torch_model, torch_forward

    pt = Path("/tmp/espaloma_model.pt")
    if not pt.is_file():
        pytest.skip("Need /tmp/espaloma_model.pt")

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    torch.set_grad_enabled(False)
    m = load_torch_model(str(pt))
    gf = from_rdkit_mol(mol)
    x, send, recv, qref = graph_to_jax(gf)
    h0t = torch.tensor(np.array(x))
    st = torch.tensor(np.array(send), dtype=torch.long)
    rt = torch.tensor(np.array(recv), dtype=torch.long)
    qreft = torch.tensor(np.array(qref), dtype=torch.float32)
    tc = float(Chem.GetFormalCharge(mol))
    q_pt = torch_forward(m, h0t, st, rt, tc, qreft).numpy()

    jax_m = load_eqx(model_eqx_path)
    n = x.shape[0]
    seg = jnp.zeros((n,), dtype=jnp.int32)
    q_jx = np.array(
        jax_m(jnp.asarray(x), jnp.asarray(send), jnp.asarray(recv), seg, 1, tc, q_ref=qref)
    ).flatten()

    assert np.max(np.abs(q_pt - q_jx)) < 1e-5


def test_golden_aspirin():
    from expaloma.infer import charges_for_smiles

    golden = ROOT / "tests" / "data" / "espaloma_golden" / "aspirin_charges.npy"
    if not golden.is_file():
        pytest.skip("Golden file missing")
    expected = np.load(golden)
    got = charges_for_smiles("CC(=O)Oc1ccccc1C(=O)O")
    assert got.shape == expected.shape
    assert np.max(np.abs(got - expected)) < 1e-5
