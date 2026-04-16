"""Padding helpers."""

import numpy as np
from rdkit import Chem

from expaloma.featurize import from_rdkit_mol, graph_to_jax
from expaloma.nn.model import load_eqx
from expaloma.padding import charges_trimmed, pad_graph_features, pick_bucket

import jax.numpy as jnp


def test_pick_bucket():
    assert pick_bucket(10) == 64
    assert pick_bucket(100) == 128


def test_pad_and_trim():
    h0 = np.random.randn(5, 116).astype(np.float32)
    send = np.array([0, 1], dtype=np.int32)
    recv = np.array([1, 0], dtype=np.int32)
    g = pad_graph_features(h0, send, recv, bucket=64)
    assert g.h0.shape == (64, 116)
    q = np.random.randn(64, 1).astype(np.float32)
    t = charges_trimmed(q, 5)
    assert t.shape == (5, 1)


def test_full_model_padded_matches_unpadded(model_eqx_path):
    """``EspalomaModel`` with mask + bucket matches the unbatched single-graph path."""
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    gf = from_rdkit_mol(mol)
    x, send, recv, qref = graph_to_jax(gf)
    n = int(x.shape[0])
    tc = float(Chem.GetFormalCharge(mol))
    seg = jnp.zeros((n,), dtype=jnp.int32)

    model = load_eqx(model_eqx_path)
    q_un = np.asarray(
        model(jnp.asarray(x), jnp.asarray(send), jnp.asarray(recv), seg, 1, tc, q_ref=qref)
    ).reshape(-1)

    bucket = pick_bucket(n)
    h0_np = np.asarray(x, dtype=np.float32)
    pg = pad_graph_features(h0_np, np.asarray(send), np.asarray(recv), bucket=bucket)
    qref_pad = np.zeros((bucket, 1), dtype=np.float32)
    qref_pad[:n, 0] = np.asarray(qref)[:, 0]
    seg_pad = jnp.zeros((bucket,), dtype=jnp.int32)
    q_pad = model(
        pg.h0,
        pg.senders,
        pg.receivers,
        seg_pad,
        1,
        tc,
        q_ref=jnp.asarray(qref_pad),
        atom_mask=pg.atom_mask,
        n_atoms_real=n,
    )
    q_trim = np.asarray(charges_trimmed(q_pad, n)).reshape(-1)

    assert np.max(np.abs(q_un - q_trim)) < 1e-5
