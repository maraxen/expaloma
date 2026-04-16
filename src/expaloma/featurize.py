"""RDKit molecule → node features and edge list (``espaloma_charge.utils`` parity)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem

SUPPORTED_ELEMENTS = [
    "H",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
]

HYBRIDIZATION_RDKIT: dict = {}


def _hybridization_one_hot(hyb: Chem.rdchem.HybridizationType) -> np.ndarray:
    if not HYBRIDIZATION_RDKIT:
        HYBRIDIZATION_RDKIT.update(
            {
                Chem.rdchem.HybridizationType.SP: np.array([1, 0, 0, 0, 0], dtype=np.float64),
                Chem.rdchem.HybridizationType.SP2: np.array([0, 1, 0, 0, 0], dtype=np.float64),
                Chem.rdchem.HybridizationType.SP3: np.array([0, 0, 1, 0, 0], dtype=np.float64),
                Chem.rdchem.HybridizationType.SP3D: np.array([0, 0, 0, 1, 0], dtype=np.float64),
                Chem.rdchem.HybridizationType.SP3D2: np.array([0, 0, 0, 0, 1], dtype=np.float64),
                Chem.rdchem.HybridizationType.S: np.array([0, 0, 0, 0, 0], dtype=np.float64),
                Chem.rdchem.HybridizationType.UNSPECIFIED: np.array([0, 0, 0, 0, 0], dtype=np.float64),
            }
        )
    return HYBRIDIZATION_RDKIT[hyb].astype(np.float32)


def fp_rdkit(atom: Chem.Atom) -> np.ndarray:
    element = atom.GetSymbol()
    if element not in SUPPORTED_ELEMENTS:
        raise ValueError(f"Element {element} is not supported.")
    scalars = np.array(
        [
            atom.GetTotalDegree(),
            atom.GetTotalValence(),
            atom.GetExplicitValence(),
            atom.GetIsAromatic() * 1.0,
            atom.GetMass(),
            atom.IsInRingSize(3) * 1.0,
            atom.IsInRingSize(4) * 1.0,
            atom.IsInRingSize(5) * 1.0,
            atom.IsInRingSize(6) * 1.0,
            atom.IsInRingSize(7) * 1.0,
            atom.IsInRingSize(8) * 1.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([scalars, _hybridization_one_hot(atom.GetHybridization())], axis=0)


@dataclass(frozen=True)
class GraphFeatures:
    """Single molecule graph."""

    h0: np.ndarray
    """Shape ``(n_atoms, 116)`` float32."""
    senders: np.ndarray
    receivers: np.ndarray
    """Directed edges (bidirectional bonds), int32 indices."""
    q_ref: np.ndarray
    """Shape ``(n_atoms, 1)`` formal charges per atom."""
    atomic_num: np.ndarray


def from_rdkit_mol(mol: Chem.Mol, *, use_fp: bool = True) -> GraphFeatures:
    """Mirror ``espaloma_charge.utils.from_rdkit_mol`` (atom order = ``mol.GetAtoms()``)."""
    n_atoms = mol.GetNumAtoms()
    types = np.array([[atom.GetAtomicNum()] for atom in mol.GetAtoms()], dtype=np.float64)
    q_ref = np.array([[atom.GetFormalCharge()] for atom in mol.GetAtoms()], dtype=np.float64)
    h_v = np.zeros((n_atoms, 100), dtype=np.float32)
    h_v[np.arange(n_atoms), np.squeeze(types).astype(np.int64)] = 1.0
    if use_fp:
        h_v_fp = np.stack([fp_rdkit(atom) for atom in mol.GetAtoms()], axis=0)
        h_v = np.concatenate([h_v, h_v_fp], axis=-1)

    bonds = list(mol.GetBonds())
    bonds_begin = [bond.GetBeginAtomIdx() for bond in bonds]
    bonds_end = [bond.GetEndAtomIdx() for bond in bonds]
    senders = np.array(bonds_begin + bonds_end, dtype=np.int32)
    receivers = np.array(bonds_end + bonds_begin, dtype=np.int32)

    return GraphFeatures(
        h0=np.asarray(h_v, dtype=np.float32),
        senders=senders,
        receivers=receivers,
        q_ref=np.asarray(q_ref, dtype=np.float32),
        atomic_num=np.squeeze(types.astype(np.int32)),
    )


def graph_to_jax(g: GraphFeatures):
    """Convert arrays to JAX arrays for the model."""
    import jax.numpy as jnp

    return (
        jnp.asarray(g.h0),
        jnp.asarray(g.senders),
        jnp.asarray(g.receivers),
        jnp.asarray(g.q_ref),
    )
