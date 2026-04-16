"""Full Espaloma charge model: GNN + readout + QEq."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp

from expaloma.nn.config import FEATURE_UNITS, INPUT_UNITS, N_SAGE_LAYERS, SAGE_HIDDEN
from expaloma.nn.layers import EspalomaGNN, SAGEConvMean
from expaloma.nn.qeq import charge_equilibrium


class EspalomaModel(eqx.Module):
    """End-to-end charge model (``torch.nn.Sequential`` parity: GNN, readout, equilibrium)."""

    gnn: EspalomaGNN
    w_readout: jnp.ndarray
    b_readout: jnp.ndarray

    def __call__(
        self,
        x: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        segment_ids: jnp.ndarray,
        num_graphs: int,
        total_charge: jnp.ndarray | float,
        q_ref: jnp.ndarray | None = None,
        atom_mask: jnp.ndarray | None = None,
        n_atoms_real: int | None = None,
    ) -> jnp.ndarray:
        zero_edges = senders.shape[0] == 0
        h = self.gnn(
            x,
            senders,
            receivers,
            zero_edges=zero_edges,
            atom_mask=atom_mask,
        )
        if n_atoms_real is not None:
            h = h[:n_atoms_real]
        es = h @ self.w_readout.T + self.b_readout
        e = es[:, :1]
        s = es[:, 1:2]
        sid = segment_ids if n_atoms_real is None else segment_ids[:n_atoms_real]
        qref = q_ref if (q_ref is None or n_atoms_real is None) else q_ref[:n_atoms_real]
        return charge_equilibrium(
            e,
            s,
            sid,
            num_graphs,
            total_charge,
            q_ref=qref,
        )


def template_model_v0_0_8() -> EspalomaModel:
    """Parameter-shaped skeleton matching the public v0.0.8 ``model.pt`` (four 128-d SAGE layers)."""

    def _sage() -> SAGEConvMean:
        return SAGEConvMean(
            w_self=jnp.zeros((SAGE_HIDDEN, SAGE_HIDDEN)),
            b_self=jnp.zeros((SAGE_HIDDEN,)),
            w_neigh=jnp.zeros((SAGE_HIDDEN, SAGE_HIDDEN)),
        )

    gnn = EspalomaGNN(
        w_f_in=jnp.zeros((INPUT_UNITS, FEATURE_UNITS)),
        b_f_in=jnp.zeros((INPUT_UNITS,)),
        sages=tuple(_sage() for _ in range(N_SAGE_LAYERS)),
    )
    return EspalomaModel(
        gnn=gnn,
        w_readout=jnp.zeros((2, SAGE_HIDDEN)),
        b_readout=jnp.zeros((2,)),
    )


def load_eqx(path: str | Path) -> EspalomaModel:
    """Load ``EspalomaModel`` leaves from ``eqx.tree_serialise_leaves`` output."""
    import equinox as eqx

    return eqx.tree_deserialise_leaves(str(path), template_model_v0_0_8())


def model_from_torch_state_dict(sd: dict[str, Any]) -> EspalomaModel:
    """Build an ``EspalomaModel`` from a PyTorch ``state_dict`` (CPU tensors or numpy)."""

    def _arr(k: str) -> jnp.ndarray:
        v = sd[k]
        if hasattr(v, "detach"):
            t = v.detach().cpu()
            try:
                return jnp.asarray(t.numpy())
            except (RuntimeError, TypeError):
                return jnp.asarray(t.tolist(), dtype=jnp.float32).reshape(tuple(t.shape))
        return jnp.asarray(v)

    w_f_in = _arr("0.f_in.0.weight")
    b_f_in = _arr("0.f_in.0.bias")
    sages: list[SAGEConvMean] = []
    for idx in (0, 2, 4, 6):
        p = f"0._sequential.d{idx}."
        sages.append(
            SAGEConvMean(
                w_self=_arr(p + "fc_self.weight"),
                b_self=_arr(p + "bias"),
                w_neigh=_arr(p + "fc_neigh.weight"),
            )
        )
    gnn = EspalomaGNN(w_f_in=w_f_in, b_f_in=b_f_in, sages=tuple(sages))
    return EspalomaModel(
        gnn=gnn,
        w_readout=_arr("1.fc_params.weight"),
        b_readout=_arr("1.fc_params.bias"),
    )
