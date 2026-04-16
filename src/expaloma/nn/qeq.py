"""Analytical charge equilibrium (QEq) matching ``espaloma_charge.models.ChargeEquilibrium``."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def charge_equilibrium(
    e: jnp.ndarray,
    s: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int,
    total_charge: jnp.ndarray | float,
    q_ref: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Compute partial charges ``q`` with graph-wise charge constraint.

    Parameters
    ----------
    e, s
        Shape ``(n_atoms, 1)``, same dtype (float32 recommended).
    segment_ids
        Integer array shape ``(n_atoms,)`` mapping each atom to molecule id ``0 .. num_segments-1``.
    num_segments
        Number of molecules in the batch.
    total_charge
        If ``q_ref`` is None: scalar or shape ``(num_segments, 1)`` total charge per molecule.
    q_ref
        Optional per-atom reference charges ``(n_atoms, 1)``; if given, per-graph totals are
        ``sum(q_ref)`` per segment (matches DGL when ``q_ref`` is in graph ndata).
    """
    s_inv = 1.0 / s
    e_s_inv = e * s_inv

    sid = segment_ids
    sum_s_inv = jax.ops.segment_sum(s_inv.squeeze(-1), sid, num_segments)
    sum_e_s_inv = jax.ops.segment_sum(e_s_inv.squeeze(-1), sid, num_segments)

    if q_ref is not None:
        tc = jax.ops.segment_sum(q_ref.squeeze(-1), sid, num_segments)
    else:
        tc = jnp.asarray(total_charge, dtype=e.dtype)
        if tc.ndim == 0:
            tc = jnp.full((num_segments,), tc)
        else:
            tc = jnp.reshape(tc, (-1,))
            if tc.shape[0] == 1 and num_segments > 1:
                tc = jnp.broadcast_to(tc, (num_segments,))

    sum_s_inv_n = sum_s_inv[sid][:, None]
    sum_e_s_inv_n = sum_e_s_inv[sid][:, None]
    tc_n = tc[sid][:, None]

    q = -e * s_inv + s_inv * (tc_n + sum_e_s_inv_n) / sum_s_inv_n
    return q
