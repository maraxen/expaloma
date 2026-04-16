"""Bucketed padding for fixed-shape JIT (optional batched inference)."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

DEFAULT_BUCKETS: tuple[int, ...] = (64, 128, 256, 512)


def pick_bucket(n_atoms: int, buckets: tuple[int, ...] = DEFAULT_BUCKETS) -> int:
    """Smallest bucket size ``>= n_atoms``."""
    for b in buckets:
        if n_atoms <= b:
            return b
    return buckets[-1]


@dataclass(frozen=True)
class PaddedGraph:
    """Single graph padded to a static bucket."""

    h0: jnp.ndarray
    """(bucket, F)"""
    senders: jnp.ndarray
    receivers: jnp.ndarray
    """Edge indices in padded node space."""
    atom_mask: jnp.ndarray
    """(bucket,) bool — True for real atoms."""
    bucket: int
    n_atoms: int


def pad_graph_features(
    h0: np.ndarray,
    senders: np.ndarray,
    receivers: np.ndarray,
    *,
    bucket: int,
) -> PaddedGraph:
    """Pad node features to ``bucket``; keep edges among real atoms only."""
    n_atoms, f = h0.shape
    if n_atoms > bucket:
        raise ValueError(f"n_atoms={n_atoms} exceeds bucket={bucket}")
    pad_h = np.zeros((bucket, f), dtype=np.float32)
    pad_h[:n_atoms] = h0
    mask = np.zeros((bucket,), dtype=np.bool_)
    mask[:n_atoms] = True

    send = np.asarray(senders, dtype=np.int32)
    recv = np.asarray(receivers, dtype=np.int32)

    return PaddedGraph(
        h0=jnp.asarray(pad_h),
        senders=jnp.asarray(send),
        receivers=jnp.asarray(recv),
        atom_mask=jnp.asarray(mask),
        bucket=bucket,
        n_atoms=n_atoms,
    )


def charges_trimmed(full_q: jnp.ndarray, n_atoms: int) -> jnp.ndarray:
    """Strip padding from charge vector."""
    return full_q[:n_atoms]
