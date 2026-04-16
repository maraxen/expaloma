"""GraphSAGE (mean) and input stack matching DGL `dgl.nn.SAGEConv(aggregator_type='mean')`."""

from __future__ import annotations

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp


def _mean_aggregate_messages(
    msg: jnp.ndarray,
    receivers: jnp.ndarray,
    n_nodes: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sum messages into dst nodes and count in-degrees (for mean)."""
    dim = msg.shape[1]
    sum_m = jnp.zeros((n_nodes, dim), dtype=msg.dtype)
    sum_m = sum_m.at[receivers].add(msg)
    deg = jnp.zeros((n_nodes,), dtype=msg.dtype)
    deg = deg.at[receivers].add(1.0)
    return sum_m, deg


def sage_mean_forward(
    x: jnp.ndarray,
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    *,
    w_self: jnp.ndarray,
    b_self: jnp.ndarray,
    w_neigh: jnp.ndarray,
    zero_edges: bool,
) -> jnp.ndarray:
    """
    One SAGEConv mean layer matching DGL when `fc_neigh` has no bias.

    `lin_before_mp = (in_feats > out_feats)` uses neighbor transform before
    aggregation; otherwise transform after mean (DGL ``SAGEConv`` source).
    """
    in_feats = x.shape[1]
    out_feats = w_self.shape[0]
    lin_before_mp = in_feats > out_feats

    if zero_edges:
        # DGL: ``neigh`` is zeros before ``fc_neigh`` when ``not lin_before_mp``; with no
        # bias, ``fc_neigh(zeros) == 0``. For ``lin_before_mp``, message pipeline yields zeros.
        neigh_mean = jnp.zeros((x.shape[0], out_feats), dtype=x.dtype)
    elif lin_before_mp:
        src_feat = x @ w_neigh.T
        msg = src_feat[senders]
        sum_m, deg = _mean_aggregate_messages(msg, receivers, x.shape[0])
        neigh_mean = jnp.where(deg[:, None] > 0, sum_m / deg[:, None], 0.0)
    else:
        msg = x[senders]
        sum_m, deg = _mean_aggregate_messages(msg, receivers, x.shape[0])
        agg = jnp.where(deg[:, None] > 0, sum_m / deg[:, None], 0.0)
        neigh_mean = agg @ w_neigh.T

    h_self = x @ w_self.T + b_self
    return h_self + neigh_mean


class SAGEConvMean(eqx.Module):
    """Learnable SAGE mean layer (same interface as a single DGL ``SAGEConv``)."""

    w_self: jnp.ndarray
    b_self: jnp.ndarray
    w_neigh: jnp.ndarray

    def __call__(
        self,
        x: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        *,
        zero_edges: bool,
    ) -> jnp.ndarray:
        return sage_mean_forward(
            x,
            senders,
            receivers,
            w_self=self.w_self,
            b_self=self.b_self,
            w_neigh=self.w_neigh,
            zero_edges=zero_edges,
        )


class EspalomaGNN(eqx.Module):
    """``Sequential`` from espaloma_charge: Linear+Tanh, then SAGE+ReLU x4."""

    w_f_in: jnp.ndarray
    b_f_in: jnp.ndarray
    sages: Tuple[SAGEConvMean, ...]

    def __call__(
        self,
        x: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        *,
        zero_edges: bool,
        atom_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        m = atom_mask
        if m is not None:
            m = m.astype(x.dtype)[:, None]
            pre = (x @ self.w_f_in.T + self.b_f_in) * m
            h = jnp.tanh(pre)
        else:
            h = jnp.tanh(x @ self.w_f_in.T + self.b_f_in)
        for i, sage in enumerate(self.sages):
            h = sage(h, senders, receivers, zero_edges=zero_edges)
            h = jax.nn.relu(h)
            if m is not None:
                h = h * m
        return h


