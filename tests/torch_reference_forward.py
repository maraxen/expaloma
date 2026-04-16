"""
PyTorch forward matching ``espaloma_charge`` / DGL SAGE mean — no DGL import.

Used for parity tests against the JAX implementation.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


def _sage_mean(
    x: torch.Tensor,
    senders: torch.Tensor,
    receivers: torch.Tensor,
    fc_self: nn.Linear,
    fc_neigh: nn.Linear,
    layer_bias: torch.Tensor | None,
) -> torch.Tensor:
    in_f = x.shape[1]
    out_f = fc_self.out_features
    lin_before_mp = in_f > out_f
    n = x.shape[0]
    if senders.numel() == 0:
        neigh = torch.zeros(n, in_f if not lin_before_mp else out_f, device=x.device, dtype=x.dtype)
    elif lin_before_mp:
        src_feat = fc_neigh(x)
        msg = src_feat[senders]
        sum_m = torch.zeros(n, msg.shape[1], device=x.device, dtype=x.dtype)
        sum_m.index_add_(0, receivers, msg)
        deg = torch.zeros(n, device=x.device, dtype=x.dtype)
        deg.index_add_(0, receivers, torch.ones_like(receivers, dtype=x.dtype))
        d = deg.clamp_min(1.0e-30).unsqueeze(-1)
        neigh = sum_m / d
        neigh = torch.where(deg.unsqueeze(-1) > 0, neigh, torch.zeros_like(neigh))
    else:
        msg = x[senders]
        sum_m = torch.zeros(n, msg.shape[1], device=x.device, dtype=x.dtype)
        sum_m.index_add_(0, receivers, msg)
        deg = torch.zeros(n, device=x.device, dtype=x.dtype)
        deg.index_add_(0, receivers, torch.ones_like(receivers, dtype=x.dtype))
        d = deg.clamp_min(1.0e-30).unsqueeze(-1)
        agg = sum_m / d
        agg = torch.where(deg.unsqueeze(-1) > 0, agg, torch.zeros_like(agg))
        neigh = fc_neigh(agg)
    h_self = torch.nn.functional.linear(x, fc_self.weight, fc_self.bias)
    if layer_bias is not None:
        h_self = h_self + layer_bias
    return h_self + neigh


def torch_forward(
    model: nn.Sequential,
    h0: torch.Tensor,
    senders: torch.Tensor,
    receivers: torch.Tensor,
    total_charge: float,
    q_ref: torch.Tensor | None,
) -> torch.Tensor:
    """Run Espaloma ``torch.nn.Sequential`` without a DGL graph object."""
    seq = model[0]
    readout = model[1]

    x = seq.f_in(h0)
    for exe in seq._sequential.exes:
        if exe.startswith("d"):
            layer = getattr(seq._sequential, exe)
            lb = getattr(layer, "bias", None)
            x = _sage_mean(x, senders, receivers, layer.fc_self, layer.fc_neigh, lb)
        else:
            x = getattr(seq._sequential, exe)(x)

    es = readout.fc_params(x)
    e, s = es.split(1, -1)

    n = h0.shape[0]
    s_inv = s**-1
    e_s_inv = e * s_inv
    sum_s_inv = s_inv.sum(dim=0, keepdim=True)
    sum_e_s_inv = e_s_inv.sum(dim=0, keepdim=True)
    if q_ref is not None:
        tc = q_ref.sum()
    else:
        tc = torch.tensor([[total_charge]], device=h.device, dtype=h.dtype)
    sum_q = tc.expand(n, 1)
    sum_s_inv_b = sum_s_inv.expand(n, 1)
    sum_e_s_inv_b = sum_e_s_inv.expand(n, 1)
    q = -e * s_inv + s_inv * (sum_q + sum_e_s_inv_b) / sum_s_inv_b
    return q.squeeze(-1)


def load_torch_model(path: str) -> nn.Sequential:
    import sys
    import types
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    stub = root / "tests/stubs"
    if str(stub) not in sys.path:
        sys.path.insert(0, str(stub))
    ref = root / "references/espaloma-charge/espaloma_charge/models.py"
    spec = importlib.util.spec_from_file_location("espaloma_charge.models", ref)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    pkg = types.ModuleType("espaloma_charge")
    pkg.models = mod
    sys.modules["espaloma_charge"] = pkg
    sys.modules["espaloma_charge.models"] = mod

    return torch.load(path, map_location="cpu", weights_only=False)
