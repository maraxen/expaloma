"""Minimal ``SAGEConv`` compatible with PyTorch unpickling (weights only)."""

from __future__ import annotations

import torch.nn as nn


class SAGEConv(nn.Module):
    """Subset of DGL ``SAGEConv`` (mean) needed to hold ``state_dict``."""

    def __init__(
        self,
        in_feats,
        out_feats,
        aggregator_type,
        feat_drop=0.0,
        bias=True,
        norm=None,
        activation=None,
    ):
        super().__init__()
        self.fc_neigh = nn.Linear(in_feats, out_feats, bias=False)
        self.fc_self = nn.Linear(in_feats, out_feats, bias=bias)
        self._aggre_type = aggregator_type

    def forward(self, graph, feat):
        raise RuntimeError("Stub SAGEConv: use reference forward in tests only")
