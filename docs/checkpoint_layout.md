# Released `model.pt` layout (v0.0.8)

Inspected from `https://github.com/choderalab/espaloma_charge/releases/download/v0.0.8/model.pt`.

Top-level module: `torch.nn.Sequential(Sequential, ChargeReadout, ChargeEquilibrium)`.

## State dict keys

| Key | Shape | Role |
|-----|-------|------|
| `0.f_in.0.weight` | (128, 116) | Input linear |
| `0.f_in.0.bias` | (128,) | |
| `0._sequential.d{0,2,4,6}.fc_neigh.weight` | (128, 128) | SAGE neighbor transform |
| `0._sequential.d{0,2,4,6}.fc_self.weight` | (128, 128) | SAGE self transform |
| `0._sequential.d{0,2,4,6}.bias` | (128,) | Standalone bias tensor (DGL stores ``fc_self`` as ``Linear(..., bias=False)`` plus this parameter; add to the self linear term) |
| `1.fc_params.weight` | (2, 128) | Charge readout |
| `1.fc_params.bias` | (2,) | |

No BatchNorm or dropout in this checkpoint. Four SAGEConv layers (mean), 128→128 each, with ReLU after each. `ChargeEquilibrium` has no parameters.

Input width **116** matches `Sequential(feature_units=116)` and `from_rdkit_mol`.

For these layers `in_feats == out_feats == 128`, DGL `lin_before_mp` is false: neighbor mean of raw features, then `fc_neigh` on the aggregate.
