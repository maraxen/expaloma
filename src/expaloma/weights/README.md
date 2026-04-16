# Bundled checkpoint (v0.0.8)

| Artifact | Source |
|----------|--------|
| `espaloma_v0_0_8.eqx` | Equinox leaf serialization of the public PyTorch release |

**Upstream PyTorch weights:** [espaloma_charge v0.0.8 `model.pt`](https://github.com/choderalab/espaloma_charge/releases/download/v0.0.8/model.pt)

**SHA256 (verify after download):**

```
model.pt (upstream):     ed3e396df3f5ee8dd42240165202be10c59bd7e60c41841512c18b0329ed76f2
espaloma_v0_0_8.eqx:     a7fa565aca2961b2f960b62801025e9426ca3a4f3536194a424bbf9a4ea44993
```

Regenerate `.eqx` from a local `model.pt`:

```bash
uv run python scripts/convert_weights.py path/to/model.pt src/expaloma/weights/espaloma_v0_0_8.eqx
sha256sum src/expaloma/weights/espaloma_v0_0_8.eqx  # compare with line above
```

Derived weights are distributed under the same spirit as [espaloma_charge](https://github.com/choderalab/espaloma_charge) (MIT); retain upstream copyright notices in distributions.
