import sys
from pathlib import Path
from typing import Optional

import numpy as np
import typer

app = typer.Typer(
    help="Expaloma: Native JAX/Equinox port of the espaloma-charge partial charge inference model."
)


@app.command()
def infer(
    smiles: str = typer.Argument(..., help="SMILES string to assign charges to"),
    weights: Optional[Path] = typer.Option(
        None,
        "--weights",
        "-w",
        help="Equinox weights (.eqx); defaults to bundled v0.0.8 checkpoint",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional path to save charges as .npy",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print verbose output"),
):
    """Assign ESPALOMA charges to a molecule (implicit H; matches ``espaloma_charge.charge``)."""
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from expaloma.infer import charges_for_smiles

    if verbose:
        typer.echo(f"Assigning charges to SMILES: {smiles}")

    q = charges_for_smiles(smiles, weights)

    if output:
        np.save(output, q)
        typer.echo(f"Charges saved to {output}")
    else:
        typer.echo("Charges:")
        typer.echo(q)


@app.command("convert-weights")
def convert_weights(
    pytorch_model: Path = typer.Argument(..., help="Path to the original .espaloma_charge_model.pt"),
    output: Path = typer.Argument(..., help="Path to write Equinox weights (.eqx)"),
):
    """Convert PyTorch checkpoint to Equinox ``.eqx`` (requires PyTorch for this command)."""
    import subprocess

    script = Path(__file__).resolve().parents[2] / "scripts" / "convert_weights.py"
    subprocess.check_call([sys.executable, str(script), str(pytorch_model), str(output)])


def main():
    app()


if __name__ == "__main__":
    main()
