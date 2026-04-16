import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import typer


def _find_repo_root_for_scripts() -> Path | None:
    """Locate project root (contains ``scripts/convert_weights.py``), or None if not a dev checkout."""
    p = Path(__file__).resolve()
    for _ in range(8):
        if (p / "scripts" / "convert_weights.py").is_file():
            return p
        if p.parent == p:
            break
        p = p.parent
    return None

app = typer.Typer(
    name="expaloma",
    help="Expaloma: JAX/Equinox port of the espaloma-charge partial charge inference model.",
    invoke_without_command=True,
)


@app.callback()
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-V", help="Print package version and exit."),
) -> None:
    if version:
        import importlib.metadata

        typer.echo(importlib.metadata.version("expaloma"))
        raise typer.Exit(0)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(0)


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
    repo = _find_repo_root_for_scripts()
    if repo is None:
        typer.secho(
            "convert-weights needs a git checkout with scripts/ (and submodules). "
            "From the repo: python scripts/convert_weights.py MODEL.pt OUT.eqx",
            err=True,
        )
        raise typer.Exit(1)
    script = repo / "scripts" / "convert_weights.py"
    subprocess.check_call([sys.executable, str(script), str(pytorch_model), str(output)])


def run() -> None:
    app()


if __name__ == "__main__":
    run()
