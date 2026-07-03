"""Visualisation scientifique pour publication (Matplotlib/Seaborn statique, Plotly interactif)."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import seaborn as sns  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

FloatArray = NDArray[np.float64]


def configure_latex_rendering() -> bool:
    """Active le rendu LaTeX natif (text.usetex) si un binaire latex est disponible

    sur le PATH. Repli automatique sur le moteur mathtext interne de Matplotlib
    sinon, afin de respecter la contrainte de reproductibilite "No-Sudo" (une
    installation TeX Live via apt necessiterait des privileges root sur les
    executeurs GitHub Actions).
    """
    latex_available = shutil.which("latex") is not None
    plt.rcParams["text.usetex"] = latex_available
    plt.rcParams["font.family"] = "serif"
    return latex_available


def plot_solution_and_error(
    x_grid: FloatArray,
    t_grid: FloatArray,
    u_predicted: FloatArray,
    u_reference: FloatArray,
    output_path: Path,
) -> Path:
    """Genere une figure a deux panneaux : heatmap de la solution PINN et evolution

    temporelle de l'erreur absolue de prediction. Export vectoriel exclusif en PDF.
    """
    configure_latex_rendering()
    sns.set_theme(style="whitegrid")

    absolute_error = np.abs(u_predicted - u_reference)
    mean_error_over_time = absolute_error.mean(axis=0)
    t_axis = t_grid[0, :]

    fig, (ax_heatmap, ax_error) = plt.subplots(1, 2, figsize=(11, 4.5))

    mesh = ax_heatmap.pcolormesh(t_grid, x_grid, u_predicted, shading="auto", cmap="viridis")
    ax_heatmap.set_xlabel(r"$t$")
    ax_heatmap.set_ylabel(r"$x$")
    ax_heatmap.set_title(r"Solution approchee $\hat{u}(x, t)$ (PINN)")
    fig.colorbar(mesh, ax=ax_heatmap)

    ax_error.plot(t_axis, mean_error_over_time, color="firebrick", linewidth=2)
    ax_error.set_xlabel(r"$t$")
    ax_error.set_ylabel(r"Erreur absolue moyenne $|\hat{u} - u|$")
    ax_error.set_title("Evolution temporelle de l'erreur")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    return output_path


def plot_interactive_surface(
    x_grid: FloatArray, t_grid: FloatArray, u_values: FloatArray, output_path: Path
) -> Path:
    """Produit une surface 3D interactive u_hat(x, t) exportee en HTML autonome (Plotly)."""
    figure = go.Figure(
        data=[
            go.Surface(
                x=t_grid,
                y=x_grid,
                z=u_values,
                colorscale="Viridis",
                colorbar={"title": "u_hat"},
            )
        ]
    )
    figure.update_layout(
        title="Surface interactive u_hat(x, t)",
        scene={"xaxis_title": "t", "yaxis_title": "x", "zaxis_title": "u_hat"},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)
    return output_path


def generate_plots(
    x_grid: FloatArray,
    t_grid: FloatArray,
    u_predicted: FloatArray,
    u_reference: FloatArray,
    figures_dir: Path,
) -> tuple[Path, Path]:
    """Point d'entree du pipeline de visualisation pour l'orchestration Snakemake."""
    pdf_path = plot_solution_and_error(
        x_grid, t_grid, u_predicted, u_reference, figures_dir / "solution_and_error.pdf"
    )
    html_path = plot_interactive_surface(
        x_grid, t_grid, u_predicted, figures_dir / "interactive_surface.html"
    )
    return pdf_path, html_path
