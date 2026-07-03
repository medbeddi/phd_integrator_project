"""Tests du module visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.numerical_core import build_spatiotemporal_grid
from src.visualization import (
    configure_latex_rendering,
    generate_plots,
    plot_interactive_surface,
    plot_solution_and_error,
)


def test_configure_latex_rendering_returns_bool() -> None:
    result = configure_latex_rendering()
    assert isinstance(result, bool)


def test_plot_solution_and_error_writes_pdf(tmp_path: Path) -> None:
    x_grid, t_grid = build_spatiotemporal_grid(20, 20)
    u_predicted = np.tanh(x_grid - t_grid)
    u_reference = np.tanh(x_grid - t_grid) + 0.01
    output_path = tmp_path / "figures" / "solution_and_error.pdf"

    result_path = plot_solution_and_error(x_grid, t_grid, u_predicted, u_reference, output_path)

    assert result_path.exists()
    assert result_path.suffix == ".pdf"
    assert result_path.stat().st_size > 0


def test_plot_interactive_surface_writes_html(tmp_path: Path) -> None:
    x_grid, t_grid = build_spatiotemporal_grid(15, 15)
    u_values = np.tanh(x_grid - t_grid)
    output_path = tmp_path / "figures" / "surface.html"

    result_path = plot_interactive_surface(x_grid, t_grid, u_values, output_path)

    assert result_path.exists()
    content = result_path.read_text(encoding="utf-8")
    assert "<html" in content.lower()


def test_generate_plots_produces_both_artifacts(tmp_path: Path) -> None:
    x_grid, t_grid = build_spatiotemporal_grid(10, 10)
    u_predicted = np.tanh(x_grid - t_grid)
    u_reference = u_predicted + 0.02

    pdf_path, html_path = generate_plots(x_grid, t_grid, u_predicted, u_reference, tmp_path)

    assert pdf_path.exists()
    assert html_path.exists()
