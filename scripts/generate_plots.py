"""Produit les rapports visuels statiques (PDF) et interactifs (HTML) à partir

des prédictions du PINN sauvegardées par scripts/train_pinn.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.visualization import generate_plots

MODELS_DIR = Path("outputs/models")
FIGURES_DIR = Path("outputs/figures")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(MODELS_DIR / "pinn_predictions.npz")

    pdf_path, html_path = generate_plots(
        data["x_grid"], data["t_grid"], data["u_predicted"], data["u_reference"], FIGURES_DIR
    )
    print(f"Figures generees: {pdf_path}, {html_path}")


if __name__ == "__main__":
    main()
