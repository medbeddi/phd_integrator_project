"""Entraîne le PINN et sauvegarde les poids ainsi que les prédictions sur grille."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.deep_pinn import PINN, get_device, predict, train_pinn
from src.numerical_core import build_spatiotemporal_grid

MODELS_DIR = Path("outputs/models")

C_COEFFICIENT = 1.0
NU_COEFFICIENT = 0.05


def _make_training_points() -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
]:
    x_c = torch.linspace(-5, 5, 40).reshape(-1, 1)
    t_c = torch.linspace(0, 2, 40).reshape(-1, 1)

    t_b = torch.linspace(0, 2, 20).reshape(-1, 1)
    x_b = torch.cat([torch.full_like(t_b, -5.0), torch.full_like(t_b, 5.0)])
    t_b = torch.cat([t_b, t_b])
    u_b = torch.tanh(x_b - C_COEFFICIENT * t_b)

    x_s = torch.linspace(-5, 5, 20).reshape(-1, 1)
    t_s = torch.zeros((20, 1))
    u_s = torch.tanh(x_s)

    return (x_c, t_c), (x_b, t_b), u_b, (x_s, t_s), u_s


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    collocation, boundary_xt, boundary_u, sensor_xt, sensor_u = _make_training_points()

    torch.manual_seed(0)
    model = PINN(hidden_size=32, n_hidden_layers=4)
    train_pinn(
        model, collocation, boundary_xt, boundary_u, sensor_xt, sensor_u,
        c=C_COEFFICIENT, nu=NU_COEFFICIENT, epochs=100, lr=1e-3,
    )
    torch.save(model.state_dict(), MODELS_DIR / "pinn_weights.pt")

    x_grid, t_grid = build_spatiotemporal_grid(60, 60)
    x_flat = torch.tensor(x_grid.reshape(-1, 1), dtype=torch.float32)
    t_flat = torch.tensor(t_grid.reshape(-1, 1), dtype=torch.float32)
    u_predicted = predict(model, x_flat, t_flat).numpy().reshape(x_grid.shape)
    u_reference = (torch.tanh(x_flat - C_COEFFICIENT * t_flat).numpy()).reshape(x_grid.shape)

    np.savez(
        MODELS_DIR / "pinn_predictions.npz",
        x_grid=x_grid, t_grid=t_grid, u_predicted=u_predicted, u_reference=u_reference,
    )

    print(f"Device utilise pour l'entrainement: {get_device()}")
    print(f"Poids sauvegardes dans {MODELS_DIR / 'pinn_weights.pt'}")


if __name__ == "__main__":
    main()
