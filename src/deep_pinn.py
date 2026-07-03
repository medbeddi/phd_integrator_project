"""Physics-Informed Neural Network (PINN) pour l'équation d'advection-diffusion."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

Tensor = torch.Tensor


class PINN(nn.Module):
    """Perceptron multicouche (x, t) -> u_hat(x, t)."""

    def __init__(self, hidden_size: int = 32, n_hidden_layers: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, hidden_size), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        inputs = torch.cat([x, t], dim=1)
        result: Tensor = self.net(inputs)
        return result


def get_device() -> torch.device:
    """Détecte et sélectionne la meilleure architecture disponible (CUDA, MPS, CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def physics_residual(model: PINN, x: Tensor, t: Tensor, c: float, nu: float) -> Tensor:
    """Calcule le résidu physique du/dt + c*du/dx - nu*d2u/dx2 - f(x,t) via

    différentiation automatique (torch.autograd.grad), sans maillage explicite.
    """
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = model(x, t)

    grad_outputs = torch.ones_like(u)
    du_dx = torch.autograd.grad(u, x, grad_outputs=grad_outputs, create_graph=True)[0]
    du_dt = torch.autograd.grad(u, t, grad_outputs=grad_outputs, create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(
        du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True
    )[0]

    # Terme source exact issu de Module 3 (methode des solutions manufacturees):
    # pour u_exact = tanh(x - c*t), f(x,t) = du/dt + c*du/dx - nu*d2u/dx2
    #                                       = 2 * nu * u_exact * (1 - u_exact**2)
    u_exact = torch.tanh(x - c * t)
    f_source = 2.0 * nu * u_exact * (1.0 - u_exact**2)

    residual: Tensor = du_dt + c * du_dx - nu * d2u_dx2 - f_source
    return residual


@dataclass(frozen=True)
class LossWeights:
    """Pondération des trois composantes de la perte totale du PINN."""

    physics: float = 1.0
    boundary: float = 1.0
    data: float = 1.0


@dataclass
class TrainingHistory:
    """Historique des pertes enregistrées pendant l'entraînement."""

    total: list[float] = field(default_factory=list)
    physics: list[float] = field(default_factory=list)
    boundary: list[float] = field(default_factory=list)
    data: list[float] = field(default_factory=list)


def compute_total_loss(
    model: PINN,
    collocation_xt: tuple[Tensor, Tensor],
    boundary_xt: tuple[Tensor, Tensor],
    boundary_u: Tensor,
    sensor_xt: tuple[Tensor, Tensor],
    sensor_u: Tensor,
    c: float,
    nu: float,
    weights: LossWeights,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Combine la perte physique (résidu EDP), la perte aux conditions aux limites

    et la perte sur les données capteurs en une perte totale unique.
    """
    x_c, t_c = collocation_xt
    residual = physics_residual(model, x_c, t_c, c, nu)
    loss_physics = torch.mean(residual**2)

    x_b, t_b = boundary_xt
    u_pred_boundary = model(x_b, t_b)
    loss_boundary = torch.mean((u_pred_boundary - boundary_u) ** 2)

    x_s, t_s = sensor_xt
    u_pred_sensor = model(x_s, t_s)
    loss_data = torch.mean((u_pred_sensor - sensor_u) ** 2)

    total = (
        weights.physics * loss_physics
        + weights.boundary * loss_boundary
        + weights.data * loss_data
    )
    return total, loss_physics, loss_boundary, loss_data


def train_pinn(
    model: PINN,
    collocation_xt: tuple[Tensor, Tensor],
    boundary_xt: tuple[Tensor, Tensor],
    boundary_u: Tensor,
    sensor_xt: tuple[Tensor, Tensor],
    sensor_u: Tensor,
    c: float = 1.0,
    nu: float = 0.05,
    epochs: int = 200,
    lr: float = 1e-3,
    weights: LossWeights | None = None,
) -> TrainingHistory:
    """Boucle d'entraînement complète du PINN sur le device optimal détecté."""
    device = get_device()
    model.to(device)
    loss_weights = weights if weights is not None else LossWeights()

    collocation_xt = (collocation_xt[0].to(device), collocation_xt[1].to(device))
    boundary_xt = (boundary_xt[0].to(device), boundary_xt[1].to(device))
    boundary_u = boundary_u.to(device)
    sensor_xt = (sensor_xt[0].to(device), sensor_xt[1].to(device))
    sensor_u = sensor_u.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = TrainingHistory()

    for _ in range(epochs):
        optimizer.zero_grad()
        total, l_phys, l_bound, l_data = compute_total_loss(
            model, collocation_xt, boundary_xt, boundary_u, sensor_xt, sensor_u,
            c, nu, loss_weights,
        )
        total.backward()
        optimizer.step()

        history.total.append(float(total.item()))
        history.physics.append(float(l_phys.item()))
        history.boundary.append(float(l_bound.item()))
        history.data.append(float(l_data.item()))

    return history


def predict(model: PINN, x: Tensor, t: Tensor) -> Tensor:
    """Inférence u_hat(x, t) en mode évaluation, sans calcul de gradient."""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        result: Tensor = model(x.to(device), t.to(device))
    model.train()
    return result.cpu()
