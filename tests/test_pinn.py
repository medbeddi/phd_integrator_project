"""Tests du module deep_pinn (PyTorch)."""

from __future__ import annotations

import torch

from src.deep_pinn import (
    PINN,
    LossWeights,
    compute_total_loss,
    get_device,
    physics_residual,
    predict,
    train_pinn,
)


def test_get_device_returns_valid_torch_device() -> None:
    device = get_device()
    assert device.type in {"cpu", "cuda", "mps"}


def test_pinn_forward_shape() -> None:
    model = PINN(hidden_size=8, n_hidden_layers=2)
    x = torch.zeros((5, 1))
    t = torch.zeros((5, 1))
    output = model(x, t)
    assert output.shape == (5, 1)


def test_physics_residual_shape_matches_input() -> None:
    model = PINN(hidden_size=8, n_hidden_layers=2)
    x = torch.linspace(-2, 2, 10).reshape(-1, 1)
    t = torch.linspace(0, 1, 10).reshape(-1, 1)
    residual = physics_residual(model, x, t, c=1.0, nu=0.05)
    assert residual.shape == (10, 1)
    assert torch.isfinite(residual).all()


def test_physics_residual_is_near_zero_for_exact_solution_module() -> None:
    """Un modele qui reproduit exactement tanh(x - c*t) doit annuler le residu physique."""

    class ExactModel(torch.nn.Module):
        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return torch.tanh(x - 1.0 * t)

    model = ExactModel()
    x = torch.linspace(-2, 2, 8).reshape(-1, 1)
    t = torch.linspace(0, 1, 8).reshape(-1, 1)
    residual = physics_residual(model, x, t, c=1.0, nu=0.05)  # type: ignore[arg-type]
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-5)


def test_compute_total_loss_combines_three_components() -> None:
    model = PINN(hidden_size=8, n_hidden_layers=2)
    x_c = torch.linspace(-2, 2, 6).reshape(-1, 1)
    t_c = torch.linspace(0, 1, 6).reshape(-1, 1)
    x_b = torch.tensor([[-2.0], [2.0]])
    t_b = torch.tensor([[0.5], [0.5]])
    u_b = torch.tanh(x_b - t_b)
    x_s = torch.tensor([[0.0]])
    t_s = torch.tensor([[0.0]])
    u_s = torch.tensor([[0.0]])

    total, l_phys, l_bound, l_data = compute_total_loss(
        model, (x_c, t_c), (x_b, t_b), u_b, (x_s, t_s), u_s, c=1.0, nu=0.05,
        weights=LossWeights(),
    )
    assert total.item() >= 0.0
    assert torch.isclose(total, l_phys + l_bound + l_data)


def test_train_pinn_reduces_loss_over_epochs() -> None:
    torch.manual_seed(0)
    model = PINN(hidden_size=16, n_hidden_layers=2)

    x_c = torch.linspace(-2, 2, 20).reshape(-1, 1)
    t_c = torch.linspace(0, 1, 20).reshape(-1, 1)
    x_b = torch.tensor([[-2.0], [2.0]] * 5)
    t_b = torch.linspace(0, 1, 10).reshape(-1, 1)
    u_b = torch.tanh(x_b - t_b)
    x_s = torch.linspace(-2, 2, 10).reshape(-1, 1)
    t_s = torch.zeros((10, 1))
    u_s = torch.tanh(x_s)

    history = train_pinn(
        model, (x_c, t_c), (x_b, t_b), u_b, (x_s, t_s), u_s,
        c=1.0, nu=0.05, epochs=15, lr=1e-2,
    )
    assert len(history.total) == 15
    assert history.total[-1] <= history.total[0]


def test_predict_returns_cpu_tensor_without_grad() -> None:
    model = PINN(hidden_size=8, n_hidden_layers=2)
    x = torch.linspace(-1, 1, 5).reshape(-1, 1)
    t = torch.zeros((5, 1))
    prediction = predict(model, x, t)
    assert prediction.device.type == "cpu"
    assert prediction.requires_grad is False
    assert prediction.shape == (5, 1)
