"""Tests du module symbolic_derivation."""

from __future__ import annotations

import numpy as np
import sympy as sp

from src.symbolic_derivation import (
    build_symbolic_context,
    compute_derivatives,
    derive_symbolic,
    get_symbols,
    lambdify_expression,
    residual_source_term,
    solitary_wave_solution,
)


def test_get_symbols_returns_four_distinct_symbols() -> None:
    x, t, c, nu = get_symbols()
    assert {x, t, c, nu} == {x, t, c, nu}
    assert len({x, t, c, nu}) == 4


def test_solitary_wave_solution_matches_tanh() -> None:
    x, t, c, _ = get_symbols()
    u = solitary_wave_solution(x, t, c)
    assert u == sp.tanh(x - c * t)


def test_compute_derivatives_matches_manual_sympy_diff() -> None:
    x, t, c, _ = get_symbols()
    u = solitary_wave_solution(x, t, c)
    du_dt, du_dx, d2u_dx2 = compute_derivatives(u, x, t)
    assert sp.simplify(du_dt - sp.diff(u, t)) == 0
    assert sp.simplify(du_dx - sp.diff(u, x)) == 0
    assert sp.simplify(d2u_dx2 - sp.diff(u, x, 2)) == 0


def test_residual_source_term_is_zero_for_exact_solution() -> None:
    """u = tanh(x - c*t) est une solution exacte de l'equation homogene:

    le residu source doit donc etre analytiquement nul.
    """
    x, t, c, nu = get_symbols()
    u = solitary_wave_solution(x, t, c)
    du_dt, du_dx, d2u_dx2 = compute_derivatives(u, x, t)
    residual = residual_source_term(du_dt, du_dx, d2u_dx2, c, nu)
    assert sp.simplify(residual.subs(nu, 0)) == 0


def test_build_symbolic_context_is_consistent() -> None:
    ctx = build_symbolic_context()
    assert ctx.u == sp.tanh(ctx.x - ctx.c * ctx.t)
    assert ctx.du_dt is not None
    assert ctx.residual is not None


def test_lambdify_expression_evaluates_numerically() -> None:
    x, t, c, _ = get_symbols()
    u = solitary_wave_solution(x, t, c)
    u_numpy = lambdify_expression(u, (x, t, c))
    value = u_numpy(0.0, 0.0, 1.0)
    assert np.isclose(value, 0.0, atol=1e-10)


def test_lambdify_expression_is_vectorized() -> None:
    x, t, c, _ = get_symbols()
    u = solitary_wave_solution(x, t, c)
    u_numpy = lambdify_expression(u, (x, t, c))
    xs = np.linspace(-2, 2, 10)
    values = u_numpy(xs, 0.0, 1.0)
    assert values.shape == (10,)
    assert np.allclose(values, np.tanh(xs))


def test_derive_symbolic_returns_callable_functions() -> None:
    functions = derive_symbolic()
    assert set(functions.keys()) == {"u", "f"}
    u_val = functions["u"](1.0, 0.0, 1.0)
    assert np.isclose(u_val, np.tanh(1.0))
