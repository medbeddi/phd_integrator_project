"""Derivation symbolique de l'equation d'advection-diffusion via SymPy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import sympy as sp

NumpyCallable = Callable[..., Any]


@dataclass(frozen=True)
class SymbolicContext:
    """Regroupe les symboles et expressions de l'onde solitaire tanh(x - c*t)."""

    x: sp.Symbol
    t: sp.Symbol
    c: sp.Symbol
    nu: sp.Symbol
    u: sp.Expr
    du_dt: sp.Expr
    du_dx: sp.Expr
    d2u_dx2: sp.Expr
    residual: sp.Expr


def get_symbols() -> tuple[sp.Symbol, sp.Symbol, sp.Symbol, sp.Symbol]:
    """Declare les variables (x, t) et les parametres physiques (c, nu)."""
    x, t = sp.symbols("x t", real=True)
    c, nu = sp.symbols("c nu", positive=True)
    return x, t, c, nu


def solitary_wave_solution(x: sp.Symbol, t: sp.Symbol, c: sp.Symbol) -> sp.Expr:
    """Solution analytique candidate de type onde solitaire u(x, t) = tanh(x - c*t)."""
    return sp.tanh(x - c * t)


def compute_derivatives(
    u: sp.Expr, x: sp.Symbol, t: sp.Symbol
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Calcule du/dt, du/dx et d2u/dx2 par differentiation symbolique."""
    du_dt = sp.diff(u, t)
    du_dx = sp.diff(u, x)
    d2u_dx2 = sp.diff(u, x, 2)
    return du_dt, du_dx, d2u_dx2


def residual_source_term(
    du_dt: sp.Expr, du_dx: sp.Expr, d2u_dx2: sp.Expr, c: sp.Symbol, nu: sp.Symbol
) -> sp.Expr:
    """Terme source residuel exact f(x,t) = du/dt + c*du/dx - nu*d2u/dx2."""
    return sp.simplify(du_dt + c * du_dx - nu * d2u_dx2)


def build_symbolic_context() -> SymbolicContext:
    """Construit le contexte symbolique complet (solution, derivees, residu)."""
    x, t, c, nu = get_symbols()
    u = solitary_wave_solution(x, t, c)
    du_dt, du_dx, d2u_dx2 = compute_derivatives(u, x, t)
    residual = residual_source_term(du_dt, du_dx, d2u_dx2, c, nu)
    return SymbolicContext(
        x=x, t=t, c=c, nu=nu, u=u, du_dt=du_dt, du_dx=du_dx, d2u_dx2=d2u_dx2, residual=residual
    )


def lambdify_expression(
    expr: sp.Expr, symbols: tuple[sp.Symbol, ...]
) -> NumpyCallable:
    """Exporte une expression SymPy en fonction NumPy vectorisee via lambdify."""
    return cast(NumpyCallable, sp.lambdify(symbols, expr, modules="numpy"))


def derive_symbolic() -> dict[str, NumpyCallable]:
    """Point d'entree du pipeline: retourne les fonctions lambdifiees u et f."""
    ctx = build_symbolic_context()
    u_numpy = lambdify_expression(ctx.u, (ctx.x, ctx.t, ctx.c))
    f_numpy = lambdify_expression(ctx.residual, (ctx.x, ctx.t, ctx.c, ctx.nu))
    return {"u": u_numpy, "f": f_numpy}


if __name__ == "__main__":
    context = build_symbolic_context()
    sp.pprint(context.u)
    sp.pprint(context.residual)
