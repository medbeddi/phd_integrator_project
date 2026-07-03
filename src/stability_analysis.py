"""Sensibilite, conditionnement et analyse des erreurs IEEE 754."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PrecisionErrorReport:
    """Erreurs de reconstruction du systeme A.alpha = b par precision flottante."""

    dimension: int
    condition_number: float
    error_float16: float
    error_float32: float
    error_float64: float


@dataclass(frozen=True)
class PerturbationReport:
    """Amplification d'une perturbation du second membre b sur la solution alpha."""

    condition_number: float
    relative_perturbation_b: float
    relative_perturbation_alpha: float
    amplification_ratio: float


def build_hilbert_matrix(n: int) -> FloatArray:
    """Construit la matrice de Hilbert n x n, A_ij = 1 / (i + j + 1)."""
    return np.array(
        [[1.0 / (i + j + 1) for j in range(n)] for i in range(n)], dtype=np.float64
    )


def condition_number(matrix: FloatArray) -> float:
    """Nombre de conditionnement kappa(A) = ||A|| * ||A^-1|| (norme 2)."""
    return float(np.linalg.cond(matrix))


def _solve_at_precision(
    matrix: FloatArray, rhs: FloatArray, dtype: type[np.floating[Any]]
) -> FloatArray:
    a_cast: FloatArray = matrix.astype(dtype)
    b_cast: FloatArray = rhs.astype(dtype)
    solution = np.linalg.solve(a_cast.astype(np.float64), b_cast.astype(np.float64))
    return solution.astype(np.float64)


def reconstruction_error(
    matrix: FloatArray, rhs: FloatArray, dtype: type[np.floating[Any]]
) -> float:
    """Erreur de reconstruction ||A alpha - b|| pour une precision donnee."""
    alpha = _solve_at_precision(matrix, rhs, dtype)
    residual = matrix.astype(np.float64) @ alpha - rhs.astype(np.float64)
    return float(np.linalg.norm(residual))


def scan_precision_errors(dimensions: list[int]) -> list[PrecisionErrorReport]:
    """Fait varier n de 5 a 25 et compare float16/32/64 sur la matrice de Hilbert."""
    reports = []
    for n in dimensions:
        matrix = build_hilbert_matrix(n)
        rhs = matrix @ np.ones(n, dtype=np.float64)
        reports.append(
            PrecisionErrorReport(
                dimension=n,
                condition_number=condition_number(matrix),
                error_float16=reconstruction_error(matrix, rhs, np.float16),
                error_float32=reconstruction_error(matrix, rhs, np.float32),
                error_float64=reconstruction_error(matrix, rhs, np.float64),
            )
        )
    return reports


def perturbation_sensitivity(
    matrix: FloatArray, rhs: FloatArray, epsilon: float = 1e-7, seed: int = 0
) -> PerturbationReport:
    """Mesure l'amplification d'une perturbation d'ordre epsilon sur b vers alpha,

    et relie empiriquement ce ratio a kappa(A).
    """
    rng = np.random.default_rng(seed)
    alpha = np.linalg.solve(matrix, rhs)

    delta_b = epsilon * rng.standard_normal(rhs.shape)
    perturbed_rhs = rhs + delta_b
    perturbed_alpha = np.linalg.solve(matrix, perturbed_rhs)

    rel_perturbation_b = float(np.linalg.norm(delta_b) / np.linalg.norm(rhs))
    rel_perturbation_alpha = float(
        np.linalg.norm(perturbed_alpha - alpha) / np.linalg.norm(alpha)
    )
    amplification = (
        rel_perturbation_alpha / rel_perturbation_b if rel_perturbation_b > 0 else 0.0
    )
    return PerturbationReport(
        condition_number=condition_number(matrix),
        relative_perturbation_b=rel_perturbation_b,
        relative_perturbation_alpha=rel_perturbation_alpha,
        amplification_ratio=amplification,
    )


def validate_solution(
    matrix: FloatArray,
    alpha: FloatArray,
    rhs: FloatArray,
    atol: float = 1e-6,
    rtol: float = 1e-4,
) -> bool:
    """Valide Aalpha ~= b via np.isclose (les egalites strictes == sont proscrites

    a cause des erreurs d'arrondi IEEE 754 non reproductibles bit-a-bit).
    """
    residual = matrix @ alpha
    return bool(np.allclose(residual, rhs, atol=atol, rtol=rtol))


def analyze_stability(dimensions: list[int] | None = None) -> list[PrecisionErrorReport]:
    """Point d'entree du pipeline de stabilite pour l'orchestration Snakemake."""
    dims = dimensions if dimensions is not None else list(range(5, 26))
    return scan_precision_errors(dims)
