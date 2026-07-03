"""Accélération JIT (Numba) d'un filtre de convolution local sur la grille numérique."""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _residual_l2_norm(params: tuple[float, float]) -> float:
    """Evalue la norme L2 du residu tanh pour un couple (c, nu) donne.

    Fonction module-level requise par ProcessPoolExecutor (picklable).
    """
    c, nu = params
    x = np.linspace(-5.0, 5.0, 200)
    t = np.linspace(0.0, 2.0, 200)
    x_grid, t_grid = np.meshgrid(x, t, indexing="ij")
    u = np.tanh(x_grid - c * t_grid)
    residual = (1 - u**2) * (-c) + c * (1 - u**2) - nu * (-2 * u * (1 - u**2) ** 2)
    return float(np.linalg.norm(residual))


def filter_2d_python(grid: FloatArray, kernel: FloatArray) -> FloatArray:
    """Filtrage local naïf en boucles Python pures (référence, non optimisé)."""
    n_rows, n_cols = grid.shape
    k_rows, k_cols = kernel.shape
    pad_r, pad_c = k_rows // 2, k_cols // 2
    output = np.zeros_like(grid)

    for i in range(pad_r, n_rows - pad_r):
        for j in range(pad_c, n_cols - pad_c):
            acc = 0.0
            for ki in range(k_rows):
                for kj in range(k_cols):
                    acc += grid[i - pad_r + ki, j - pad_c + kj] * kernel[ki, kj]
            output[i, j] = acc
    return output


@njit(parallel=True, fastmath=True, cache=True)
def filter_2d_numba(grid: FloatArray, kernel: FloatArray) -> FloatArray:
    """Filtrage local accéléré JIT : boucles parallèles multi-cœurs (prange) et

    relaxation IEEE 754 (fastmath) pour autoriser les réordonnancements du compilateur.
    """
    n_rows, n_cols = grid.shape
    k_rows, k_cols = kernel.shape
    pad_r, pad_c = k_rows // 2, k_cols // 2
    output = np.zeros_like(grid)

    for i in prange(pad_r, n_rows - pad_r):  # type: ignore[attr-defined]
        for j in range(pad_c, n_cols - pad_c):
            acc = 0.0
            for ki in range(k_rows):
                for kj in range(k_cols):
                    acc += grid[i - pad_r + ki, j - pad_c + kj] * kernel[ki, kj]
            output[i, j] = acc
    return output


def make_smoothing_kernel(size: int = 3) -> FloatArray:
    """Noyau de lissage uniforme normalise de taille size x size."""
    return np.full((size, size), 1.0 / (size * size), dtype=np.float64)


@dataclass(frozen=True)
class BenchmarkResult:
    """Temps d'exécution comparés entre implémentation Python et Numba JIT."""

    python_seconds: float
    numba_seconds: float
    speedup: float


def benchmark_filter(grid: FloatArray, kernel: FloatArray, repeats: int = 3) -> BenchmarkResult:
    """Mesure le gain de performance de Numba par rapport aux boucles Python via timeit."""
    filter_2d_numba(grid[:8, :8], kernel)  # warm-up: force la compilation JIT

    start = time.perf_counter()
    for _ in range(repeats):
        filter_2d_python(grid, kernel)
    python_time = (time.perf_counter() - start) / repeats

    start = time.perf_counter()
    for _ in range(repeats):
        filter_2d_numba(grid, kernel)
    numba_time = (time.perf_counter() - start) / repeats

    speedup = python_time / numba_time if numba_time > 0 else float("inf")
    return BenchmarkResult(python_seconds=python_time, numba_seconds=numba_time, speedup=speedup)


@dataclass(frozen=True)
class ParameterSweepResult:
    """Temps d'exécution d'un balayage de paramètres (c, nu) en fonction du nombre de workers."""

    n_combinations: int
    n_workers: int
    seconds: float
    results: tuple[float, ...]


def parameter_sweep(
    c_values: FloatArray, nu_values: FloatArray, n_workers: int = 4
) -> ParameterSweepResult:
    """Explore l'espace des paramètres (c, nu) en parallèle multi-processus

    (concurrent.futures.ProcessPoolExecutor, natif) et mesure le temps global.
    """
    combinations = [(float(c), float(nu)) for c in c_values for nu in nu_values]

    start = time.perf_counter()
    if n_workers <= 1:
        results = [_residual_l2_norm(p) for p in combinations]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_residual_l2_norm, combinations))
    elapsed = time.perf_counter() - start

    return ParameterSweepResult(
        n_combinations=len(combinations),
        n_workers=n_workers,
        seconds=elapsed,
        results=tuple(results),
    )
