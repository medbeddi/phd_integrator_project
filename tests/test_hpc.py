"""Tests du module hpc_acceleration."""

from __future__ import annotations

import numpy as np

from src.hpc_acceleration import (
    benchmark_filter,
    filter_2d_numba,
    filter_2d_python,
    make_smoothing_kernel,
    parameter_sweep,
)


def test_make_smoothing_kernel_sums_to_one() -> None:
    kernel = make_smoothing_kernel(3)
    assert kernel.shape == (3, 3)
    assert np.isclose(kernel.sum(), 1.0)


def test_filter_2d_python_matches_scipy_style_uniform_average() -> None:
    grid = np.arange(36, dtype=np.float64).reshape(6, 6)
    kernel = make_smoothing_kernel(3)
    output = filter_2d_python(grid, kernel)
    center_value = grid[0:3, 0:3].mean()
    assert np.isclose(output[1, 1], center_value)


def test_filter_2d_numba_matches_python_reference() -> None:
    rng = np.random.default_rng(0)
    grid = rng.standard_normal((12, 12))
    kernel = make_smoothing_kernel(3)
    expected = filter_2d_python(grid, kernel)
    actual = filter_2d_numba(grid, kernel)
    assert np.allclose(actual, expected, atol=1e-5)


def test_benchmark_filter_returns_positive_timings() -> None:
    rng = np.random.default_rng(1)
    grid = rng.standard_normal((20, 20))
    kernel = make_smoothing_kernel(3)
    result = benchmark_filter(grid, kernel, repeats=1)
    assert result.python_seconds > 0
    assert result.numba_seconds > 0
    assert result.speedup > 0


def test_parameter_sweep_sequential_matches_parallel() -> None:
    c_values = np.array([1.0, 1.5])
    nu_values = np.array([0.05, 0.1])

    sequential = parameter_sweep(c_values, nu_values, n_workers=1)
    parallel = parameter_sweep(c_values, nu_values, n_workers=2)

    assert sequential.n_combinations == 4
    assert parallel.n_combinations == 4
    assert np.allclose(sorted(sequential.results), sorted(parallel.results))
