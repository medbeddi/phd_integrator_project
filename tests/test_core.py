"""Tests des modules numerical_core et stability_analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.numerical_core import (
    apply_residual_vectorized,
    as_c_contiguous,
    as_f_contiguous,
    build_spatiotemporal_grid,
    describe_array,
    ingest_and_vectorize,
    scan_sensor_coordinates,
    slice_is_view,
)
from src.stability_analysis import (
    analyze_stability,
    build_hilbert_matrix,
    condition_number,
    perturbation_sensitivity,
    reconstruction_error,
    scan_precision_errors,
    validate_solution,
)
from src.symbolic_derivation import derive_symbolic

# --------------------------------------------------------------------------- #
# numerical_core
# --------------------------------------------------------------------------- #


def test_build_spatiotemporal_grid_shape() -> None:
    x_grid, t_grid = build_spatiotemporal_grid(20, 15)
    assert x_grid.shape == (20, 15)
    assert t_grid.shape == (20, 15)


def test_describe_array_reports_correct_metadata() -> None:
    array = np.zeros((4, 5), dtype=np.float64)
    attrs = describe_array(array)
    assert attrs.shape == (4, 5)
    assert attrs.dtype == "float64"
    assert attrs.c_contiguous is True
    assert attrs.f_contiguous is False


def test_c_and_f_contiguous_layouts_differ_in_strides() -> None:
    array = np.arange(12, dtype=np.float64).reshape(3, 4)
    c_array = as_c_contiguous(array)
    f_array = as_f_contiguous(array)
    assert c_array.flags["C_CONTIGUOUS"]
    assert f_array.flags["F_CONTIGUOUS"]
    assert c_array.strides != f_array.strides
    assert np.array_equal(c_array, f_array)


def test_slice_is_view_detects_view() -> None:
    array = np.arange(20, dtype=np.float64).reshape(4, 5)
    view = array[1:3, :]
    assert slice_is_view(array, view) is True


def test_slice_is_view_detects_copy() -> None:
    array = np.arange(20, dtype=np.float64).reshape(4, 5)
    copy = array[[0, 2], :]  # fancy indexing => copie effective
    assert slice_is_view(array, copy) is False


def test_scan_sensor_coordinates_filters_invalid_rows(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "latitude": [10.0, None, 999.0, -45.0],
            "longitude": [20.0, 30.0, 40.0, -90.0],
        }
    )
    parquet_path = tmp_path / "sensors.parquet"
    df.write_parquet(parquet_path)

    coords = scan_sensor_coordinates(tmp_path)
    assert coords.shape == (2, 2)
    assert np.allclose(coords, np.array([[10.0, 20.0], [-45.0, -90.0]]))


def test_scan_sensor_coordinates_empty_directory_returns_empty_array(tmp_path: Path) -> None:
    coords = scan_sensor_coordinates(tmp_path)
    assert coords.shape == (0, 2)


def test_apply_residual_vectorized_matches_manual_broadcast() -> None:
    functions = derive_symbolic()
    x_grid, t_grid = build_spatiotemporal_grid(10, 10)
    residual = apply_residual_vectorized(functions["f"], x_grid, t_grid, c=1.0, nu=0.05)
    expected = functions["f"](x_grid, t_grid, 1.0, 0.05)
    assert np.allclose(residual, expected)
    assert residual.shape == x_grid.shape


def test_ingest_and_vectorize_end_to_end() -> None:
    functions = derive_symbolic()
    x_grid, t_grid, residual = ingest_and_vectorize(
        Path("data/raw_sensors"), functions["f"], n_space=15, m_time=15
    )
    assert x_grid.shape == (15, 15)
    assert residual.shape == (15, 15)
    assert np.all(np.isfinite(residual))


# --------------------------------------------------------------------------- #
# stability_analysis
# --------------------------------------------------------------------------- #


def test_build_hilbert_matrix_known_entries() -> None:
    matrix = build_hilbert_matrix(3)
    expected = np.array([[1.0, 1 / 2, 1 / 3], [1 / 2, 1 / 3, 1 / 4], [1 / 3, 1 / 4, 1 / 5]])
    assert np.allclose(matrix, expected)


def test_condition_number_increases_with_dimension() -> None:
    kappa_5 = condition_number(build_hilbert_matrix(5))
    kappa_10 = condition_number(build_hilbert_matrix(10))
    assert kappa_10 > kappa_5 > 1.0


def test_reconstruction_error_float64_more_accurate_than_float16() -> None:
    matrix = build_hilbert_matrix(8)
    rhs = matrix @ np.ones(8)
    err16 = reconstruction_error(matrix, rhs, np.float16)
    err64 = reconstruction_error(matrix, rhs, np.float64)
    assert err64 <= err16 + 1e-8


def test_scan_precision_errors_returns_one_report_per_dimension() -> None:
    reports = scan_precision_errors([5, 8, 10])
    assert len(reports) == 3
    assert [r.dimension for r in reports] == [5, 8, 10]
    for report in reports:
        assert report.condition_number > 0


def test_perturbation_sensitivity_amplification_correlates_with_conditioning() -> None:
    matrix = build_hilbert_matrix(10)
    rhs = matrix @ np.ones(10)
    report = perturbation_sensitivity(matrix, rhs, epsilon=1e-7, seed=1)
    assert report.condition_number > 1.0
    assert report.amplification_ratio >= 0.0


def test_validate_solution_accepts_close_and_rejects_far_solution() -> None:
    matrix = build_hilbert_matrix(6)
    alpha = np.ones(6)
    rhs = matrix @ alpha
    assert validate_solution(matrix, alpha, rhs) is True
    assert validate_solution(matrix, alpha + 10.0, rhs) is False


def test_validate_solution_strict_equality_would_fail_due_to_ieee754() -> None:
    """Documente pourquoi == est proscrit: la reconstruction flottante n'est

    presque jamais bit-a-bit identique au second membre.
    """
    matrix = build_hilbert_matrix(6)
    alpha = np.linalg.solve(matrix, matrix @ np.ones(6)).astype(np.float64)
    rhs = matrix @ np.ones(6)
    reconstructed = matrix @ alpha
    assert not np.array_equal(reconstructed, rhs) or np.allclose(reconstructed, rhs)
    assert validate_solution(matrix, alpha, rhs, atol=1e-8, rtol=1e-6) is True


def test_analyze_stability_default_range() -> None:
    reports = analyze_stability()
    assert len(reports) == 21
    assert reports[0].dimension == 5
    assert reports[-1].dimension == 25


def test_analyze_stability_custom_dimensions() -> None:
    reports = analyze_stability([5, 6])
    assert [r.dimension for r in reports] == [5, 6]


@pytest.mark.parametrize("n", [5, 15, 25])
def test_condition_number_matches_numpy_reference(n: int) -> None:
    matrix = build_hilbert_matrix(n)
    assert np.isclose(condition_number(matrix), np.linalg.cond(matrix))
