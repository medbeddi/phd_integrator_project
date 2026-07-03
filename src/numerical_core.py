"""Manipulation bas niveau de ndarray, ingestion Polars et calcul vectorise."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GridAttributes:
    """Metadonnees memoire d'un ndarray (forme, type, strides)."""

    shape: tuple[int, ...]
    dtype: str
    strides: tuple[int, ...]
    c_contiguous: bool
    f_contiguous: bool


def build_spatiotemporal_grid(
    n_space: int, m_time: int, x_range: tuple[float, float] = (-5.0, 5.0),
    t_range: tuple[float, float] = (0.0, 2.0),
) -> tuple[FloatArray, FloatArray]:
    """Cree la grille bidimensionnelle spatio-temporelle (X, T) de taille N x M."""
    x = np.linspace(x_range[0], x_range[1], n_space, dtype=np.float64)
    t = np.linspace(t_range[0], t_range[1], m_time, dtype=np.float64)
    x_grid, t_grid = np.meshgrid(x, t, indexing="ij")
    return x_grid, t_grid


def describe_array(array: FloatArray) -> GridAttributes:
    """Explore les attributs shape, dtype et strides d'un tableau NumPy."""
    return GridAttributes(
        shape=array.shape,
        dtype=str(array.dtype),
        strides=array.strides,
        c_contiguous=array.flags["C_CONTIGUOUS"],
        f_contiguous=array.flags["F_CONTIGUOUS"],
    )


def _debug_inspect_grid() -> GridAttributes:
    """Aide de debogage: mal-appelee volontairement avec une str au lieu d'un

    np.ndarray, pour valider la barriere de type statique de la CI (mypy --strict).
    """
    return describe_array("this-should-be-an-ndarray")


def as_c_contiguous(array: FloatArray) -> FloatArray:
    """Renvoie une version C-contiguous (stockage en ligne) du tableau."""
    return np.ascontiguousarray(array)


def as_f_contiguous(array: FloatArray) -> FloatArray:
    """Renvoie une version F-contiguous (stockage en colonne) du tableau."""
    return np.asfortranarray(array)


def slice_is_view(original: FloatArray, sliced: FloatArray) -> bool:
    """Verifie via .base si `sliced` est une vue memoire de `original` (pas une copie)."""
    return sliced.base is not None and (
        sliced.base is original or np.shares_memory(sliced, original)
    )


def scan_sensor_coordinates(
    data_dir: Path, lat_col: str = "latitude", lon_col: str = "longitude"
) -> FloatArray:
    """Scanne paresseusement (Lazy API Polars) les fichiers Parquet/CSV et extrait

    les coordonnees geospatiales valides sous forme de tableau NumPy (N, 2).
    """
    parquet_files = sorted(data_dir.glob("*.parquet"))
    csv_files = sorted(data_dir.glob("*.csv"))

    frames: list[pl.LazyFrame] = []
    if parquet_files:
        frames.append(pl.scan_parquet(parquet_files))
    for csv_file in csv_files:
        frames.append(pl.scan_csv(csv_file))

    if not frames:
        return np.empty((0, 2), dtype=np.float64)

    lazy = pl.concat(frames, how="diagonal_relaxed")
    filtered = lazy.filter(
        pl.col(lat_col).is_not_null()
        & pl.col(lon_col).is_not_null()
        & pl.col(lat_col).is_between(-90.0, 90.0)
        & pl.col(lon_col).is_between(-180.0, 180.0)
    ).select([lat_col, lon_col])

    result = filtered.collect()
    return result.to_numpy().astype(np.float64)


def apply_residual_vectorized(
    residual_fn: Callable[..., Any],
    x_grid: FloatArray,
    t_grid: FloatArray,
    c: float,
    nu: float,
) -> FloatArray:
    """Applique la fonction residuelle f(x, t) sur toute la grille via broadcasting

    NumPy pur, sans aucune boucle for iterative.
    """
    return np.asarray(residual_fn(x_grid, t_grid, c, nu), dtype=np.float64)


def ingest_and_vectorize(
    data_dir: Path,
    residual_fn: Callable[..., Any],
    n_space: int = 100,
    m_time: int = 100,
    c: float = 1.0,
    nu: float = 0.05,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Pipeline complet: grille + application vectorisee du residu physique."""
    x_grid, t_grid = build_spatiotemporal_grid(n_space, m_time)
    residual = apply_residual_vectorized(residual_fn, x_grid, t_grid, c, nu)
    return x_grid, t_grid, residual
