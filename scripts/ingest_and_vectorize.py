"""Script Snakemake: filtre les données Parquet et vectorise la grille de discrétisation."""

from pathlib import Path

from src.numerical_core import ingest_and_vectorize
from src.symbolic_derivation import derive_symbolic

functions = derive_symbolic()
x_grid, t_grid, residual = ingest_and_vectorize(
    Path("data/raw_sensors"), functions["f"], n_space=100, m_time=100
)
print(f"Grille vectorisee generee: {x_grid.shape}, residu fini: {residual.dtype}")
