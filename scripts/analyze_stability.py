"""Script Snakemake: évalue le conditionnement et la tolérance IEEE 754."""

from src.stability_analysis import analyze_stability

# `snakemake` est injecte automatiquement dans le namespace par la directive `script:`.
snakemake = globals()["snakemake"]  # noqa: F821

reports = analyze_stability()
with open(snakemake.output.report, "w", encoding="utf-8") as fh:
    for r in reports:
        fh.write(
            f"n={r.dimension} kappa={r.condition_number:.4e} "
            f"err16={r.error_float16:.4e} err32={r.error_float32:.4e} "
            f"err64={r.error_float64:.4e}\n"
        )
print("Rapport de stabilite genere.")
