"""Orchestration du pipeline scientifique complet via Snakemake (DAG reproductible)."""

rule all:
    input:
        "outputs/figures/solution_and_error.pdf",
        "outputs/figures/interactive_surface.html",
        "outputs/reports/stability_report.txt",


rule derive_symbolic:
    output:
        touch("outputs/.derive_symbolic.done"),
    log:
        "outputs/logs/derive_symbolic.log",
    shell:
        "uv run python -m src.symbolic_derivation > {log} 2>&1"


rule ingest_and_vectorize:
    input:
        "outputs/.derive_symbolic.done",
    output:
        touch("outputs/.ingest_and_vectorize.done"),
    log:
        "outputs/logs/ingest_and_vectorize.log",
    script:
        "scripts/ingest_and_vectorize.py"


rule analyze_stability:
    input:
        "outputs/.derive_symbolic.done",
    output:
        report="outputs/reports/stability_report.txt",
    log:
        "outputs/logs/analyze_stability.log",
    script:
        "scripts/analyze_stability.py"


rule train_pinn:
    input:
        "outputs/.ingest_and_vectorize.done",
    output:
        weights="outputs/models/pinn_weights.pt",
        predictions="outputs/models/pinn_predictions.npz",
    log:
        "outputs/logs/train_pinn.log",
    shell:
        "uv run python scripts/train_pinn.py > {log} 2>&1"


rule generate_plots:
    input:
        "outputs/models/pinn_predictions.npz",
    output:
        pdf="outputs/figures/solution_and_error.pdf",
        html="outputs/figures/interactive_surface.html",
    log:
        "outputs/logs/generate_plots.log",
    shell:
        "uv run python scripts/generate_plots.py > {log} 2>&1"
