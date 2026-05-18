# Reproducibility notes

This repository is intended to reproduce the simulator-based tables and figures reported in the Sentinel-RTA revised manuscript. It does not validate live-network deployment, hardware data planes, P4/eBPF enforcement, or real traffic generation.

## Validation checklist

Run from the repository root:

```bash
python -m compileall -q sentinel_rta
pytest -q
python sentinel_rta/scripts/audit_missing_files.py
python sentinel_rta/scripts/audit_result_consistency.py
python sentinel_rta/scripts/generate_manuscript_result_figures.py
python sentinel_rta/scripts/generate_fig7_ieee_v11.py
python sentinel_rta/scripts/generate_manuscript_latex_tables.py
```

## Result package

The curated manuscript artifacts are stored in:

```text
results/manuscript_results_package/
```

The `source_csv/` directory contains the tabular inputs used to regenerate the reported LaTeX tables and figures. The `audits/` directory contains missing-file and consistency checks. The `figures/` and `tables/` directories contain generated manuscript artifacts.

## Interpretation boundary

The artifact supports simulator-based reproducibility and ablation checking. It does not establish unrestricted robustness under arbitrary traffic, unseen adversaries, or deployed hardware constraints.
