# Reproducibility notes

Run validation from the repository root:

```bash
python -m compileall -q sentinel_rta
pytest -q
python sentinel_rta/scripts/audit_missing_files.py
python sentinel_rta/scripts/audit_result_consistency.py
python sentinel_rta/scripts/generate_manuscript_result_figures.py
python sentinel_rta/scripts/generate_fig7_ieee_v11.py
python sentinel_rta/scripts/generate_manuscript_latex_tables.py
```

The artifact is simulator-based and does not validate live deployment, hardware data planes, P4/eBPF enforcement, or real traffic generation.
