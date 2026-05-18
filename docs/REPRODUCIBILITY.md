# Reproducibility notes

This is a **code-only** repository for the Sentinel-RTA simulator and implementation. It supports simulator-based execution and code validation. It does not include manuscript PDFs, LaTeX files, precomputed manuscript figures/tables, live-network deployment code, or real attack-traffic generators.

## Validation checklist

Run from the repository root:

```bash
python -m compileall -q sentinel_rta
pytest -q
```

The default test configuration excludes long-running experiment smoke tests. To include long-running tests, run:

```bash
pytest -q -m expensive
```

## Typical workflow

1. Install dependencies from `requirements.txt`.
2. Run the compile and test checks above.
3. Use the configuration files in `sentinel_rta/configs/` to run simulator experiments.
4. Use the scripts in `sentinel_rta/scripts/` for training, evaluation, temporal-shield stress tests, reward sweeps, and adaptive-attacker experiments.

## Interpretation boundary

The code supports simulator-based research and ablation checking. It does not establish unrestricted robustness under arbitrary traffic, unseen adversaries, or deployed hardware constraints. Operational validation would require real traces, data-plane integration, latency/throughput testing, and operator review.
