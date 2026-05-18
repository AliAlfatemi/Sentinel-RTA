# Sentinel-RTA: Temporal Runtime Assurance for Safe RL-Based DDoS Mitigation

Sentinel-RTA is a simulator-based research artifact for the revised manuscript **“Sentinel-RTA: Temporal Runtime Assurance for Safe Reinforcement-Learning-Based DDoS Mitigation under Adaptive Attackers.”**

The repository contains a Gymnasium-compatible DDoS mitigation simulator, PPO-based defender components, instantaneous and temporal runtime-assurance shields, adaptive-attacker evaluation utilities, curated result CSVs, and scripts for regenerating the manuscript figures and tables. It is intended for simulator-based reproducibility only. It does **not** provide live-network attack tooling, operational DDoS traffic generators, or deployment-ready enforcement code.

## Repository structure

```text
sentinel_rta/
  agents/       heuristic defender baselines
  attackers/    simulated attacker policies and Hall-of-Fame utilities
  configs/      YAML experiment configurations
  envs/         Gymnasium-compatible DDoS mitigation environments
  metrics/      evaluation and scoring utilities
  scripts/      experiment, audit, figure, and table scripts
  shields/      instantaneous and temporal RTA shields
  tests/        automated tests and reproducibility checks
  training/     co-evolution training utilities
results/
  manuscript_results_package/
    source_csv/ curated CSV files used for reported tables and figures
    figures/    generated manuscript figures
    tables/     generated LaTeX tables
    audits/     consistency and missing-file audit outputs
    captions/   figure-caption and claim-map files
docs/
  REPRODUCIBILITY.md
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The artifact was prepared with Python 3.13 in the validation environment, but it is intended to run on Python 3.10 or newer.

## Quick validation

Run these commands from the repository root:

```bash
python -m compileall -q sentinel_rta
pytest -q
python sentinel_rta/scripts/audit_missing_files.py
python sentinel_rta/scripts/audit_result_consistency.py
```

The default pytest configuration excludes long-running training smoke tests. To include long-running tests, use:

```bash
pytest -q -m expensive
```

## Regenerating manuscript figures and tables

```bash
python sentinel_rta/scripts/generate_manuscript_result_figures.py
python sentinel_rta/scripts/generate_fig7_ieee_v11.py
python sentinel_rta/scripts/generate_manuscript_latex_tables.py
```

The scripts read curated CSV files from `results/manuscript_results_package/source_csv/` and write outputs to `results/manuscript_results_package/figures/` and `results/manuscript_results_package/tables/`.

## Scope and ethics

This artifact is for defensive, simulator-based research. It must not be used to generate unauthorized traffic or test third-party systems. The reported results are bounded to the modeled simulator configuration and should not be interpreted as live-network deployment evidence.

## Manuscript code-availability text

```latex
\noindent\textbf{Code Availability:} The Sentinel-RTA simulator, runtime-assurance shields, evaluation scripts, curated result CSVs, and reproducibility artifacts are publicly available at \url{https://github.com/AliAlfatemi/Sentinel-RTA}. The repository is intended for simulator-based reproducibility only and does not provide live-network attack or deployment tools.
```

## Citation

```bibtex
@misc{alfatemi2026sentinelrta,
  title        = {Sentinel-RTA: Temporal Runtime Assurance for Safe Reinforcement-Learning-Based DDoS Mitigation under Adaptive Attackers},
  author       = {Alfatemi, Ali and Alfaqeer, Ahmed and Rahouti, Mohamed and Bhuiyan, Zakirul Alam and Chehri, Abdellah},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/AliAlfatemi/Sentinel-RTA}
}
```

## License

This project is released under the MIT License.
