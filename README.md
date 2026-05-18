# Sentinel-RTA Code Repository

This is the **code-only** repository for **Sentinel-RTA: Temporal Runtime Assurance for Safe Reinforcement-Learning-Based DDoS Mitigation under Adaptive Attackers**.

The repository contains the simulator and implementation code for the Sentinel-RTA framework: Gymnasium-compatible DDoS mitigation environments, PPO-compatible defender components, heuristic baselines, instantaneous and temporal runtime-assurance shields, adaptive-attacker utilities, Hall-of-Fame utilities, training/evaluation scripts, configuration files, and automated tests.

This repository is intended for **simulator-based defensive research only**. It does **not** include manuscript PDFs, LaTeX source, precomputed manuscript figures/tables, real traffic traces, live-network attack tools, operational DDoS traffic generators, or deployment-ready enforcement code.

## Repository structure

```text
sentinel_rta/
  agents/       heuristic defender baselines
  attackers/    simulated attacker policies and Hall-of-Fame utilities
  configs/      YAML experiment configurations
  envs/         Gymnasium-compatible DDoS mitigation environments
  metrics/      evaluation and scoring utilities
  scripts/      training, evaluation, analysis, and plotting scripts
  shields/      instantaneous and temporal RTA shields
  tests/        automated code tests
  training/     co-evolution training utilities
docs/
  REPRODUCIBILITY.md
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The code was validated with Python 3.13 in the preparation environment and is intended to run on Python 3.10 or newer.

## Quick validation

Run these commands from the repository root:

```bash
python -m compileall -q sentinel_rta
pytest -q
```

The default pytest configuration excludes long-running experiment smoke tests. To include long-running tests, use:

```bash
pytest -q -m expensive
```

## Example scripts

```bash
python sentinel_rta/scripts/run_minimal.py
python sentinel_rta/scripts/train_ppo.py
python sentinel_rta/scripts/evaluate_ppo.py
python sentinel_rta/scripts/run_temporal_shield_stress.py
python sentinel_rta/scripts/run_phase3_coevolution.py
python sentinel_rta/scripts/run_phase3c_hof_sweep.py
python sentinel_rta/scripts/run_reward_sweep.py
```

Some scripts write outputs to local result folders created at runtime. Precomputed manuscript result CSVs, figures, and tables are intentionally not included in this code-only release.

## Scope and ethics

This repository is for defensive, simulator-based research. It must not be used to generate unauthorized traffic, attack third-party systems, or test systems without permission. The code models DDoS mitigation decisions inside a simulator and should not be interpreted as live-network deployment evidence.

## Manuscript code-availability text

```latex
\noindent\textbf{Code Availability:}
The Sentinel-RTA simulator, runtime-assurance shields, defender/attacker components, training scripts, evaluation scripts, and configuration files are publicly available at
\url{https://github.com/AliAlfatemi/Sentinel-RTA}.
The repository is intended for simulator-based defensive research only and does not provide live-network attack or deployment tools.
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
