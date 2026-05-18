# Sentinel-RTA: Temporal Runtime Assurance for Safe RL-Based DDoS Mitigation

Sentinel-RTA is a simulator-based code and reproducibility repository for temporal runtime assurance in RL-based DDoS mitigation.

## Repository Structure

- `agents/`: RL agent implementation and logic.
- `attackers/`: Simulated adaptive and static DDoS attackers.
- `configs/`: YAML configuration files for experiments.
- `envs/`: Gymnasium simulator environments for network mitigation.
- `metrics/`: Custom metric collection and logging.
- `results/`: Contains the `manuscript_results_package` with final CSVs, tables, and figures.
- `scripts/`: Scripts for running experiments, regenerating tables, and plotting figures.
- `shields/`: Implementation of Instantaneous and Temporal Runtime Assurance (RTA) Shields.
- `tests/`: Automated test suite for the framework.
- `training/`: Core PPO training loops and callbacks.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Smoke Test

Verify the installation and basic framework components by running the test suite:

```bash
pytest sentinel_rta/tests/
```

## Reproducing Manuscript Figures & Tables

The repository includes pre-evaluated logs in `results/manuscript_results_package/` to regenerate the paper's exact figures and tables without re-running expensive training.

To regenerate the primary manuscript figures (Figures 1-6):
```bash
python3 sentinel_rta/scripts/generate_manuscript_result_figures.py
```

To regenerate the final polished Figure 7 (Safety-Performance Trade-off):
```bash
python3 sentinel_rta/scripts/generate_fig7_ieee_v11.py
```

To regenerate the manuscript LaTeX tables:
```bash
python3 sentinel_rta/scripts/generate_manuscript_latex_tables.py
```

## Reproducing Experiments

Experiments are divided by compute requirements:

- **Smoke Tests**: Run `python3 sentinel_rta/scripts/run_minimal.py` to verify the environment pipeline locally in seconds.
- **Preliminary Experiments**: Run `python3 sentinel_rta/scripts/run_temporal_shield_stress.py` to evaluate shield bounds under deterministic settings.
- **Full Expensive Experiments**: Run `python3 sentinel_rta/scripts/run_phase3_coevolution.py` for full adaptive co-evolution. *Warning: this requires significant compute (GPUs recommended) and time.*

> **Result-Status Warning**: The paper reports simulator-based evidence. The adaptive co-evolution experiments are extended-preliminary. Results are bounded strictly to the simulation environment configuration and should not be interpreted as universal production guarantees.

## Citation

```bibtex
@article{alfatemi2025sentinel,
  title={Sentinel-RTA: Temporal Runtime Assurance for Safe Reinforcement-Learning-Based DDoS Mitigation under Adaptive Attackers},
  author={Alfatemi, Ali and others},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

Ali Alfatemi - [AliAlfatemi on GitHub](https://github.com/AliAlfatemi)
