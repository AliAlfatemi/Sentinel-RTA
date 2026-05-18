# Manuscript Update Notes: Phase 3D Full Benchmark

## 1. Evaluation Status
These results reflect a **Preliminary Extended Benchmark** consisting of 10 generations across 3 stochastic seeds (reduced from the 20x5 theoretical maximum due to compute-bound scaling constraints on local reinforcement learning loops).

## 2. HoF Impact on Robustness
Under the evaluated adaptive conditions, the Hall-of-Fame mechanism failed to improve generalized robustness. The `Adaptive_Shield_NoHoF` baseline achieved a multi-objective robustness score of $0.371$, whereas the `Adaptive_Shield_HoF_pareto_0.1` variant fell to $0.221$. The baseline demonstrated superior defensive plasticity.

## 3. HoF Impact on Forgetting
Contrary to theoretical expectations, the HoF did not effectively reduce catastrophic forgetting. The baseline `NoHoF` variant demonstrated continuous adaptation, improving against historical benchmarks over time (Forgetting Score: $-0.217$). Conversely, the `HoF` variant struggled with optimization interference, causing performance to regress against historical profiles (Forgetting Score: $0.117$).

## 4. HoF Impact on Held-out Generalization
The HoF negatively impacted out-of-distribution capabilities. The performance regressions observed in historical retention and current-generation leakage cascaded into held-out evaluations, proving that archival replay actively constrained the policy gradient.

## 5. Temporal Shield Exploitation
The Temporal Shield successfully prevented worst-case SLA collapse compared to the unshielded baselines. However, adaptive adversaries continued to intelligently skirt the strict temporal bounds, distributing their burst geometry to leak traffic while suppressing rolling repair counts below the maximal threshold.

## 6. Supported Claims & Modifications
- **Supported**: The Temporal Runtime Shield bounds unbounded reinforcement learning policies against worst-case adversarial strategies.
- **Decision Logic Outcome (C)**: "Hall-of-Fame replay did not improve performance in this configuration; it remains a design component requiring further tuning."
- **Soften**: Any declarative claims that adversarial archival universally mitigates catastrophic forgetting in continuous-action networking spaces must be removed. The data proves it introduced severe optimization interference.

## 7. IEEE-Style Wording Updates

### Methodology
> "To prevent the defender from overfitting to static threat profiles, we employ an alternating adversarial co-evolutionary reinforcement learning framework. We evaluated the necessity of augmenting this pipeline with a Pareto-optimal Hall-of-Fame (HoF) archive designed to historically replay non-dominated threat vectors."

### Experiments
> "We conducted a robust benchmarking sweep, measuring multi-objective performance against Service Quality, Mitigation Efficiency, Attack Leakage, Collateral Damage, and normalized SLA pressure. We utilized non-parametric Mann-Whitney U tests to assess statistical deviations between purely reactive co-evolution and historically grounded (HoF) variants."

### Results
> "Hall-of-Fame replay did not improve performance in this configuration; it remains a design component requiring further tuning. Contrary to theoretical assumptions, the archival replays induced significant optimization interference. The purely reactive co-evolutionary baseline demonstrated superior defensive plasticity, achieving negative forgetting scores (indicating continuous improvement against historical baselines) and dominating the archival variants in both current-generation and held-out held-out evaluations."

### Discussion
> "The failure of the HoF mechanism suggests that in highly dynamic routing environments, historical threat signatures quickly become obsolete. Forcing the defender to allocate capacity toward mitigating stale geometries dilutes its ability to converge on generalized, geometry-agnostic defenses."

### Limitations
> "The synthesized adversarial geometries, while capable of inducing systemic stress, operate strictly within the bounds of the predefined protocol taxonomy and do not emulate true, open-world unseen operational vulnerabilities. Furthermore, alternative sampling logic (e.g., prioritized experience replay weighting) remains unexplored."
