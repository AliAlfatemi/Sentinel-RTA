# Figure Selection Report

- **Figure 1**: Architecture Schematic.
  - **Purpose**: Explain the interaction between the defender, instantaneous and temporal shields, and adaptive attacker evaluation.
  - **Source**: Conceptual/Diagram (Not auto-generated here, user supplies).
  - **Status**: Kept (requires external diagram generation).

- **Figure 2**: Baseline Defender Comparison.
  - **Purpose**: Establish operational extremes.
  - **Source**: `results/final_experiment_package/heuristic_baselines_summary.csv`
  - **Section**: Baseline Defender Comparison
  - **Status**: Regenerated
  - **Caption**: "Baseline defender comparison. The heuristic baselines reveal complementary failure modes: service-preserving policies such as Adaptive Threshold and Shield-only maintain high service quality but allow high attack leakage, whereas more aggressive thresholding reduces leakage at the cost of increased SLA violations. These results motivate learned mitigation with explicit runtime assurance."
  - **Limitation**: Evaluated strictly on packet-level generic routing metrics.

- **Figure 3**: Temporal Runtime Assurance Stress Validation.
  - **Purpose**: Show reduced cumulative SLA violations under sustained edge-riding.
  - **Source**: `results/phase2f_temporal_stress/shield_mode_comparison.csv`
  - **Section**: Temporal Runtime Assurance Stress Validation
  - **Status**: Regenerated
  - **Caption**: "Temporal Runtime Assurance stress validation. Compared with no shielding and instantaneous runtime assurance, Temporal RTA reduced cumulative SLA violations in the evaluated stress setting by using rolling safety context to tighten the admissible action bound. This safety improvement increased attack leakage, illustrating the safety-mitigation trade-off in the modeled simulator."
  - **Limitation**: Evaluated using the aggressive deterministic edge-riding policy.

- **Figure 4**: Temporal Shield Time-Series.
  - **Status**: Skipped (Omitted due to 60MB file size footprint per log).

- **Figure 5**: Adaptive Attacker Leakage Over Generations.
  - **Purpose**: Visualize performance trends under co-evolutionary scaling.
  - **Source**: `results/phase3d_coevolution_limited_seed/exp_*/generation_metrics.csv`
  - **Section**: Adaptive Attacker Evaluation
  - **Status**: Regenerated
  - **Caption**: "Adaptive attacker leakage over generations in the limited-seed co-evolution benchmark. Reactive co-evolution with Temporal RTA and no Hall-of-Fame replay achieved lower leakage than the HoF variant in this configuration. Because this benchmark uses 10 generations and 3 seeds, the trend should be interpreted as limited-seed evidence."
  - **Limitation**: Limited-seed map constrained to 10 generations and 3 seeds.

- **Figure 5b**: Final Generation Leakage.
  - **Status**: Regenerated (Optional bar chart visualization of final generation bounds).

- **Figure 6**: Hall-of-Fame Ablation.
  - **Purpose**: Support HoF optimization interference interpretation.
  - **Source**: `results/phase3d_coevolution_limited_seed/robustness_analysis.csv`
  - **Section**: Hall-of-Fame Replay Ablation
  - **Status**: Regenerated
  - **Caption**: "Hall-of-Fame replay ablation. In the limited-seed benchmark, naive HoF replay reduced the multi-objective robustness score and produced a positive forgetting score relative to reactive co-evolution without HoF. This indicates optimization interference from historical replay in the tested configuration."
  - **Limitation**: Based solely on Pareto-admission criteria.

- **Figure 7**: Safety-Performance Trade-off.
  - **Purpose**: Map all results onto the SLA vs Leakage continuum.
  - **Source**: `phase3d_coevolution_limited_seed/final_summary.csv`
  - **Section**: Safety-Performance Trade-off Summary
  - **Status**: Regenerated
  - **Caption**: "Safety-performance trade-off. The temporal stress and adaptive co-evolution results show that reducing SLA-violation risk can increase attack leakage. Marker size indicates shield repair activity. Results are simulator-specific and should be interpreted as evidence of the trade-off."
  - **Limitation**: Demonstrates theoretical metric limits rather than deployment readiness.
