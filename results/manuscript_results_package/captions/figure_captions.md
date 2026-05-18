# Figure Captions

## Figure 1: System Architecture
**Caption:** "System Architecture. The environment simulates an adaptive attacker targeting a network service. The defender uses PPO to propose mitigation actions, which are evaluated by an Instantaneous Runtime Assurance shield and a Temporal Runtime Assurance shield using rolling/cumulative SLA context. An optional Hall-of-Fame replay ablation evaluates the effect of historical retention on defender plasticity."

## Figure 2: Baseline Defender Comparison
**Caption:** "Baseline defender comparison. The heuristic baselines reveal complementary failure modes: service-preserving policies such as Adaptive Threshold and Shield-only maintain high service quality but allow high attack leakage, whereas more aggressive thresholding reduces leakage at the cost of increased SLA violations. These results motivate learned mitigation with explicit runtime assurance. Higher service quality is better. Lower attack leakage and fewer SLA violations are better."

## Figure 3: Temporal Runtime Assurance Stress Validation
**Caption:** "Temporal Runtime Assurance stress validation. Compared with no shielding and instantaneous runtime assurance, Temporal RTA reduced cumulative SLA violations in the evaluated stress setting by using rolling safety context to tighten the admissible action bound. This safety improvement increased attack leakage, illustrating the safety–mitigation trade-off in the modeled simulator. Lower SLA violations and lower attack leakage are better."

## Figure 5: Adaptive Attacker Leakage Over Generations
**Caption:** "Adaptive attacker leakage over generations in the limited-seed co-evolution benchmark. Reactive co-evolution with Temporal RTA and no Hall-of-Fame replay achieved lower leakage than the HoF variant in this configuration. Because this benchmark uses 10 generations and 3 seeds, the trend should be interpreted as limited-seed evidence. Lower attack leakage is better."

## Figure 6: Hall-of-Fame Ablation
**Caption:** "Hall-of-Fame replay ablation. In the limited-seed benchmark, naive HoF replay reduced the multi-objective robustness score and produced a positive forgetting score relative to reactive co-evolution without HoF. This indicates optimization interference from historical replay in the tested configuration. Higher robustness is better. Forgetting score is positive when retention worsens and negative when retention improves."

## Figure 7: Safety-Performance Trade-off
**Caption:** "Safety–performance trade-off. The temporal stress and adaptive co-evolution results show that reducing SLA-violation risk can increase attack leakage. Marker size indicates shield repair activity. Results are simulator-specific and should be interpreted as evidence of the trade-off. Lower SLA violation rate and lower attack leakage are better."
