# Final Manuscript Experiment Notes

## Evaluation Status
- **Manuscript-Ready:** Temporal Stress Validation (Phase 2F), Heuristic Baselines.
- **Limited-seed:** Phase 3D Co-evolution (10 generations instead of 20), Adaptive HoF Diagnostic.

## Claim Adjudication
- **Supported:** Temporal Runtime Assurance successfully bounds rolling/cumulative SLA risk under adaptive stress. Temporal shielding establishes a controllable attack-leakage trade-off. Reactive co-evolution improves bounding geometry plasticity without historical constraints.
- **Partially Supported:** PPO with temporal shielding bounds safe mitigation relative to unshielded models. 
- **Contradicted:** Naive Hall-of-Fame (HoF) replay reduces forgetting and improves robustness. (Empirically, it caused optimization interference).
- **Unsupported:** Zero-day robustness, open-world threat neutrality, hardware-deployment readiness.

## Exact Rewrite Directives
### 1. Abstract
Remove any claims that Sentinel-RTA achieves "zero safety violations" or mitigates "open-world unseen operational attacks." Replace with: "Sentinel-RTA bounds mitigation geometry through a Temporal Runtime Assurance shield, mathematically limiting rolling SLA degradation under modeled adaptive stress."

### 2. Contributions
Replace the 5th contribution with: "An empirical ablation study on historical threat retention (Hall-of-Fame replay), demonstrating that naive archival constraints induce optimization interference in continuous-action networking spaces."

### 3. Experiments Section
**Add:** 
> "Our evaluation pipeline includes four purely heuristic baselines—Random Defender, Static Threshold, Adaptive Threshold, and Shield-Only—to ground the deep learning policies. We evaluated adaptive attenuation over a bounded MultiDiscrete threat space within a controlled episodic simulator."

### 4. Results Section
**Add:** 
> "Table 1 establishes the operational bounds of standard heuristics, where Adaptive Thresholding achieves acceptable but rigid Service Quality. Table 2 details the Temporal Stress validation, proving unshielded models collapse under sustained edge-riding bursts, while Temporal Runtime Assurance forcibly clamps cumulative SLA violations. Table 3 details the final co-evolution bounds, where reactive learning without archival constraints (NoHoF) outperformed historically anchored variants by avoiding optimization interference against out-of-distribution shifts."

### 5. Limitations Section
**Add:** 
> "The findings herein reflect bounded execution inside an aggregated traffic simulator, strictly mapping parameter vectors rather than deep packet inspection. Furthermore, while the neuro-symbolic bounding effectively halts mathematical SLA breaches, it does not infer root-cause protocol exploitation."
