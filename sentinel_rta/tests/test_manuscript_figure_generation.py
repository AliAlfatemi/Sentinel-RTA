import os
import pytest

def test_manuscript_figure_generation():
    required_figures = [
        "fig2_baseline_comparison.pdf", "fig2_baseline_comparison.png",
        "fig3_temporal_stress_summary.pdf", "fig3_temporal_stress_summary.png",
        "fig5_adaptive_leakage_generations.pdf", "fig5_adaptive_leakage_generations.png",
        "fig6_hof_ablation.pdf", "fig6_hof_ablation.png",
        "fig7_safety_performance_tradeoff.pdf", "fig7_safety_performance_tradeoff.png"
    ]
    
    out_dir = "results/manuscript_results_package/figures"
    for fig in required_figures:
        assert os.path.exists(os.path.join(out_dir, fig)), f"Missing generated figure: {fig}"
