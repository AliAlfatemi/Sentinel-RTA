import os
import pytest

def test_figure_files_updated():
    required_figures = [
        "fig1_architecture",
        "fig2_baseline_comparison",
        "fig3_temporal_stress_summary",
        "fig5_adaptive_leakage_generations",
        "fig5b_final_generation_leakage",
        "fig6_hof_ablation",
        "fig7_safety_performance_tradeoff"
    ]
    
    out_dir = "results/manuscript_results_package/figures"
    
    for fig in required_figures:
        pdf_path = os.path.join(out_dir, f"{fig}.pdf")
        png_path = os.path.join(out_dir, f"{fig}.png")
        assert os.path.exists(pdf_path), f"Missing generated figure: {pdf_path}"
        assert os.path.exists(png_path), f"Missing generated figure: {png_path}"
