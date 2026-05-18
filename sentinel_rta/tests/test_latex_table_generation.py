import os
import pytest

def test_latex_table_generation():
    tables = [
        "table1_baseline_comparison.tex",
        "table2_temporal_stress.tex",
        "table3_adaptive_attacker.tex",
        "table4_hof_ablation.tex",
        "table5_heldout_generalization.tex"
    ]
    out_dir = "results/manuscript_results_package/tables"
    for t in tables:
        path = os.path.join(out_dir, t)
        assert os.path.exists(path), f"Table {t} missing."
        assert os.path.getsize(path) > 10, f"Table {t} is empty."
