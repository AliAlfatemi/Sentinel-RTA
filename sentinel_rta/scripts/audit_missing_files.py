from pathlib import Path
import glob

BASE = Path("results/manuscript_results_package")
files = [
    BASE / "ARTIFACT_MANIFEST.md",
    BASE / "source_csv/heuristic_baselines_summary.csv",
    BASE / "source_csv/shield_mode_comparison.csv",
    BASE / "source_csv/final_summary.csv",
    BASE / "source_csv/robustness_analysis.csv",
    BASE / "source_csv/forgetting_analysis.csv",
    BASE / "source_csv/heldout_attacker_evaluation.csv",
    BASE / "tables/table1_baseline_comparison.tex",
    BASE / "tables/table2_temporal_stress.tex",
    BASE / "tables/table3_adaptive_attacker.tex",
    BASE / "tables/table4_hof_ablation.tex",
    BASE / "tables/table5_heldout_generalization.tex",
    BASE / "figures/fig1_architecture.pdf",
    BASE / "figures/fig2_baseline_comparison.pdf",
    BASE / "figures/fig3_temporal_stress_summary.pdf",
    BASE / "figures/fig5_adaptive_leakage_generations.pdf",
    BASE / "figures/fig5b_final_generation_leakage.pdf",
    BASE / "figures/fig6_hof_ablation.pdf",
    BASE / "figures/fig7_safety_performance_tradeoff.pdf",
]
missing = [str(p) for p in files if not p.exists()]
if not glob.glob(str(BASE / "source_csv/phase3d_generation_metrics/exp_*/generation_metrics.csv")):
    missing.append(str(BASE / "source_csv/phase3d_generation_metrics/exp_*/generation_metrics.csv"))
audits = BASE / "audits"
audits.mkdir(parents=True, exist_ok=True)
with open(audits / "missing_files.md", "w", encoding="utf-8") as f:
    f.write("# Missing Files Audit\n\n")
    if missing:
        f.write("The following files were not found:\n")
        for m in missing: f.write(f"- {m}\n")
    else:
        f.write("All required manuscript artifact files were found successfully.\n")
print("Missing files:", missing)
