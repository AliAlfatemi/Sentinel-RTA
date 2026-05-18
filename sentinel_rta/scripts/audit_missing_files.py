import os

files_to_check = [
    "results/final_experiment_package/heuristic_baselines_summary.csv",
    "results/final_experiment_package/table1_baseline_comparison.tex",
    "results/final_experiment_package/table2_temporal_stress.tex",
    "results/final_experiment_package/table3_adaptive_attacker.tex",
    "results/final_experiment_package/table4_hof_ablation.tex",
    "results/final_experiment_package/table5_heldout_generalization.tex",
    "results/final_experiment_package/final_manuscript_experiment_notes.md",
    "results/phase2f_temporal_stress/shield_mode_comparison.csv",
    "results/phase2f_temporal_stress/stress_policy_summary.csv",
    "results/phase2f_temporal_stress/temporal_stress_summary.csv",
    "results/phase2f_temporal_stress/per_step_logs.csv",
    "results/phase3d_coevolution_extended_preliminary/final_summary.csv",
    "results/phase3d_coevolution_extended_preliminary/robustness_analysis.csv",
    "results/phase3d_coevolution_extended_preliminary/forgetting_analysis.csv",
    "results/phase3d_coevolution_extended_preliminary/heldout_attacker_evaluation.csv",
    "results/phase3d_coevolution_extended_preliminary/statistical_validation.csv",
    "results/phase3d_coevolution_extended_preliminary/shield_tradeoff_diagnostics.csv",
    "results/final_experiment_package/adaptive_hof/final_summary.csv",
    "results/experiment_improvement_audit/evidence_registry.csv"
]

missing = []
for f in files_to_check:
    if not os.path.exists(f):
        missing.append(f)

# Also check for wildcard exp_*/generation_metrics.csv
import glob
gen_metrics = glob.glob("results/phase3d_coevolution_extended_preliminary/exp_*/generation_metrics.csv")
if len(gen_metrics) == 0:
    missing.append("results/phase3d_coevolution_extended_preliminary/exp_*/generation_metrics.csv")

with open("results/manuscript_results_package/audits/missing_files.md", "w") as out:
    out.write("# Missing Files Audit\n\n")
    if missing:
        out.write("The following files were not found:\n")
        for m in missing:
            out.write(f"- {m}\n")
    else:
        out.write("All required source files were found successfully.\n")

print("Missing files:", missing)
