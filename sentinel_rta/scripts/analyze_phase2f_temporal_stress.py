import pandas as pd
import numpy as np
import scipy.stats as stats
import argparse
import os

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    if n == 0: return 0.0, 0.0, 0.0
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1) if n > 1 else 0.0
    return m, np.std(a), h

def analyze_phase2f(results_dir):
    summary_path = os.path.join(results_dir, "temporal_stress_summary.csv")
    if not os.path.exists(summary_path):
        print("No summary file found.")
        return
        
    df = pd.read_csv(summary_path)
    
    # 1. Shield Mode Comparison (Aggregated across policies and scenarios)
    shield_comparison = df.groupby("shield_mode").agg({
        "cumulative_sla_violation_count": "mean",
        "temporal_shield_repair_count": "mean",
        "instantaneous_shield_repair_count": "mean",
        "attack_leakage": "mean",
        "latency_mean_ms": "mean"
    }).reset_index()
    shield_comparison.to_csv(os.path.join(results_dir, "shield_mode_comparison.csv"), index=False)
    
    # 2. Temporal Repair Activation Table
    # Filter only rows where temporal repairs occurred
    activation_df = df[df["temporal_shield_repair_count"] > 0][["policy_name", "shield_mode", "scenario", "temporal_shield_repair_count", "cumulative_sla_violation_count"]]
    activation_df.to_csv(os.path.join(results_dir, "temporal_repair_activation_table.csv"), index=False)
    
    # 3. Stress Policy Summary
    policy_summary = df.groupby(["policy_name", "shield_mode"]).agg({
        "cumulative_sla_violation_count": "mean",
        "temporal_shield_repair_count": "mean"
    }).reset_index()
    policy_summary.to_csv(os.path.join(results_dir, "stress_policy_summary.csv"), index=False)
    
    # 4. Statistical Validation (Temporal vs Instantaneous SLA Violations)
    stats_results = []
    
    for policy in df["policy_name"].unique():
        for scenario in df["scenario"].unique():
            sub_df = df[(df["policy_name"] == policy) & (df["scenario"] == scenario)]
            inst_df = sub_df[sub_df["shield_mode"] == "Instantaneous Runtime Shield"]
            temp_df = sub_df[sub_df["shield_mode"] == "Temporal Runtime Shield"]
            
            if not inst_df.empty and not temp_df.empty:
                inst_sla = inst_df["cumulative_sla_violation_count"].values
                temp_sla = temp_df["cumulative_sla_violation_count"].values
                
                def safe_mann_whitney(a, b):
                    if len(a) < 2 or len(b) < 2: return 1.0
                    if np.var(a) == 0 and np.var(b) == 0 and a[0] == b[0]: return 1.0
                    try:
                        return stats.mannwhitneyu(a, b, alternative='two-sided')[1]
                    except:
                        return 1.0
                        
                p_val = safe_mann_whitney(temp_sla, inst_sla)
                stats_results.append({
                    "Policy": policy,
                    "Scenario": scenario,
                    "Inst_SLA_Mean": np.mean(inst_sla),
                    "Temp_SLA_Mean": np.mean(temp_sla),
                    "p_value": p_val,
                    "Significant": p_val < 0.05
                })
                
    pd.DataFrame(stats_results).to_csv(os.path.join(results_dir, "statistical_validation.csv"), index=False)
    
    # 5. LaTeX Table
    metrics_to_format = [
        ("cumulative_sla_violation_count", "Cumul. SLA Violations"),
        ("temporal_shield_repair_count", "Temp. Repairs"),
        ("attack_leakage", "Attack Leakage"),
        ("latency_mean_ms", "Latency (ms)")
    ]
    
    latex_lines = [
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\caption{Phase 2F: Temporal Shield Stress Validation Across All Scenarios}",
        "\\label{tab:phase2f_stress}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{l l " + "c" * len(metrics_to_format) + "}",
        "\\toprule",
        "Policy & Shield Mode & " + " & ".join([name for col, name in metrics_to_format]) + " \\\\",
        "\\midrule"
    ]
    
    for policy in sorted(df["policy_name"].unique()):
        for shield in sorted(df["shield_mode"].unique()):
            sub = df[(df["policy_name"] == policy) & (df["shield_mode"] == shield)]
            if sub.empty: continue
            
            row_str = f"{policy} & {shield} "
            for col, _ in metrics_to_format:
                mean, std, _ = mean_confidence_interval(sub[col].dropna())
                if "count" in col:
                    row_str += f"& {mean:.1f} $\\pm$ {std:.1f} "
                else:
                    row_str += f"& {mean:.3f} $\\pm$ {std:.3f} "
            row_str += "\\\\"
            latex_lines.append(row_str)
        latex_lines.append("\\midrule")
        
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table*}"
    ])
    
    with open(os.path.join(results_dir, "main_comparison_table.tex"), 'w') as f:
        f.write("\n".join(latex_lines))
        
    print("Successfully generated Phase 2F analysis.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()
    analyze_phase2f(args.results_dir)
