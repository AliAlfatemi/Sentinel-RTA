import pandas as pd
import numpy as np
import scipy.stats as stats
import argparse
import os

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, np.std(a), h

def analyze_results(results_dir):
    # Load all seed summaries
    summary_path = os.path.join(results_dir, "seed_summary.csv")
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found.")
        return
        
    df = pd.read_csv(summary_path)
    
    policies = df["Policy"].unique()
    
    # Target policies for statistical comparison
    target_policy = "PPO with Runtime Shield"
    baselines_to_compare = [
        "PPO without Shield",
        "Static Threshold Defender",
        "Shield-only policy",
        "Random Defender"
    ]
    
    # 1. Statistical Validation (Mann-Whitney U on Service Quality and SLA Violations)
    print("--- Statistical Validation (Mann-Whitney U) ---")
    stats_results = []
    
    if target_policy in policies:
        target_sq = df[df["Policy"] == target_policy]["Avg_SQ"].values
        target_sla = df[df["Policy"] == target_policy]["SLA_Violations"].values
        
        for baseline in baselines_to_compare:
            if baseline in policies:
                base_sq = df[df["Policy"] == baseline]["Avg_SQ"].values
                base_sla = df[df["Policy"] == baseline]["SLA_Violations"].values
                
                # Check variance before running tests
                if np.var(target_sq) == 0 and np.var(base_sq) == 0 and target_sq[0] == base_sq[0]:
                    p_sq = 1.0
                else:
                    _, p_sq = stats.mannwhitneyu(target_sq, base_sq, alternative='two-sided')
                    
                if np.var(target_sla) == 0 and np.var(base_sla) == 0 and target_sla[0] == base_sla[0]:
                    p_sla = 1.0
                else:
                    _, p_sla = stats.mannwhitneyu(target_sla, base_sla, alternative='two-sided')
                    
                stats_results.append({
                    "Comparison": f"{target_policy} vs {baseline}",
                    "SQ_p_value": p_sq,
                    "SLA_p_value": p_sla,
                    "SQ_Significant": p_sq < 0.05,
                    "SLA_Significant": p_sla < 0.05
                })
                
                print(f"{target_policy} vs {baseline}:")
                print(f"  SQ p-value:  {p_sq:.4f} (Significant: {p_sq < 0.05})")
                print(f"  SLA p-value: {p_sla:.4f} (Significant: {p_sla < 0.05})")
    
    pd.DataFrame(stats_results).to_csv(os.path.join(results_dir, "statistical_validation.csv"), index=False)
    
    # 2. Generate LaTeX Table
    metrics_to_format = [
        ("Avg_SQ", "Service Quality"),
        ("Avg_ME", "Mitig. Efficiency"),
        ("SLA_Violations", "SLA Viol."),
        ("Safety_Violations", "Safety Viol."),
        ("Shield_Repairs", "Repairs"),
        ("Avg_Total_Control_Latency_ms", "Latency (ms)")
    ]
    
    latex_lines = [
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\caption{Phase 2B: Full Empirical Evaluation of Sentinel-RTA (Mean $\\pm$ Std Dev)}",
        "\\label{tab:phase2_results}",
        "\\begin{tabular}{l" + "c" * len(metrics_to_format) + "}",
        "\\toprule",
        "Defender Policy & " + " & ".join([name for col, name in metrics_to_format]) + " \\\\",
        "\\midrule"
    ]
    
    # Aggregate data with mean and std
    for policy in policies:
        policy_df = df[df["Policy"] == policy]
        row_str = f"{policy} "
        
        for col, _ in metrics_to_format:
            if col in policy_df.columns:
                mean, std, ci = mean_confidence_interval(policy_df[col].dropna())
                if "Violations" in col or "Repairs" in col:
                    row_str += f"& {mean:.1f} $\\pm$ {std:.1f} "
                else:
                    row_str += f"& {mean:.3f} $\\pm$ {std:.3f} "
            else:
                row_str += "& - "
                
        row_str += "\\\\"
        latex_lines.append(row_str)
        
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}"
    ])
    
    latex_path = os.path.join(results_dir, "main_comparison_table.tex")
    with open(latex_path, 'w') as f:
        f.write("\n".join(latex_lines))
        
    print(f"\nSaved LaTeX table to {latex_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()
    analyze_results(args.results_dir)
