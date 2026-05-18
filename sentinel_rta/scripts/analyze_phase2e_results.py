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

def analyze_phase2e(results_dir):
    all_steps = []
    policies = ["PPO without Shield", "PPO with Instantaneous Runtime Shield", "PPO with Temporal Runtime Shield", "Random Defender", "Static Threshold Defender", "Shield-only policy", "Untrained stochastic policy"]
    
    for seed_dir in os.listdir(results_dir):
        sdir = os.path.join(results_dir, seed_dir)
        if not os.path.isdir(sdir) or "seed_" not in seed_dir: continue
        seed = int(seed_dir.split("_")[-1])
        
        for d in ["eval_no_shield", "eval_shield"]:
            path = os.path.join(sdir, d, "per_step_logs.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                df["is_attack"] = (df["attack_intensity"] > 0.0).astype(float)
                all_steps.append(df)
                
    if all_steps:
        df_steps = pd.concat(all_steps, ignore_index=True)
        df_steps.to_csv(os.path.join(results_dir, "per_step_logs.csv"), index=False)
        
        # Calculate Action Diagnostics
        action_diagnostics = []
        for policy in df_steps["policy_name"].unique():
            sub_df = df_steps[df_steps["policy_name"] == policy]
            mean_action = sub_df["final_action"].mean()
            action_std = sub_df["final_action"].std()
            frac_below_05 = (sub_df["final_action"] < 0.05).mean()
            frac_above_50 = (sub_df["final_action"] > 0.50).mean()
            frac_above_80 = (sub_df["final_action"] > 0.80).mean()
            
            attack_mask = sub_df["is_attack"] > 0.5
            benign_mask = sub_df["is_attack"] <= 0.5
            
            attack_mean_action = sub_df[attack_mask]["final_action"].mean() if not sub_df[attack_mask].empty else 0.0
            benign_mean_action = sub_df[benign_mask]["final_action"].mean() if not sub_df[benign_mask].empty else 0.0
            
            corr = sub_df["final_action"].corr(sub_df["is_attack"])
            
            action_diagnostics.append({
                "Policy": policy,
                "mean_action": mean_action,
                "action_std": action_std,
                "frac_below_0.05": frac_below_05,
                "frac_above_0.50": frac_above_50,
                "frac_above_0.80": frac_above_80,
                "attack_mean_action": attack_mean_action,
                "benign_mean_action": benign_mean_action,
                "action_attack_correlation": corr
            })
            
        pd.DataFrame(action_diagnostics).to_csv(os.path.join(results_dir, "action_distribution_summary.csv"), index=False)
        
    # Statistical Validation
    summary_path = os.path.join(results_dir, "seed_summary.csv")
    if not os.path.exists(summary_path): return
    df = pd.read_csv(summary_path)
    
    target_policy = "PPO with Temporal Runtime Shield"
    stats_results = []
    
    if target_policy in df["Policy"].values:
        target_sq = df[df["Policy"] == target_policy]["Avg_SQ"].values
        target_sla = df[df["Policy"] == target_policy]["SLA_Violations"].values
        
        for baseline in policies:
            if baseline == target_policy or baseline not in df["Policy"].values: continue
            
            base_sq = df[df["Policy"] == baseline]["Avg_SQ"].values
            base_sla = df[df["Policy"] == baseline]["SLA_Violations"].values
            
            def safe_mann_whitney(a, b):
                if len(a) < 2 or len(b) < 2: return 1.0
                if np.var(a) == 0 and np.var(b) == 0 and a[0] == b[0]: return 1.0
                try:
                    return stats.mannwhitneyu(a, b, alternative='two-sided')[1]
                except:
                    return 1.0
                    
            p_sq = safe_mann_whitney(target_sq, base_sq)
            p_sla = safe_mann_whitney(target_sla, base_sla)
            
            stats_results.append({
                "Comparison": f"{target_policy} vs {baseline}",
                "SQ_p_value": p_sq,
                "SLA_p_value": p_sla,
                "SQ_Significant": p_sq < 0.05,
                "SLA_Significant": p_sla < 0.05
            })
            
    pd.DataFrame(stats_results).to_csv(os.path.join(results_dir, "statistical_validation.csv"), index=False)
    
    # LaTeX Table
    metrics_to_format = [
        ("Avg_SQ", "Service Quality"),
        ("Avg_ME", "Mitig. Efficiency"),
        ("Avg_Leakage", "Attack Leakage"),
        ("SLA_Violations", "Cumulative SLA Violations"),
        ("Instantaneous_Repairs", "Inst. Repairs"),
        ("Temporal_Repairs", "Temp. Repairs"),
        ("Shield_Repairs", "Total Repairs"),
        ("Avg_Total_Control_Latency_ms", "Control Latency (ms)")
    ]
    
    latex_lines = [
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\caption{Phase 2E: Temporal Runtime Assurance vs Baselines (Mean $\\pm$ Std Dev)}",
        "\\label{tab:phase2e_results}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{l" + "c" * len(metrics_to_format) + "}",
        "\\toprule",
        "Defender Policy & " + " & ".join([name for col, name in metrics_to_format]) + " \\\\",
        "\\midrule"
    ]
    
    for policy in policies:
        policy_df = df[df["Policy"] == policy]
        row_str = f"{policy} "
        
        for col, _ in metrics_to_format:
            if col in policy_df.columns:
                mean, std, _ = mean_confidence_interval(policy_df[col].dropna())
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
        "}",
        "\\end{table*}"
    ])
    
    with open(os.path.join(results_dir, "main_comparison_table.tex"), 'w') as f:
        f.write("\n".join(latex_lines))
        
    print("Successfully generated Phase 2E analysis.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()
    analyze_phase2e(args.results_dir)
