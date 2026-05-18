import pandas as pd
import numpy as np
import scipy.stats as stats
import argparse
import os

def is_passive_policy(mean_action, attack_period_mean_action, attack_leakage):
    """
    Diagnostics that detect policy collapse.
    A policy is considered passive if:
    - mean_action < 0.05, and
    - attack_period_mean_action < 0.10, and
    - attack_leakage remains high (> 0.20)
    """
    return (mean_action < 0.05) and (attack_period_mean_action < 0.10) and (attack_leakage > 0.20)

def calculate_selection_score(sq, me, cd, leak, sla, lat):
    """
    Multi-objective score.
    """
    # Normalize SLA: 500 max expected
    norm_sla = min(1.0, sla / 500.0)
    # Normalize Latency: say 2.0 ms is max expected
    norm_lat = min(1.0, lat / 2.0)
    
    score = (0.30 * sq) + (0.30 * me) - (0.20 * cd) - (0.10 * leak) - (0.05 * norm_sla) - (0.05 * norm_lat)
    return score

def analyze_sweep(results_dir):
    per_step_path = os.path.join(results_dir, "reward_sweep_per_step_logs.csv")
    if not os.path.exists(per_step_path):
        print("Error: Sweep per-step logs not found.")
        return
        
    df_steps = pd.read_csv(per_step_path)
    
    # Calculate Action Diagnostics
    action_diagnostics = []
    
    for variant in df_steps["Variant"].unique():
        for policy in ["PPO without Shield", "PPO with Runtime Shield"]:
            mask = (df_steps["Variant"] == variant) & (df_steps["policy_name"] == policy)
            sub_df = df_steps[mask]
            if sub_df.empty: continue
            
            mean_action = sub_df["final_action"].mean()
            action_std = sub_df["final_action"].std()
            
            frac_below_05 = (sub_df["final_action"] < 0.05).mean()
            frac_above_50 = (sub_df["final_action"] > 0.50).mean()
            frac_above_80 = (sub_df["final_action"] > 0.80).mean()
            
            attack_mask = sub_df["is_attack"] > 0.5
            benign_mask = sub_df["is_attack"] <= 0.5
            
            attack_mean_action = sub_df[attack_mask]["final_action"].mean() if not sub_df[attack_mask].empty else 0.0
            benign_mean_action = sub_df[benign_mask]["final_action"].mean() if not sub_df[benign_mask].empty else 0.0
            
            attack_leakage = sub_df["attack_leakage"].mean()
            
            is_passive = is_passive_policy(mean_action, attack_mean_action, attack_leakage)
            
            action_diagnostics.append({
                "Variant": variant,
                "Policy": policy,
                "mean_action": mean_action,
                "action_std": action_std,
                "frac_below_0.05": frac_below_05,
                "frac_above_0.50": frac_above_50,
                "frac_above_0.80": frac_above_80,
                "attack_mean_action": attack_mean_action,
                "benign_mean_action": benign_mean_action,
                "attack_leakage": attack_leakage,
                "is_passive": is_passive
            })
            
    df_actions = pd.DataFrame(action_diagnostics)
    df_actions.to_csv(os.path.join(results_dir, "action_distribution_summary.csv"), index=False)
    
    # Calculate Multi-Objective Score and Best Config
    summary_path = os.path.join(results_dir, "reward_sweep_summary.csv")
    df_summary = pd.read_csv(summary_path)
    
    best_score = -float('inf')
    best_variant = None
    
    for idx, row in df_summary.iterrows():
        # We evaluate the shield policy for the final score, or average them. Let's use Shielded.
        if row["Policy"] == "PPO without Shield": 
            # We specifically want to score the base PPO to ensure it learned properly
            score = calculate_selection_score(
                sq=row["Avg_SQ"],
                me=row["Avg_ME"],
                cd=row["Avg_Collateral"],
                leak=row["Avg_Leakage"],
                sla=row["SLA_Violations"],
                lat=row["Avg_Total_Latency_ms"]
            )
            df_summary.at[idx, "Selection_Score"] = score
            
            if score > best_score:
                best_score = score
                best_variant = row["Variant"]
                
    df_summary.to_csv(os.path.join(results_dir, "reward_sweep_scored_summary.csv"), index=False)
    
    print(f"\nBest Reward Configuration Selected: {best_variant} (Score: {best_score:.3f})")
    with open(os.path.join(results_dir, "best_reward_config.yaml"), "w") as f:
        f.write(f"best_variant: {best_variant}\nscore: {best_score}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()
    analyze_sweep(args.results_dir)
