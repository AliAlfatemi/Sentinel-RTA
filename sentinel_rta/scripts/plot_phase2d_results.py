import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def plot_bar_chart(df, metric_col, title, ylabel, filename, results_dir, color='tab:blue'):
    if metric_col not in df.columns: return
    plt.figure(figsize=(10, 6))
    x = np.arange(len(df))
    plt.bar(x, df[metric_col], color=color)
    plt.xticks(x, df["Policy"], rotation=20, ha='right')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, filename), dpi=300)
    plt.close()

def plot_phase2d_results(results_dir):
    summary_path = os.path.join(results_dir, "final_summary.csv")
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found.")
        return
        
    df = pd.read_csv(summary_path)
    
    plot_bar_chart(df, "Avg_SQ", "Average Service Quality by Policy", "Service Quality", "service_quality_plot.png", results_dir, 'mediumseagreen')
    plot_bar_chart(df, "Avg_ME", "Mitigation Efficiency by Policy", "Mitigation Efficiency", "mitigation_efficiency_plot.png", results_dir, 'steelblue')
    plot_bar_chart(df, "Avg_Leakage", "Attack Leakage by Policy", "Attack Leakage", "attack_leakage_plot.png", results_dir, 'darkred')
    plot_bar_chart(df, "Avg_Collateral", "Collateral Damage by Policy", "Collateral Damage", "collateral_damage_plot.png", results_dir, 'darkorange')
    plot_bar_chart(df, "Safety_Violations", "Safety Violations by Policy", "Total Violations", "safety_violations_plot.png", results_dir, 'crimson')
    plot_bar_chart(df, "Shield_Repairs", "Shield Repairs by Policy", "Total Repairs", "shield_repairs_plot.png", results_dir, 'purple')
    plot_bar_chart(df, "Avg_Total_Control_Latency_ms", "Total Control Latency (ms)", "Latency (ms)", "control_latency_plot.png", results_dir, 'goldenrod')

    # Action distribution plot
    actions_path = os.path.join(results_dir, "action_distribution_summary.csv")
    if os.path.exists(actions_path):
        df_act = pd.read_csv(actions_path)
        
        plt.figure(figsize=(12, 6))
        width = 0.35
        x_act = np.arange(len(df_act["Policy"]))
        
        plt.bar(x_act - width/2, df_act["benign_mean_action"], width, label='Benign Mean Action', color='lightblue')
        plt.bar(x_act + width/2, df_act["attack_mean_action"], width, label='Attack Mean Action', color='salmon')
        
        plt.xticks(x_act, df_act["Policy"], rotation=20, ha='right')
        plt.ylabel('Mean Action (Drop Intensity)')
        plt.title('Action Distribution: Benign vs Attack Periods')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "action_distribution_plot.png"), dpi=300)
        plt.close()

    # Reward Component Breakdown
    per_step_path = os.path.join(results_dir, "per_step_logs.csv")
    if os.path.exists(per_step_path):
        df_steps = pd.read_csv(per_step_path)
        ppo_base = df_steps[df_steps["policy_name"] == "PPO without Shield"]
        if not ppo_base.empty and "reward_service_quality" in ppo_base.columns:
            components = {
                "SQ Reward": ppo_base["reward_service_quality"].mean(),
                "Mitigation Reward": ppo_base["reward_mitigation"].mean(),
                "Leakage Penalty": ppo_base["penalty_attack_leakage"].mean(),
                "Collateral Penalty": ppo_base["penalty_collateral_damage"].mean(),
                "SLA Penalty": ppo_base["penalty_sla_violation"].mean(),
            }
            
            # Save to CSV as well
            pd.DataFrame([components]).to_csv(os.path.join(results_dir, "reward_component_summary.csv"), index=False)
            
            plt.figure(figsize=(8, 5))
            plt.bar(components.keys(), components.values(), color=['green', 'blue', 'red', 'orange', 'darkred'])
            plt.title("Reward Component Breakdown (PPO without Shield)")
            plt.ylabel("Average Value per Step")
            plt.xticks(rotation=15)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "reward_component_breakdown.png"), dpi=300)
            plt.close()

    print("Successfully generated all Phase 2D empirical plots.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()
    plot_phase2d_results(args.results_dir)
