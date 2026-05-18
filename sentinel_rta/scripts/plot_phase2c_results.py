import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def plot_results(results_dir):
    summary_path = os.path.join(results_dir, "reward_sweep_summary.csv")
    if not os.path.exists(summary_path): return
    df = pd.read_csv(summary_path)
    
    # Filter for just PPO without shield to compare reward variants
    df_ppo = df[df["Policy"] == "PPO without Shield"]
    
    variants = df_ppo["Variant"].values
    x = np.arange(len(variants))
    
    def plot_metric(metric, title, ylabel, filename, color):
        plt.figure(figsize=(8, 5))
        plt.bar(x, df_ppo[metric], color=color)
        plt.xticks(x, variants)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, filename), dpi=300)
        plt.close()
        
    plot_metric("Avg_SQ", "Reward Variant vs Service Quality", "SQ", "sweep_sq_plot.png", "mediumseagreen")
    plot_metric("Avg_ME", "Reward Variant vs Mitigation Efficiency", "ME", "sweep_me_plot.png", "steelblue")
    plot_metric("Avg_Leakage", "Reward Variant vs Attack Leakage", "Leakage", "sweep_leakage_plot.png", "darkred")
    plot_metric("Avg_Collateral", "Reward Variant vs Collateral Damage", "Collateral Damage", "sweep_cd_plot.png", "darkorange")
    plot_metric("SLA_Violations", "Reward Variant vs SLA Violations", "SLA Violations", "sweep_sla_plot.png", "crimson")
    
    # Read action diagnostics
    actions_path = os.path.join(results_dir, "action_distribution_summary.csv")
    if os.path.exists(actions_path):
        df_act = pd.read_csv(actions_path)
        df_act_ppo = df_act[df_act["Policy"] == "PPO without Shield"]
        
        plt.figure(figsize=(10, 6))
        width = 0.35
        x_act = np.arange(len(df_act_ppo["Variant"]))
        
        plt.bar(x_act - width/2, df_act_ppo["benign_mean_action"], width, label='Benign Mean Action', color='lightblue')
        plt.bar(x_act + width/2, df_act_ppo["attack_mean_action"], width, label='Attack Mean Action', color='salmon')
        
        plt.xticks(x_act, df_act_ppo["Variant"])
        plt.ylabel('Mean Action (Drop Intensity)')
        plt.title('PPO Action Distribution: Benign vs Attack Periods')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "sweep_action_dist_plot.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()
    plot_results(args.results_dir)
