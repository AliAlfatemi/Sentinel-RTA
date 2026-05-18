import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to allow importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.ddos_env import DDoSEnv
from shields.rta_shield import RTAShield
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def run_experiment():
    print("Running Sentinel-RTA Minimal Working Version Benchmark (Phase 1 Audit)...")
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'minimal.yaml'))
    config = load_config(config_path)
    
    # Ensure results dir is absolute too
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    config['results_dir'] = results_dir
    os.makedirs(config['results_dir'], exist_ok=True)
    
    env = DDoSEnv(max_steps=config['max_steps'])
    shield = RTAShield()
    
    # Corrected terminology as per audit requirements
    agents = {
        "Random Defender": lambda obs: env.action_space.sample(),
        "Static Threshold Defender": lambda obs: np.array([0.5]),
        "Untrained stochastic policy": lambda obs: np.array([0.9]) if obs[14] > 0.5 else np.array([0.6]),
        "Shield-only policy": lambda obs: shield.repair_action(obs, np.array([1.0]), env.get_shield_context())[0],
        "Shielded stochastic policy": lambda obs: shield.repair_action(obs, np.array([0.9]) if obs[14] > 0.5 else np.array([0.6]), env.get_shield_context())[0]
    }
    
    all_results = []
    
    for seed in config['seeds']:
        print(f"Evaluating Seed: {seed}")
        for agent_name, policy in agents.items():
            obs, _ = env.reset(seed=seed)
            total_sq = 0.0
            total_me = 0.0
            total_violations = 0
            total_repairs = 0
            
            for step in range(env.max_steps):
                raw_action = policy(obs)
                
                # Check for repair
                shield_active = "Shield" in agent_name
                if shield_active:
                    shield_context = env.get_shield_context()
                    final_action, shield_info = shield.repair_action(obs, raw_action, shield_context)
                    if shield_info['total_shield_repair']:
                        total_repairs += 1
                    raw_action = final_action
                    
                obs, reward, done, _, info = env.step(raw_action)
                
                total_sq += info['sq']
                total_me += info['me']
                total_violations += info['sla_violation']
                
                if done: break
                
            all_results.append({
                "Seed": seed,
                "Agent": agent_name,
                "Avg Service Quality (SQ)": total_sq / env.max_steps,
                "Avg Mitigation Efficiency (ME)": total_me / env.max_steps,
                "SLA Violations": total_violations,
                "Shield Repairs": total_repairs
            })
            
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(config['results_dir'], "minimal_seed_summary.csv"), index=False)
    
    # Calculate final summary (mean over seeds)
    final_df = df.groupby("Agent").mean(numeric_only=True).reset_index()
    final_df.to_csv(os.path.join(config['results_dir'], "minimal_final_summary.csv"), index=False)
    
    print("\nFinal Results Summary (Averaged over 5 seeds):")
    print(final_df.to_string())
    
    # Generate Plot from CSV data
    print("\nGenerating evaluation plot from CSV data...")
    plt.figure(figsize=(10, 6))
    x = np.arange(len(final_df))
    width = 0.35
    
    sq_vals = final_df["Avg Service Quality (SQ)"]
    vio_vals = final_df["SLA Violations"]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Defensive Policy')
    ax1.set_ylabel('Service Quality', color=color)
    ax1.bar(x - width/2, sq_vals, width, color=color, label='Service Quality')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.1)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('SLA Violations', color=color)
    ax2.bar(x + width/2, vio_vals, width, color=color, label='SLA Violations')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.xticks(x, final_df["Agent"], rotation=15)
    plt.title("Phase 1 Audit: SQ vs SLA Violations (Mean over 5 seeds)")
    fig.tight_layout()
    plt.savefig(os.path.join(config['results_dir'], "minimal_evaluation_plot.png"), dpi=300)
    plt.close()
    print("Saved plot to results/minimal_evaluation_plot.png")

if __name__ == "__main__":
    run_experiment()
