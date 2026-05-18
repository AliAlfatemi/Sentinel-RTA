import os
import sys
import yaml
import subprocess
import argparse
import pandas as pd

def run_phase2_experiment(config_path, output_dir, seeds, total_timesteps=None, eval_episodes=None):
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths to scripts
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(base_dir, "train_ppo.py")
    eval_script = os.path.join(base_dir, "evaluate_ppo.py")
    run_minimal_script = os.path.join(base_dir, "run_minimal.py")
    plot_script = os.path.join(base_dir, "plot_phase2e_results.py")
    analyze_script = os.path.join(base_dir, "analyze_phase2e_results.py")
    
    all_summaries = []
    
    for seed in seeds:
        print(f"\n======================================")
        print(f"Starting Phase 2 Evaluation for Seed {seed}")
        print(f"======================================\n")
        
        seed_dir = os.path.join(output_dir, f"seed_{seed}")
        
        # 1. Train PPO
        train_cmd = [sys.executable, train_script, "--config", config_path, "--seed", str(seed), "--output_dir", seed_dir]
        if total_timesteps: train_cmd.extend(["--total_timesteps", str(total_timesteps)])
        subprocess.run(train_cmd, check=True)
        
        model_path = os.path.join(seed_dir, "model.zip")
        
        # 2. Evaluate
        # Eval No Shield
        eval_no_dir = os.path.join(seed_dir, "eval_no_shield")
        eval_cmd1 = [sys.executable, eval_script, "--config", config_path, "--model_path", model_path, "--seed", str(seed), "--shield", "off", "--output_dir", eval_no_dir]
        if eval_episodes: eval_cmd1.extend(["--eval_episodes", str(eval_episodes)])
        subprocess.run(eval_cmd1, check=True)
        
        # Eval Instantaneous Shield
        eval_inst_dir = os.path.join(seed_dir, "eval_inst_shield")
        eval_cmd2 = [sys.executable, eval_script, "--config", config_path, "--model_path", model_path, "--seed", str(seed), "--shield", "on", "--temporal", "off", "--output_dir", eval_inst_dir]
        if eval_episodes: eval_cmd2.extend(["--eval_episodes", str(eval_episodes)])
        subprocess.run(eval_cmd2, check=True)
        
        # Eval Temporal Shield
        eval_temp_dir = os.path.join(seed_dir, "eval_shield")
        eval_cmd3 = [sys.executable, eval_script, "--config", config_path, "--model_path", model_path, "--seed", str(seed), "--shield", "on", "--temporal", "on", "--output_dir", eval_temp_dir]
        if eval_episodes: eval_cmd3.extend(["--eval_episodes", str(eval_episodes)])
        subprocess.run(eval_cmd3, check=True)
        
        for d in [eval_no_dir, eval_inst_dir, eval_temp_dir]:
            df_sum = pd.read_csv(os.path.join(d, "episode_summary.csv"))
            all_summaries.append(df_sum)
        
    # Run Baselines (Random, Static, Untrained, Shield-Only) via run_minimal.py
    print("\nRunning Baselines via run_minimal.py...")
    subprocess.run([sys.executable, run_minimal_script], check=True)
    df_baselines = pd.read_csv(os.path.join(output_dir, "..", "minimal_final_summary.csv"))
    # Rename columns to match PPO summary
    df_baselines = df_baselines.rename(columns={
        "Agent": "Policy",
        "Avg Service Quality (SQ)": "Avg_SQ",
        "Avg Mitigation Efficiency (ME)": "Avg_ME",
        "SLA Violations": "SLA_Violations",
        "Shield Repairs": "Shield_Repairs",
        "Avg Attack Leakage": "Avg_Leakage",
        "Avg Collateral Damage": "Avg_Collateral"
    })
    # Add dummy latency columns for baselines
    df_baselines["Avg_Inference_Latency_ms"] = 0.0
    df_baselines["Avg_Shield_Latency_ms"] = 0.0
    df_baselines["Avg_Total_Control_Latency_ms"] = 0.0
    
    # Aggregate and save final summaries
    final_df = pd.concat([pd.concat(all_summaries, ignore_index=True), df_baselines], ignore_index=True)
    final_df.to_csv(os.path.join(output_dir, "seed_summary.csv"), index=False)
    
    mean_df = final_df.groupby("Policy").mean(numeric_only=True).reset_index()
    mean_df.to_csv(os.path.join(output_dir, "final_summary.csv"), index=False)
    
    print("\nPhase 2 Complete. Final Aggregated Results:")
    print(mean_df.to_string())
    
    # Run Analysis and Plotting
    print("\nGenerating Plots and Statistical Analysis...")
    subprocess.run([sys.executable, plot_script, "--results_dir", output_dir], check=True)
    subprocess.run([sys.executable, analyze_script, "--results_dir", output_dir], check=True)
    print("\nFull Experiment Pipeline Finished Successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs='+', required=True)
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--evaluation_episodes", type=int, default=None)
    args = parser.parse_args()
    
    run_phase2_experiment(args.config, args.output_dir, args.seeds, args.total_timesteps, args.evaluation_episodes)
