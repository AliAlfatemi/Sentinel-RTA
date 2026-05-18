import os
import sys
import yaml
import subprocess
import argparse
import pandas as pd

def run_reward_sweep(config_path, output_dir, seeds, total_timesteps=None, eval_episodes=None, variants=None):
    os.makedirs(output_dir, exist_ok=True)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(base_dir, "train_ppo.py")
    eval_script = os.path.join(base_dir, "evaluate_ppo.py")
    analyze_script = os.path.join(base_dir, "analyze_phase2c_results.py")
    plot_script = os.path.join(base_dir, "plot_phase2c_results.py")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    all_variants = variants if variants else list(config['reward_variants'].keys())
    
    all_summaries = []
    all_steps = []
    
    for variant in all_variants:
        variant_config = config.copy()
        # Override env weights in memory before saving a temp config
        variant_config['env']['reward_mode'] = config['reward_variants'][variant]['mode']
        variant_config['env']['reward_weights'] = config['reward_variants'][variant]['weights']
        
        var_config_path = os.path.join(output_dir, f"temp_config_{variant}.yaml")
        with open(var_config_path, 'w') as f:
            yaml.dump(variant_config, f)
            
        print(f"\n=== Running Sweep for Variant: {variant} ===")
        
        for seed in seeds:
            seed_dir = os.path.join(output_dir, f"variant_{variant}_seed_{seed}")
            
            # Train
            train_cmd = [sys.executable, train_script, "--config", var_config_path, "--seed", str(seed), "--output_dir", seed_dir]
            if total_timesteps: train_cmd.extend(["--total_timesteps", str(total_timesteps)])
            subprocess.run(train_cmd, check=True)
            
            model_path = os.path.join(seed_dir, "model.zip")
            
            # Eval No Shield
            eval_no_dir = os.path.join(seed_dir, "eval_no_shield")
            eval_cmd1 = [sys.executable, eval_script, "--config", var_config_path, "--model_path", model_path, "--seed", str(seed), "--shield", "off", "--output_dir", eval_no_dir]
            if eval_episodes: eval_cmd1.extend(["--eval_episodes", str(eval_episodes)])
            subprocess.run(eval_cmd1, check=True)
            
            # Eval Shield
            eval_yes_dir = os.path.join(seed_dir, "eval_shield")
            eval_cmd2 = [sys.executable, eval_script, "--config", var_config_path, "--model_path", model_path, "--seed", str(seed), "--shield", "on", "--output_dir", eval_yes_dir]
            if eval_episodes: eval_cmd2.extend(["--eval_episodes", str(eval_episodes)])
            subprocess.run(eval_cmd2, check=True)
            
            # Collect Summaries
            for d in [eval_no_dir, eval_yes_dir]:
                df_sum = pd.read_csv(os.path.join(d, "episode_summary.csv"))
                df_sum["Variant"] = variant
                
                # Add extra columns missing from base evaluate_ppo to match expectations
                df_steps_local = pd.read_csv(os.path.join(d, "per_step_logs.csv"))
                df_sum["Avg_Collateral"] = df_steps_local["collateral_damage"].mean()
                df_sum["Avg_Leakage"] = df_steps_local["attack_leakage"].mean()
                df_sum["Avg_Total_Latency_ms"] = df_steps_local["total_control_latency_ms"].mean()
                
                all_summaries.append(df_sum)
                
                # Collect Steps
                df_steps_local["Variant"] = variant
                # We need is_attack explicitly logic since we only logged attack_intensity. 
                df_steps_local["is_attack"] = (df_steps_local["attack_intensity"] > 0.0).astype(float)
                all_steps.append(df_steps_local)
                
    # Aggregate
    df_final_sum = pd.concat(all_summaries, ignore_index=True)
    df_final_sum.to_csv(os.path.join(output_dir, "reward_sweep_seed_summary.csv"), index=False)
    
    mean_df = df_final_sum.groupby(["Variant", "Policy"]).mean(numeric_only=True).reset_index()
    mean_df.to_csv(os.path.join(output_dir, "reward_sweep_summary.csv"), index=False)
    
    pd.concat(all_steps, ignore_index=True).to_csv(os.path.join(output_dir, "reward_sweep_per_step_logs.csv"), index=False)
    
    print("\nReward Sweep Execution Complete. Running Analysis...")
    subprocess.run([sys.executable, analyze_script, "--results_dir", output_dir], check=True)
    subprocess.run([sys.executable, plot_script, "--results_dir", output_dir], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs='+', required=True)
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--evaluation_episodes", type=int, default=None)
    parser.add_argument("--variants", type=str, nargs='*', default=None)
    args = parser.parse_args()
    
    run_reward_sweep(args.config, args.output_dir, args.seeds, args.total_timesteps, args.evaluation_episodes, args.variants)
