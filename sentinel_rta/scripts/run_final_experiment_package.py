import os
import yaml
import argparse
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.ddos_coevolution_env import CoEvolutionDDoSEnv
from attackers.policies import HeuristicAdaptiveAttackerPolicy, RandomAttackerPolicy
from agents.baselines import RandomDefender, StaticThresholdDefender, AdaptiveThresholdDefender, ShieldOnlyPolicy
from training.coevolution_trainer import evaluate_coevolution
from shields.rta_shield import RTAShield

def run_heuristic_baselines(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open('configs/coevolution_final.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('configs/baselines.yaml', 'r') as f:
        baseline_cfg = yaml.safe_load(f)
        
    env = CoEvolutionDDoSEnv(
        max_steps=500,
        defender_reward_config=config['defender']['reward_weights'],
        attacker_reward_config=config['attacker']['reward_weights']
    )
    
    shield = RTAShield(
        max_collateral_tolerance=config['shield']['max_drop_without_confidence'],
        temporal_enabled=config['shield']['temporal_enabled'],
        temporal_recovery_max_action=config['shield'].get('temporal_recovery_max_action', 0.05)
    )
    
    attacker = HeuristicAdaptiveAttackerPolicy()
    
    from gymnasium import spaces
    def_action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    
    baselines = {
        'Random_Defender': RandomDefender(def_action_space, seed=baseline_cfg['random']['seed']),
        'Static_Threshold': StaticThresholdDefender(def_action_space, **baseline_cfg['static_threshold']),
        'Adaptive_Threshold': AdaptiveThresholdDefender(def_action_space, **baseline_cfg['adaptive_threshold']),
    }
    
    baselines['Shield_Only'] = ShieldOnlyPolicy(baselines['Adaptive_Threshold'], shield)
    
    results = []
    
    for name, policy in baselines.items():
        print(f"Evaluating Baseline: {name}")
        for seed in [1, 2, 3]:
            # For heuristic baselines, since they don't learn across episodes, we just evaluate
            # We will use evaluate_coevolution, but since evaluate_coevolution steps the shield explicitly,
            # for Shield_Only we should pass the shield, for others pass None.
            use_shield = shield if name == 'Shield_Only' else None
            df_eval = evaluate_coevolution(env, policy, attacker, use_shield, episodes=5)
            
            sq = df_eval['service_quality'].mean()
            leak = df_eval['attack_leakage'].mean()
            cd = df_eval['collateral_damage'].mean()
            sla = df_eval['sla_violation'].sum() / 5.0
            repairs = df_eval['shield_repair'].sum() / 5.0 if 'shield_repair' in df_eval else 0.0
            
            results.append({
                'Experiment': name,
                'Seed': seed,
                'Final_SQ_Mean': sq,
                'Final_Leakage_Mean': leak,
                'sla_violation_count': sla,
                'Final_Shield_Repairs': repairs,
                'collateral_damage': cd
            })
            
            # Save raw metrics
            exp_dir = os.path.join(output_dir, f"exp_{name}_seed_{seed}")
            os.makedirs(exp_dir, exist_ok=True)
            df_eval.to_csv(os.path.join(exp_dir, 'evaluation_metrics.csv'), index=False)
            
            # Heldout evaluation
            from gymnasium import spaces
            old_attacker = RandomAttackerPolicy(spaces.MultiDiscrete([4, 5, 5, 3, 3]))
            df_held = evaluate_coevolution(env, policy, old_attacker, use_shield, episodes=5)
            df_held.to_csv(os.path.join(exp_dir, 'heldout_attacker_evaluation.csv'), index=False)
            
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(output_dir, 'heuristic_baselines_summary.csv'), index=False)
    
if __name__ == "__main__":
    out_dir = "results/final_experiment_package"
    print("Running Heuristic Baselines...")
    run_heuristic_baselines(out_dir)
    
    print("Running Adaptive HoF Diagnostic...")
    os.system("python scripts/run_phase3_coevolution.py "
              "--config configs/coevolution_adaptive_hof.yaml "
              "--generations 5 "
              "--defender_timesteps 2500 "
              "--attacker_timesteps 1500 "
              "--evaluation_episodes 3 "
              "--seeds 1 2 "
              f"--output_dir {out_dir}/adaptive_hof")
    
    print("Final experiment package execution orchestrated.")
