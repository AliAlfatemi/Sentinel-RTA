import argparse
import yaml
import os
import sys
from stable_baselines3 import PPO
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.ddos_coevolution_env import CoEvolutionDDoSEnv
from attackers.policies import RandomAttackerPolicy, HeuristicAdaptiveAttackerPolicy, FixedAttackerPolicy
from attackers.hall_of_fame import HallOfFame
from training.coevolution_trainer import CoEvolutionTrainer, evaluate_coevolution
from shields.rta_shield import RTAShield


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--generations', type=int, default=1)
    parser.add_argument('--defender_timesteps', type=int, default=512)
    parser.add_argument('--attacker_timesteps', type=int, default=256)
    parser.add_argument('--evaluation_episodes', type=int, default=1)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1])
    parser.add_argument('--output_dir', type=str, required=True)
    return parser.parse_args()

def run_experiment(args, exp_name, use_hof, use_shield, static_attacker=False, test_forgetting=False):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Override config with args
    config['training']['generations'] = args.generations
    config['training']['defender_timesteps_per_generation'] = args.defender_timesteps
    config['training']['attacker_timesteps_per_generation'] = args.attacker_timesteps
    config['training']['evaluation_episodes'] = args.evaluation_episodes
    config['hall_of_fame']['enabled'] = use_hof
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize components
    base_env = CoEvolutionDDoSEnv(
        max_steps=500,
        defender_reward_config=config['defender']['reward_weights'],
        attacker_reward_config=config['attacker']['reward_weights'],
        attacker_objective=config['attacker']['objective']
    )
    
    shield = None
    if use_shield:
        shield = RTAShield(
            max_collateral_tolerance=config['shield']['max_drop_without_confidence'],
            temporal_enabled=config['shield']['temporal_enabled'],
            temporal_recovery_max_action=config['shield'].get('temporal_recovery_max_action', 0.05)
        )
        
    hof = HallOfFame(config['hall_of_fame']) if use_hof else None
    
    attacker_policy = FixedAttackerPolicy([1, 0, 0, 1, 0]) if static_attacker else HeuristicAdaptiveAttackerPolicy()
    
    trainer = CoEvolutionTrainer(config, base_env, shield, hof)
    
    for seed in args.seeds:
        print(f"\\n--- Running Experiment: {exp_name} | Seed: {seed} ---")
        
        # Initialize fresh defender
        from stable_baselines3.common.env_util import make_vec_env
        from envs.ddos_coevolution_env import DefenderTrainingEnv
        
        def_env = DefenderTrainingEnv(base_env, attacker_policy, shield)
        defender_ppo = PPO("MlpPolicy", def_env, verbose=0, seed=seed)
        
        # Train Co-evolution
        exp_output_dir = os.path.join(args.output_dir, f"exp_{exp_name}_seed_{seed}")
        os.makedirs(exp_output_dir, exist_ok=True)
        
        trained_def, final_att = trainer.run_coevolution(defender_ppo, attacker_policy, exp_output_dir)
        
        # Evaluate Held-out
        held_out_attacker = FixedAttackerPolicy([3, 4, 1, 2, 2]) # Burst mixed high entropy
        heldout_df = evaluate_coevolution(base_env, trained_def, held_out_attacker, shield, args.evaluation_episodes)
        heldout_df.to_csv(os.path.join(exp_output_dir, 'heldout_attacker_evaluation.csv'), index=False)
        
        if test_forgetting:
            from gymnasium import spaces
            att_action_space = spaces.MultiDiscrete([4, 5, 5, 3, 3])
            old_attacker = RandomAttackerPolicy(att_action_space)
            
            forget_df = evaluate_coevolution(base_env, trained_def, old_attacker, shield, args.evaluation_episodes)
            forget_df.to_csv(os.path.join(exp_output_dir, 'forgetting_analysis.csv'), index=False)
            
        if use_hof:
            hof_meta = pd.DataFrame(hof.get_metadata())
            if not hof_meta.empty:
                hof_meta.to_csv(os.path.join(exp_output_dir, 'hall_of_fame_metadata.csv'), index=False)

if __name__ == "__main__":
    args = parse_args()
    
    # Extract prob from config for naming
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        prob = config['hall_of_fame'].get('sample_probability', 0.1)
        mode = config['hall_of_fame'].get('admission_mode', 'pareto')
        
    # Run required comparisons
    run_experiment(args, "Static_NoShield", use_hof=False, use_shield=False, static_attacker=True, test_forgetting=True)
    run_experiment(args, "Adaptive_NoShield", use_hof=False, use_shield=False, static_attacker=False, test_forgetting=True)
    run_experiment(args, "Adaptive_Shield_NoHoF", use_hof=False, use_shield=True, static_attacker=False, test_forgetting=True)
    run_experiment(args, f"Adaptive_Shield_HoF_{mode}_{prob}", use_hof=True, use_shield=True, static_attacker=False, test_forgetting=True)
    
    print("\\nCo-evolution execution completed.")
