import argparse
import yaml
import os
import sys
from stable_baselines3 import PPO
import pandas as pd
from gymnasium import spaces

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.ddos_coevolution_env import CoEvolutionDDoSEnv, DefenderTrainingEnv
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
    parser.add_argument('--hof_sample_probs', type=float, nargs='+', default=[0.0, 0.3])
    parser.add_argument('--admission_modes', type=str, nargs='+', default=['diversity_aware'])
    parser.add_argument('--output_dir', type=str, required=True)
    return parser.parse_args()

def run_experiment(args, exp_name, hof_prob, admission_mode):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    config['training']['generations'] = args.generations
    config['training']['defender_timesteps_per_generation'] = args.defender_timesteps
    config['training']['attacker_timesteps_per_generation'] = args.attacker_timesteps
    config['training']['evaluation_episodes'] = args.evaluation_episodes
    
    use_hof = hof_prob > 0.0
    config['hall_of_fame']['enabled'] = use_hof
    config['hall_of_fame']['sample_probability'] = hof_prob
    config['hall_of_fame']['admission_mode'] = admission_mode
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    base_env = CoEvolutionDDoSEnv(
        max_steps=500,
        defender_reward_config=config['defender']['reward_weights'],
        attacker_reward_config=config['attacker']['reward_weights'],
        attacker_objective=config['attacker']['objective']
    )
    
    shield = RTAShield(
        max_collateral_tolerance=config['shield']['max_drop_without_confidence'],
        temporal_enabled=config['shield']['temporal_enabled'],
        temporal_recovery_max_action=config['shield'].get('temporal_recovery_max_action', 0.05)
    )
        
    hof = HallOfFame(config['hall_of_fame']) if use_hof else None
    
    attacker_policy = HeuristicAdaptiveAttackerPolicy()
    
    trainer = CoEvolutionTrainer(config, base_env, shield, hof)
    
    for seed in args.seeds:
        print(f"\\n--- Running Experiment: {exp_name} | Seed: {seed} ---")
        
        def_env = DefenderTrainingEnv(base_env, attacker_policy, shield)
        defender_ppo = PPO("MlpPolicy", def_env, verbose=0, seed=seed)
        
        exp_output_dir = os.path.join(args.output_dir, f"exp_{exp_name}_seed_{seed}")
        os.makedirs(exp_output_dir, exist_ok=True)
        
        trained_def, final_att = trainer.run_coevolution(defender_ppo, attacker_policy, exp_output_dir)
        
        held_out_attacker = FixedAttackerPolicy([3, 4, 1, 2, 2])
        heldout_df = evaluate_coevolution(base_env, trained_def, held_out_attacker, shield, args.evaluation_episodes)
        heldout_df.to_csv(os.path.join(exp_output_dir, 'heldout_attacker_evaluation.csv'), index=False)
        
        if use_hof:
            hof_meta = pd.DataFrame(hof.get_metadata())
            if not hof_meta.empty:
                hof_meta.to_csv(os.path.join(exp_output_dir, 'hall_of_fame_metadata.csv'), index=False)
                
            att_action_space = spaces.MultiDiscrete([4, 5, 5, 3, 3])
            old_attacker = RandomAttackerPolicy(att_action_space)
            
            forget_df = evaluate_coevolution(base_env, trained_def, old_attacker, shield, args.evaluation_episodes)
            forget_df.to_csv(os.path.join(exp_output_dir, 'forgetting_analysis.csv'), index=False)

if __name__ == "__main__":
    args = parse_args()
    
    for prob in args.hof_sample_probs:
        if prob == 0.0:
            run_experiment(args, "NoHoF", prob, "none")
        else:
            for mode in args.admission_modes:
                exp_name = f"HoF_prob{prob}_{mode}"
                run_experiment(args, exp_name, prob, mode)
                
    print("\\nPhase 3C sweep completed.")
