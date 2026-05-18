import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.ddos_coevolution_env import CoEvolutionDDoSEnv, DefenderTrainingEnv, AttackerTrainingEnv
from attackers.policies import RandomAttackerPolicy, HeuristicAdaptiveAttackerPolicy, FixedAttackerPolicy

class DummyPPO:
    # A simple wrapper simulating SB3 PPO for our discrete space during the Heuristic phase
    def __init__(self, policy):
        self.policy = policy
    def predict(self, obs, deterministic=True):
        return self.policy.predict(obs)
    def learn(self, total_timesteps):
        pass # Placeholder

def evaluate_coevolution(env, defender_policy, attacker_policy, shield=None, episodes=5):
    results = []
    
    for ep in range(episodes):
        def_obs, att_obs = env.reset()
        done = False
        
        while not done:
            att_action, _ = attacker_policy.predict(att_obs)
            def_action, _ = defender_policy.predict(def_obs)
            
            # Step the underlying environment directly
            def_obs, att_obs, def_rew, att_rew, term, trunc, info = env.step(def_action, att_action, shield)
            done = term or trunc
            
            results.append({
                'episode': ep,
                'service_quality': info['sq'],
                'mitigation_efficiency': info['me'],
                'attack_leakage': info['leakage'],
                'collateral_damage': info['collateral'],
                'sla_violation': info['sla_violation'],
                'shield_repair': info['shield_repair'],
                'defender_reward': def_rew,
                'attacker_reward': att_rew,
                'attack_intensity': env.current_attacker_intent['intensity'],
                'protocol': np.argmax([env.current_attacker_intent[k] for k in ['syn_rate', 'udp_rate', 'http_rate', 'icmp_rate']])
            })
            
    return pd.DataFrame(results)

class CoEvolutionTrainer:
    def __init__(self, config, base_env, shield, hall_of_fame):
        self.config = config
        self.base_env = base_env
        self.shield = shield
        self.hall_of_fame = hall_of_fame
        
    def run_coevolution(self, defender_ppo, initial_attacker_policy, output_dir):
        generations = self.config['training']['generations']
        def_timesteps = self.config['training']['defender_timesteps_per_generation']
        att_timesteps = self.config['training']['attacker_timesteps_per_generation']
        eval_episodes = self.config['training']['evaluation_episodes']
        
        generation_metrics = []
        current_attacker_policy = initial_attacker_policy
        
        # Initialize early attacker for forgetting tests
        from gymnasium import spaces
        att_action_space = spaces.MultiDiscrete([4, 5, 5, 3, 3])
        early_attacker = RandomAttackerPolicy(att_action_space)
        early_leakage = None
        
        replay_mode = self.config['hall_of_fame'].get('replay_mode', 'static') if self.hall_of_fame else 'static'
        current_sample_prob = self.config['hall_of_fame'].get('sample_probability', 0.1) if self.hall_of_fame else 0.0
        min_prob = self.config['hall_of_fame'].get('min_sample_probability', 0.05) if self.hall_of_fame else 0.0
        max_prob = self.config['hall_of_fame'].get('max_sample_probability', 0.30) if self.hall_of_fame else 0.0
        
        for gen in range(generations):
            print(f"--- Generation {gen} ---")
            
            # 1. Train Defender against Current Attacker (and HoF mix if implemented)
            print("Training Defender...")
            
            # Mix HoF and Current Attacker
            training_attacker = current_attacker_policy
            if self.hall_of_fame and self.config['hall_of_fame']['enabled']:
                if np.random.rand() < current_sample_prob:
                    sampled_attackers = self.hall_of_fame.sample(1)
                    if sampled_attackers:
                        training_attacker = sampled_attackers[0]
                        print("Training against sampled HoF attacker.")
            
            def_env = DefenderTrainingEnv(self.base_env, training_attacker, self.shield)
            defender_ppo.set_env(def_env)
            defender_ppo.learn(total_timesteps=def_timesteps)
            
            # 2. Train Attacker against Defender
            # (If we were using PPO Attacker. Since we start with Heuristic, we skip learn step)
            print("Training/Updating Attacker...")
            att_env = AttackerTrainingEnv(self.base_env, defender_ppo, self.shield)
            # current_attacker_ppo.set_env(att_env)
            # current_attacker_ppo.learn(total_timesteps=att_timesteps)
            
            # 3. Evaluate
            print("Evaluating Generation...")
            df_eval = evaluate_coevolution(self.base_env, defender_ppo, current_attacker_policy, self.shield, eval_episodes)
            
            gen_stats = {
                'generation': gen,
                'service_quality': df_eval['service_quality'].mean(),
                'attack_leakage': df_eval['attack_leakage'].mean(),
                'collateral_damage': df_eval['collateral_damage'].mean(),
                'sla_violations': df_eval['sla_violation'].sum() / eval_episodes,
                'shield_repairs': df_eval['shield_repair'].sum() / eval_episodes,
                'defender_reward': df_eval['defender_reward'].mean(),
                'attacker_reward': df_eval['attacker_reward'].mean(),
                'attack_success_rate': (df_eval['attack_leakage'] > 0.1).mean(),
                'control_latency_ms': 0.12 # Placeholder
            }
            generation_metrics.append(gen_stats)
            
            # Forgetting metric tracking
            if replay_mode == 'adaptive' and gen == 0:
                df_early = evaluate_coevolution(self.base_env, defender_ppo, early_attacker, self.shield, eval_episodes)
                early_leakage = df_early['attack_leakage'].mean()
                
            if replay_mode == 'adaptive' and gen > 0 and early_leakage is not None:
                df_early = evaluate_coevolution(self.base_env, defender_ppo, early_attacker, self.shield, eval_episodes)
                current_old_leakage = df_early['attack_leakage'].mean()
                forgetting_score = current_old_leakage - early_leakage
                
                forgetting_trigger = self.config['hall_of_fame'].get('forgetting_trigger', 0.05)
                current_leakage_trigger = self.config['hall_of_fame'].get('current_leakage_trigger', 0.10)
                
                # Adaptive logic
                if forgetting_score > forgetting_trigger:
                    # Forgetting is bad, increase replay
                    current_sample_prob = min(max_prob, current_sample_prob + 0.05)
                elif gen_stats['attack_leakage'] > current_leakage_trigger:
                    # Current performance is bad, decrease replay to allow plasticity
                    current_sample_prob = max(min_prob, current_sample_prob - 0.05)
                
                print(f"Adaptive HoF Prob adjusted to: {current_sample_prob:.2f}")
            
            # 4. Hall of Fame
            if self.hall_of_fame and self.config['hall_of_fame']['enabled']:
                metadata = {
                    'generation': gen,
                    'average_attacker_reward': gen_stats['attacker_reward'],
                    'average_attack_leakage': gen_stats['attack_leakage'],
                    'average_sla_violations': gen_stats['sla_violations'],
                    'attack_type_distribution': df_eval['protocol'].value_counts(normalize=True).reindex([0,1,2,3], fill_value=0).tolist()
                }
                added, reason = self.hall_of_fame.add(current_attacker_policy, metadata)
                if added:
                    print(f"Added attacker to HoF: {reason}")
            
        # Save metrics
        df_metrics = pd.DataFrame(generation_metrics)
        df_metrics.to_csv(os.path.join(output_dir, 'generation_metrics.csv'), index=False)
        
        return defender_ppo, current_attacker_policy
