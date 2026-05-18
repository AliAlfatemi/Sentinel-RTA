import os
import sys
import yaml
import time
import argparse
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.ddos_env import DDoSEnv
from shields.rta_shield import RTAShield

def evaluate_ppo(config_path, model_path, seed, shield_on, temporal_enabled, output_dir, eval_episodes=None):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    episodes = eval_episodes if eval_episodes else config['evaluation']['episodes']
    os.makedirs(output_dir, exist_ok=True)
    
    env = DDoSEnv(max_steps=config['env']['max_steps'], 
                  base_legitimate_rate=config['env']['base_legitimate_rate'],
                  reward_mode=config['env'].get('reward_mode', 'A'),
                  reward_weights=config['env'].get('reward_weights', {}))
                  
    shield = RTAShield(max_collateral_tolerance=config['shield']['max_drop_without_confidence'], temporal_enabled=temporal_enabled)
    model = PPO.load(model_path)
    
    if shield_on:
        policy_name = "PPO with Temporal Runtime Shield" if temporal_enabled else "PPO with Instantaneous Runtime Shield"
    else:
        policy_name = "PPO without Shield"
    
    per_step_logs = []
    
    total_sq = 0.0
    total_me = 0.0
    total_sla_violations = 0
    total_safety_violations = 0
    total_repairs = 0
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        shield_context = env.get_shield_context()
        
        for step in range(env.max_steps):
            # 1. Inference Latency
            start_inf = time.perf_counter()
            raw_action, _ = model.predict(obs, deterministic=True)
            inf_lat_ms = (time.perf_counter() - start_inf) * 1000.0
            
            # 2. Shield Latency
            shield_lat_ms = 0.0
            is_inst_repair = False
            is_temp_repair = False
            repaired = False
            final_action = raw_action
            
            if shield_on:
                start_shield = time.perf_counter()
                final_action, shield_info = shield.repair_action(obs, raw_action, shield_context)
                shield_lat_ms = (time.perf_counter() - start_shield) * 1000.0
                
                is_inst_repair = shield_info['instantaneous_shield_repair']
                is_temp_repair = shield_info['temporal_shield_repair']
                repaired = shield_info['total_shield_repair']
                
                if repaired: total_repairs += 1
                
            # 3. Environment Step Latency
            start_env = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(final_action)
            env_lat_ms = (time.perf_counter() - start_env) * 1000.0
            
            shield_context = env.get_shield_context()
            
            total_sq += info['sq']
            total_me += info['me']
            total_sla_violations += info['sla_violation']
            
            # A Safety Violation occurs if the FINAL action was unsafe
            safety_violation = 0 if shield.check_safety(obs, final_action) else 1
            total_safety_violations += safety_violation
            
            per_step_logs.append({
                "seed": seed,
                "episode": ep,
                "step": step,
                "policy_name": policy_name,
                "attack_intensity": obs[14], # Est attack
                "raw_action": raw_action[0],
                "final_action": final_action[0],
                "shield_enabled": shield_on,
                "shield_repaired": repaired,
                "instantaneous_shield_repair": is_inst_repair,
                "temporal_shield_repair": is_temp_repair,
                "reward": reward,
                "service_quality": info['sq'],
                "mitigation_efficiency": info['me'],
                "attack_leakage": info['leakage'],
                "collateral_damage": info['collateral'],
                "sla_violation": info['sla_violation'],
                "safety_violation": safety_violation,
                "rolling_sla_violation_rate": info.get('rolling_sla_violation_rate', 0.0),
                "inference_latency_ms": inf_lat_ms,
                "shield_latency_ms": shield_lat_ms,
                "total_control_latency_ms": inf_lat_ms + shield_lat_ms + env_lat_ms
            })
            
            if terminated or truncated:
                break

    df_logs = pd.DataFrame(per_step_logs)
    df_logs.to_csv(os.path.join(output_dir, "per_step_logs.csv"), index=False)
    
    total_steps = episodes * env.max_steps
    
    summary = {
        "Policy": policy_name,
        "Seed": seed,
        "Avg_SQ": total_sq / total_steps,
        "Avg_ME": total_me / total_steps,
        "Avg_Leakage": df_logs['attack_leakage'].mean(),
        "Avg_Collateral": df_logs['collateral_damage'].mean(),
        "SLA_Violations": total_sla_violations,
        "Safety_Violations": total_safety_violations,
        "Shield_Repairs": total_repairs,
        "Instantaneous_Repairs": df_logs['instantaneous_shield_repair'].sum() if 'instantaneous_shield_repair' in df_logs else 0,
        "Temporal_Repairs": df_logs['temporal_shield_repair'].sum() if 'temporal_shield_repair' in df_logs else 0,
        "Avg_Inference_Latency_ms": df_logs['inference_latency_ms'].mean(),
        "Avg_Shield_Latency_ms": df_logs['shield_latency_ms'].mean(),
        "Avg_Total_Control_Latency_ms": df_logs['total_control_latency_ms'].mean()
    }
    
    df_summary = pd.DataFrame([summary])
    df_summary.to_csv(os.path.join(output_dir, "episode_summary.csv"), index=False)
    print(f"Evaluation complete for {policy_name}. Results saved to {output_dir}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shield", type=str, required=True, choices=["on", "off"])
    parser.add_argument("--temporal", type=str, default="off", choices=["on", "off"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--eval_episodes", type=int, default=None)
    args = parser.parse_args()
    
    evaluate_ppo(args.config, args.model_path, args.seed, args.shield == "on", args.temporal == "on", args.output_dir, args.eval_episodes)
