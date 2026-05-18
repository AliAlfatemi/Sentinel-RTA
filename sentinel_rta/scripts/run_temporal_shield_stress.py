import os
import sys
import argparse
import yaml
import time
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.ddos_env import DDoSEnv
from shields.rta_shield import RTAShield

class EdgeRidingPolicy:
    def __init__(self, inst_max_action):
        self.action = np.array([inst_max_action * 0.95], dtype=np.float32)
    def predict(self, obs, deterministic=True):
        return self.action, None

class SustainedModeratePolicy:
    def __init__(self):
        self.action = np.array([0.08], dtype=np.float32)
    def predict(self, obs, deterministic=True):
        return self.action, None

class BurstyNearLimitPolicy:
    def __init__(self, inst_max_action):
        self.high = np.array([inst_max_action * 0.95], dtype=np.float32)
        self.low = np.array([0.0], dtype=np.float32)
        self.step = 0
    def predict(self, obs, deterministic=True):
        self.step += 1
        return self.high if (self.step % 20 < 10) else self.low, None

class AggressivePolicy:
    def __init__(self):
        self.action = np.array([0.9], dtype=np.float32)
    def predict(self, obs, deterministic=True):
        return self.action, None

def evaluate_stress_policy(policy, policy_name, env, shield, shield_mode, config, seed, scenario, episodes, max_steps):
    per_step_logs = []
    total_repairs = 0
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        shield_context = env.get_shield_context()
        
        for step in range(max_steps):
            # 1. Inference
            start_inf = time.perf_counter()
            raw_action, _ = policy.predict(obs, deterministic=True)
            inf_lat_ms = (time.perf_counter() - start_inf) * 1000.0
            
            # 2. Shield
            shield_lat_ms = 0.0
            final_action = raw_action
            is_inst_repair = False
            is_temp_repair = False
            repaired = False
            repair_reason = "None"
            dynamic_max_action = 1.0
            
            if shield_mode != "No Shield":
                start_shield = time.perf_counter()
                final_action, shield_info = shield.repair_action(obs, raw_action, shield_context)
                shield_lat_ms = (time.perf_counter() - start_shield) * 1000.0
                
                is_inst_repair = shield_info['instantaneous_shield_repair']
                is_temp_repair = shield_info['temporal_shield_repair']
                repaired = shield_info['total_shield_repair']
                repair_reason = shield_info['repair_reason']
                dynamic_max_action = shield_info['dynamic_max_action']
                if repaired: total_repairs += 1
            
            # 3. Step
            start_env = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(final_action)
            env_lat_ms = (time.perf_counter() - start_env) * 1000.0
            
            shield_context = env.get_shield_context()
            
            per_step_logs.append({
                "policy_name": policy_name,
                "shield_mode": shield_mode,
                "scenario": scenario,
                "seed": seed,
                "step": step,
                "raw_action": raw_action[0],
                "final_action": final_action[0],
                "dynamic_max_action": dynamic_max_action,
                "action_reduction_due_to_temporal_shield": (raw_action[0] - final_action[0]) if is_temp_repair else 0.0,
                "instantaneous_shield_repair": is_inst_repair,
                "temporal_shield_repair": is_temp_repair,
                "total_shield_repair": repaired,
                "repair_reason": repair_reason,
                "service_quality": info['sq'],
                "rolling_service_quality": shield_context['rolling_service_quality'],
                "collateral_damage": info['collateral'],
                "rolling_collateral_damage": shield_context['rolling_collateral_damage'],
                "attack_leakage": info['leakage'],
                "rolling_attack_leakage": shield_context['rolling_attack_leakage'],
                "sla_violation": info['sla_violation'],
                "rolling_sla_violation": shield_context['rolling_sla_violation_rate'],
                "cumulative_sla_violation": shield_context['recent_sla_violations'],
                "safety_budget_remaining": max(0.0, 0.05 - shield_context['rolling_sla_violation_rate']),
                "inference_latency_ms": inf_lat_ms,
                "shield_latency_ms": shield_lat_ms,
                "env_step_latency_ms": env_lat_ms,
                "total_control_latency_ms": inf_lat_ms + shield_lat_ms + env_lat_ms
            })
            
    return per_step_logs

def run_temporal_stress(config_path, output_dir, seeds, scenarios, policy_names, episode_length, ppo_model_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    inst_max_action = config['shield']['max_drop_without_confidence']
    temp_recovery_action = config['shield'].get('temporal_recovery_max_action', 0.05)
    shield_modes = ["No Shield", "Instantaneous Runtime Shield", "Temporal Runtime Shield"]
    
    all_logs = []
    
    for seed in seeds:
        for scenario in scenarios:
            env = DDoSEnv(max_steps=episode_length, base_legitimate_rate=config['env']['base_legitimate_rate'], 
                          reward_mode=config['env'].get('reward_mode', 'A'), reward_weights=config['env'].get('reward_weights', {}),
                          scenario=scenario, oracle_mode=False)
                          
            for shield_mode in shield_modes:
                temp_enabled = (shield_mode == "Temporal Runtime Shield")
                shield = RTAShield(max_collateral_tolerance=inst_max_action, temporal_enabled=temp_enabled, temporal_recovery_max_action=temp_recovery_action)
                
                # Setup policies
                policies = {}
                if "edge_riding" in policy_names:
                    policies["Edge-Riding Stress Policy"] = EdgeRidingPolicy(inst_max_action)
                if "sustained_moderate" in policy_names:
                    policies["Sustained Moderate Stress Policy"] = SustainedModeratePolicy()
                if "bursty_near_limit" in policy_names:
                    policies["Bursty Near-Limit Stress Policy"] = BurstyNearLimitPolicy(inst_max_action)
                if "aggressive" in policy_names:
                    policies["Aggressive Stress Policy"] = AggressivePolicy()
                if "ppo_c1" in policy_names and ppo_model_dir:
                    model_path = os.path.join(ppo_model_dir, f"seed_{seed}", "model.zip")
                    if os.path.exists(model_path):
                        policies["PPO C1 Policy"] = PPO.load(model_path)
                    else:
                        print(f"Warning: PPO model not found at {model_path}. Skipping.")
                        
                for pol_name, pol_obj in policies.items():
                    print(f"Evaluating Seed {seed} | Scenario {scenario} | Shield {shield_mode} | Policy {pol_name}")
                    logs = evaluate_stress_policy(pol_obj, pol_name, env, shield, shield_mode, config, seed, scenario, episodes=1, max_steps=episode_length)
                    all_logs.extend(logs)
                    
    # Save step logs
    df = pd.DataFrame(all_logs)
    df.to_csv(os.path.join(output_dir, "per_step_logs.csv"), index=False)
    
    # Generate Summary
    summary = df.groupby(["policy_name", "shield_mode", "scenario"]).agg(
        service_quality=("service_quality", "mean"),
        mitigation_efficiency=("attack_leakage", lambda x: 1.0 - x.mean()), # approx
        attack_leakage=("attack_leakage", "mean"),
        collateral_damage=("collateral_damage", "mean"),
        rolling_service_quality=("rolling_service_quality", "mean"),
        rolling_collateral_damage=("rolling_collateral_damage", "mean"),
        sla_violation_count=("sla_violation", "sum"),
        rolling_sla_violation_count=("rolling_sla_violation", "sum"),
        cumulative_sla_violation_count=("cumulative_sla_violation", "max"),
        instantaneous_shield_repair_count=("instantaneous_shield_repair", "sum"),
        temporal_shield_repair_count=("temporal_shield_repair", "sum"),
        total_shield_repair_count=("total_shield_repair", "sum"),
        mean_raw_action=("raw_action", "mean"),
        mean_final_action=("final_action", "mean"),
        action_reduction_due_to_temporal_shield=("action_reduction_due_to_temporal_shield", "mean"),
        dynamic_max_action_mean=("dynamic_max_action", "mean"),
        latency_mean_ms=("total_control_latency_ms", "mean"),
        latency_p95_ms=("total_control_latency_ms", lambda x: x.quantile(0.95))
    ).reset_index()
    
    # Correct ME calculation using true attack passed / sent
    # We will do a quick estimation here, real analysis will be in the analyzer script
    
    summary.to_csv(os.path.join(output_dir, "temporal_stress_summary.csv"), index=False)
    
    # Optional Sweep
    run_parameter_sweep(config_path, output_dir, seeds[0], env, inst_max_action)

def run_parameter_sweep(config_path, output_dir, seed, env, inst_max_action):
    print("\nRunning Temporal Parameter Sweep...")
    # Reduced grid to save time
    rolling_windows = [10, 25]
    rolling_sla_thresholds = [0.00, 0.05]
    min_rolling_sqs = [0.95, 0.99]
    temporal_recovery_bounds = [0.03, 0.05, 0.08]
    sweep_results = []
    
    policy = EdgeRidingPolicy(inst_max_action)
    
    for rw in rolling_windows:
        for thresh in rolling_sla_thresholds:
            for sq_thresh in min_rolling_sqs:
                for rec_bound in temporal_recovery_bounds:
                    test_env = DDoSEnv(max_steps=500, scenario="long_sustained_attack", oracle_mode=False, reward_weights={"rolling_window": rw})
                    test_shield = RTAShield(max_collateral_tolerance=inst_max_action, temporal_enabled=True, temporal_recovery_max_action=rec_bound)
                    
                    def custom_repair(obs, raw_action, temporal_state=None):
                        safe_action = np.copy(raw_action)
                        info = {'total_shield_repair': False, 'instantaneous_shield_repair': False, 'temporal_shield_repair': False, 'dynamic_max_action': 1.0, 'repair_reason': 'None'}
                        if not test_shield.check_safety(obs, raw_action):
                            safe_action[0] = min(safe_action[0], test_shield.max_collateral_tolerance)
                            info['instantaneous_shield_repair'] = True
                            info['total_shield_repair'] = True
                            info['repair_reason'] = 'Instantaneous bounds exceeded'
                        info['dynamic_max_action'] = test_shield.max_collateral_tolerance if info['instantaneous_shield_repair'] else 1.0
                        
                        if temporal_state:
                            rolling_sq = temporal_state.get('rolling_service_quality', 1.0)
                            rolling_sla = temporal_state.get('rolling_sla_violation_rate', 0.0)
                            temp_max_action = 1.0
                            
                            if rolling_sla > thresh or rolling_sq < sq_thresh:
                                temp_max_action = test_shield.temporal_recovery_max_action
                            if safe_action[0] > temp_max_action:
                                safe_action[0] = temp_max_action
                                info['temporal_shield_repair'] = True
                                info['total_shield_repair'] = True
                                info['repair_reason'] = 'Temporal SLA risk too high'
                            info['dynamic_max_action'] = min(info['dynamic_max_action'], temp_max_action)
                        return safe_action, info
                        
                    test_shield.repair_action = custom_repair
                    
                    logs = evaluate_stress_policy(policy, "Edge-Riding", test_env, test_shield, "Temporal Runtime Shield", {}, seed, "long_sustained_attack", 1, 500)
                    df_logs = pd.DataFrame(logs)
                    
                    sweep_results.append({
                        "rolling_window": rw,
                        "rolling_sla_threshold": thresh,
                        "min_rolling_service_quality": sq_thresh,
                        "temporal_recovery_max_action": rec_bound,
                        "cumulative_sla_violations": df_logs["sla_violation"].sum(),
                        "temporal_repairs": df_logs["temporal_shield_repair"].sum()
                    })
            
    pd.DataFrame(sweep_results).to_csv(os.path.join(output_dir, "temporal_parameter_sweep.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seeds", nargs='+', type=int, required=True)
    parser.add_argument("--scenarios", nargs='+', type=str, required=True)
    parser.add_argument("--policies", nargs='+', type=str, required=True)
    parser.add_argument("--episode_length", type=int, required=True)
    parser.add_argument("--ppo_model_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    run_temporal_stress(args.config, args.output_dir, args.seeds, args.scenarios, args.policies, args.episode_length, args.ppo_model_dir)
