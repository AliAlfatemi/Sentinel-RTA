import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.ddos_env import DDoSEnv
from shields.rta_shield import RTAShield

def test_temporal_shield():
    shield = RTAShield(temporal_enabled=True)
    obs = np.zeros(15, dtype=np.float32)
    obs[14] = 1.0 # High attack confidence
    raw_action = np.array([0.9])
    
    # Simulate a bad temporal state
    temporal_state = {
        'rolling_service_quality': 0.8,
        'rolling_sla_violation_rate': 0.2
    }
    
    safe_action, shield_info = shield.repair_action(obs, raw_action, temporal_state)
    
    # Since SQ is 0.8, temp_max_action defaults to temporal_recovery_max_action (0.05)
    # Our action of 0.9 should be bounded to 0.05
    assert np.isclose(safe_action[0], 0.05)
    assert shield_info['temporal_shield_repair'] == True
    assert shield_info['instantaneous_shield_repair'] == False
    assert shield_info['total_shield_repair'] == True

def test_temporal_sla_metrics():
    env = DDoSEnv(reward_weights={"rolling_window": 5})
    env.reset()
    
    for _ in range(5):
        # Apply a terrible action that drops all traffic to force SLA violation
        env.step(np.array([1.0]))
        
    context = env.get_shield_context()
    
    assert context['rolling_service_quality'] < 1.0
    assert context['rolling_sla_violation_rate'] > 0.0
    assert context['cumulative_legitimate_dropped'] > 0.0

def test_temporal_reward():
    env = DDoSEnv(reward_mode="C", reward_weights={"w_sq": 1.0, "w_temporal_sla": 2.0, "rolling_window": 5})
    env.reset()
    
    # Step 1: Force SLA violation
    _, reward1, _, _, info1 = env.step(np.array([1.0]))
    
    # Step 2: Even if action is safe, temporal penalty should exist due to rolling history
    _, reward2, _, _, info2 = env.step(np.array([0.0]))
    
    assert 'penalty_temporal_sla_risk' in info2
    assert info2['penalty_temporal_sla_risk'] < 0.0 # Negative penalty
    
    # Verify total reward math
    expected_total = (info2['reward_service_quality'] + 
                      info2['reward_mitigation'] + 
                      info2['penalty_attack_leakage'] + 
                      info2['penalty_collateral_damage'] + 
                      info2['penalty_sla_violation'] + 
                      info2['penalty_action_cost'] + 
                      info2['penalty_action_change'] + 
                      info2['penalty_temporal_sla_risk'])
                      
    assert np.isclose(info2['total_reward'], expected_total)
