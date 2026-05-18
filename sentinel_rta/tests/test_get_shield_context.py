import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.ddos_env import DDoSEnv

def test_get_shield_context():
    env = DDoSEnv(max_steps=100)
    env.reset()
    
    context = env.get_shield_context()
    
    # Check all required keys exist
    expected_keys = [
        'current_step', 'rolling_service_quality', 'rolling_collateral_damage',
        'rolling_attack_leakage', 'rolling_sla_violation_rate', 'cumulative_legitimate_dropped',
        'cumulative_legitimate_served', 'cumulative_attack_passed', 'cumulative_attack_dropped',
        'remaining_episode_steps', 'recent_action_mean', 'recent_action_sum', 'recent_sla_violations'
    ]
    
    for key in expected_keys:
        assert key in context
        
    # Take a step and ensure context updates
    env.step(np.array([1.0]))
    new_context = env.get_shield_context()
    
    assert new_context['current_step'] == 1
    assert new_context['remaining_episode_steps'] == 99
    assert new_context['recent_action_mean'] == 1.0
