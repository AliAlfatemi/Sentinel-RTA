import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.ddos_env import DDoSEnv

def test_reward_components():
    env = DDoSEnv(reward_mode="B", reward_weights={"w_sq": 1.0, "w_me": 1.0, "w_sla": 1.0})
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.array([0.5]))
    
    assert 'reward_service_quality' in info
    assert 'reward_mitigation' in info
    assert 'penalty_attack_leakage' in info
    assert 'total_reward' in info
    
    components_sum = (info['reward_service_quality'] + info['reward_mitigation'] + 
                      info['penalty_attack_leakage'] + info['penalty_collateral_damage'] + 
                      info['penalty_sla_violation'] + info['penalty_action_cost'] + 
                      info['penalty_action_change'])
                      
    assert np.isclose(info['total_reward'], components_sum)
