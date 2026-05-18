import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.ddos_coevolution_env import CoEvolutionDDoSEnv

def test_attacker_action_space():
    env = CoEvolutionDDoSEnv()
    env.reset()
    # Test benign
    env._apply_attacker_action(np.array([0, 0, 0, 1, 0]))
    assert env.current_attacker_intent['intensity'] == 0.2
    assert env.current_attacker_intent['is_attack'] == 1.0
