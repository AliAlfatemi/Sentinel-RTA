import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.ddos_env import DDoSEnv

def test_reward_modes():
    modes = ["A", "B", "C", "D"]
    for mode in modes:
        env = DDoSEnv(reward_mode=mode, reward_weights={"w_sq": 1.0, "w_me": 1.0})
        env.reset()
        _, reward, _, _, _ = env.step(np.array([0.5]))
        assert np.isfinite(reward)
