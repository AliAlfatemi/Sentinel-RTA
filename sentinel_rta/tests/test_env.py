import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.ddos_env import DDoSEnv

def test_env_reset():
    env = DDoSEnv()
    obs, info = env.reset()
    assert obs.shape == (15,)
    assert isinstance(info, dict)
    assert np.all((obs >= 0.0) & (obs <= 1.0))

def test_env_step():
    env = DDoSEnv()
    env.reset()
    action = np.array([0.5], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert obs.shape == (15,)
    assert isinstance(reward, (float, np.floating))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert 'sq' in info
    assert 'sla_violation' in info
    assert 'me' in info
    assert 'leakage' in info
