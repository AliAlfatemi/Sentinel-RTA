import sys
import os
import numpy as np
from stable_baselines3.common.env_checker import check_env

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.ddos_env import DDoSEnv

def test_gymnasium_api_compliance():
    env = DDoSEnv()
    # SB3 provides a strictly validated checker
    check_env(env, warn=True)
