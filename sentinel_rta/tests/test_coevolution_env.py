import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.ddos_coevolution_env import CoEvolutionDDoSEnv, DefenderTrainingEnv, AttackerTrainingEnv
from attackers.policies import FixedAttackerPolicy

def test_coevolution_env():
    env = CoEvolutionDDoSEnv()
    att_pol = FixedAttackerPolicy([1, 0, 0, 1, 0])
    def_env = DefenderTrainingEnv(env, att_pol)
    obs, _ = def_env.reset()
    obs, reward, done, trunc, info = def_env.step(np.array([0.5]))
    assert len(obs) == 15
