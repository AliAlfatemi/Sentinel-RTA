import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from attackers.policies import HeuristicAdaptiveAttackerPolicy

def test_heuristic_attacker():
    pol = HeuristicAdaptiveAttackerPolicy()
    # obs: [rolling_sq, rolling_leak, rolling_action, rolling_repairs, rolling_sla, time]
    obs = np.array([1.0, 0.01, 0.1, 0.0, 0.0, 0.5])
    action, _ = pol.predict(obs)
    assert action[0] == 3 # Increased intensity because leak is low and action is low
