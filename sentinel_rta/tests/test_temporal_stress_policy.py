import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.run_temporal_shield_stress import EdgeRidingPolicy, AggressivePolicy, SustainedModeratePolicy

def test_stress_policies():
    obs = np.zeros(15)
    
    edge_policy = EdgeRidingPolicy(0.1)
    action, _ = edge_policy.predict(obs)
    assert np.isclose(action[0], 0.095)
    
    agg_policy = AggressivePolicy()
    action, _ = agg_policy.predict(obs)
    assert np.isclose(action[0], 0.9)
    
    sus_policy = SustainedModeratePolicy()
    action, _ = sus_policy.predict(obs)
    assert np.isclose(action[0], 0.08)
