import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.analyze_phase2c_results import is_passive_policy

def test_passive_policy_detector():
    assert is_passive_policy(mean_action=0.01, attack_period_mean_action=0.02, attack_leakage=0.8)
    assert not is_passive_policy(mean_action=0.5, attack_period_mean_action=0.8, attack_leakage=0.1)
    assert not is_passive_policy(mean_action=0.01, attack_period_mean_action=0.5, attack_leakage=0.1)
