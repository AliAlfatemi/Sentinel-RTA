import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from attackers.hall_of_fame import HallOfFame
from attackers.policies import FixedAttackerPolicy

def test_hof_reward_only():
    hof = HallOfFame({'admission_mode': 'reward_only', 'reward_margin': 0.05})
    pol = FixedAttackerPolicy([1,0,0,1,0])
    added, reason = hof.add(pol, {'average_attacker_reward': 10.0})
    assert added and reason == 'reward_margin'

def test_hof_pareto():
    hof = HallOfFame({'admission_mode': 'pareto'})
    pol1 = FixedAttackerPolicy([1,0,0,1,0])
    hof.add(pol1, {'average_attacker_reward': 10.0, 'average_attack_leakage': 0.5})
    
    # Dominated attacker
    pol2 = FixedAttackerPolicy([1,1,0,1,0])
    added, reason = hof.add(pol2, {'average_attacker_reward': 5.0, 'average_attack_leakage': 0.2})
    assert not added
    
    # Nondominated attacker (lower reward, higher leakage)
    added, reason = hof.add(pol2, {'average_attacker_reward': 5.0, 'average_attack_leakage': 0.8})
    assert added

