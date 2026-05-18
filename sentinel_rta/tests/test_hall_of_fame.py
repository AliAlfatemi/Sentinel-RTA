import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from attackers.hall_of_fame import HallOfFame
from attackers.policies import FixedAttackerPolicy

def test_hall_of_fame():
    hof = HallOfFame()
    pol1 = FixedAttackerPolicy([1,0,0,1,0])
    meta1 = {'average_attacker_reward': 10.0, 'attack_type_distribution': [1,0,0,0,0]}
    added, reason = hof.add(pol1, meta1)
    assert added
    
    pol2 = FixedAttackerPolicy([1,1,0,1,0])
    meta2 = {'average_attacker_reward': 15.0, 'attack_type_distribution': [0,1,0,0,0]}
    added, reason = hof.add(pol2, meta2)
    assert added
