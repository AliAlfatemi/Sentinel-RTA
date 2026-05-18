import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shields.rta_shield import RTAShield

def test_shield_safe_action():
    shield = RTAShield()
    # Mock observation: est_attack = 0.8 (high confidence)
    obs = np.zeros(15)
    obs[14] = 0.8
    action = np.array([0.9])
    
    repaired_action, shield_info = shield.repair_action(obs, action)
    assert shield_info['total_shield_repair'] == False
    
def test_shield_unsafe_action():
    shield = RTAShield()
    # Mock observation: est_attack = 0.1 (low confidence)
    obs = np.zeros(15)
    obs[14] = 0.1
    action = np.array([0.9]) # Unsafe high drop
    
    repaired_action, shield_info = shield.repair_action(obs, action)
    assert shield_info['total_shield_repair'] == True
    assert repaired_action[0] == 0.1
