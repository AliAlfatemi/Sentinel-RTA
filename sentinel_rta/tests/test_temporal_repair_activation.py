import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shields.rta_shield import RTAShield

def test_temporal_repair_activation():
    shield = RTAShield(temporal_enabled=True)
    obs = np.zeros(15)
    obs[14] = 0.8 # Attack is high
    raw_action = np.array([0.9])
    
    # Synthetic context with low SQ (should trigger temporal repair)
    temporal_state = {
        'rolling_service_quality': 0.8,
        'rolling_sla_violation_rate': 0.1
    }
    
    final_action, info = shield.repair_action(obs, raw_action, temporal_state)
    
    assert info['temporal_shield_repair'] == True
    assert final_action[0] < 0.9 # Should be bounded to max(0.05, 0.5 * 0.8) = 0.4
