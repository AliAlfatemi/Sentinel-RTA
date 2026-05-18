import pytest
import numpy as np
from gymnasium import spaces
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.baselines import RandomDefender, StaticThresholdDefender, AdaptiveThresholdDefender, ShieldOnlyPolicy

@pytest.fixture
def action_space():
    return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

def test_random_defender(action_space):
    defender1 = RandomDefender(action_space, seed=42)
    defender2 = RandomDefender(action_space, seed=42)
    obs = np.zeros(10)
    
    act1, _ = defender1.predict(obs)
    act2, _ = defender2.predict(obs)
    
    assert np.array_equal(act1, act2), "Seeded random defender should be reproducible"
    assert action_space.contains(act1), "Output action must be valid"

def test_static_threshold_defender(action_space):
    defender = StaticThresholdDefender(action_space, anomaly_threshold=0.65, mitigation_action=0.5, benign_action=0.0)
    
    # High anomaly observation
    obs_high = np.array([0.8, 0.8, 0.8, 0.8, 0, 0, 0, 0, 0, 0])
    act_high, _ = defender.predict(obs_high)
    assert np.all(act_high == 0.5), "High anomaly observation should produce mitigation"
    
    # Low anomaly observation
    obs_low = np.array([0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0])
    act_low, _ = defender.predict(obs_low)
    assert np.all(act_low == 0.0), "Low anomaly observation should produce benign action"

def test_adaptive_threshold_defender(action_space):
    defender = AdaptiveThresholdDefender(action_space, anomaly_threshold=0.65, ewma_alpha=0.1, action_step_up=0.1, action_step_down=0.1)
    
    # Repeated high anomaly
    obs_high = np.array([0.9, 0.9, 0.9, 0.9, 0, 0, 0, 0, 0, 0])
    defender.predict(obs_high)
    act1, _ = defender.predict(obs_high)
    act2, _ = defender.predict(obs_high)
    assert np.all(act2 > act1), "Repeated high anomaly should increase action"
    
    # Repeated low anomaly
    obs_low = np.array([0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0])
    defender.predict(obs_low)
    act3, _ = defender.predict(obs_low)
    act4, _ = defender.predict(obs_low)
    assert np.all(act4 < act3), "Repeated low anomaly should relax action"

def test_shield_only_policy(action_space):
    # Dummy shield that just logs and returns bounded action
    class DummyShield:
        def repair_action(self, obs, raw_action, temporal_state):
            return np.clip(raw_action, -0.5, 0.5), {"repairs": 1}
            
    base = StaticThresholdDefender(action_space, anomaly_threshold=0.65, mitigation_action=0.8, benign_action=0.0)
    shield = DummyShield()
    policy = ShieldOnlyPolicy(base, shield)
    
    obs = np.array([0.9, 0.9, 0.9, 0.9, 0, 0, 0, 0, 0, 0])
    # The policy itself just wraps the base; the env uses the shield.
    raw_act, _ = policy.predict(obs)
    
    # Simulate environment shield evaluation
    final_act, info = shield.repair_action(obs, raw_act, None)
    assert np.all(final_act <= 0.5), "Final action respects shield bound"
    assert "repairs" in info, "Repair flags are logged correctly"

def test_no_oracle_baseline_usage():
    # Verify that the baselines only require observation vectors, not ground truth objects
    import inspect
    from agents.baselines import RandomDefender, StaticThresholdDefender, AdaptiveThresholdDefender
    
    for cls in [RandomDefender, StaticThresholdDefender, AdaptiveThresholdDefender]:
        sig = inspect.signature(cls.predict)
        params = list(sig.parameters.keys())
        assert 'is_attack' not in params
        assert 'labels' not in params
