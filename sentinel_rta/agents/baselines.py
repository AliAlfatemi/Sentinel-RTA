import numpy as np

class RandomDefender:
    def __init__(self, action_space, seed=1):
        self.action_space = action_space
        self.rng = np.random.RandomState(seed)
        self.policy_name = "Random Defender"

    def predict(self, obs, deterministic=True):
        # We assume Box action space for the defender
        action = self.rng.uniform(low=self.action_space.low, high=self.action_space.high, size=self.action_space.shape)
        return action.astype(np.float32), None


class StaticThresholdDefender:
    def __init__(self, action_space, anomaly_threshold=0.65, mitigation_action=0.5, benign_action=0.0):
        self.action_space = action_space
        self.anomaly_threshold = anomaly_threshold
        self.mitigation_action = mitigation_action
        self.benign_action = benign_action
        self.policy_name = "Static Threshold Defender"

    def predict(self, obs, deterministic=True):
        # Calculate a simple anomaly score from observable features.
        # Assuming obs is a normalized vector. We can use the mean of the first few generic features 
        # (like queue length, packet arrival rate, etc.). In our env, first 4 are typically generic rates.
        anomaly_score = np.mean(obs[:4]) if len(obs) >= 4 else 0.0
        
        if anomaly_score > self.anomaly_threshold:
            action = np.full(self.action_space.shape, self.mitigation_action)
        else:
            action = np.full(self.action_space.shape, self.benign_action)
            
        return np.clip(action, self.action_space.low, self.action_space.high), None


class AdaptiveThresholdDefender:
    def __init__(self, action_space, ewma_alpha=0.1, anomaly_threshold=0.65, 
                 max_action=0.8, min_action=0.0, action_step_up=0.05, action_step_down=0.03):
        self.action_space = action_space
        self.ewma_alpha = ewma_alpha
        self.anomaly_threshold = anomaly_threshold
        self.max_action = max_action
        self.min_action = min_action
        self.action_step_up = action_step_up
        self.action_step_down = action_step_down
        self.policy_name = "Adaptive Threshold Defender"
        
        self.baseline_score = 0.0
        self.current_action = np.full(self.action_space.shape, self.min_action)

    def predict(self, obs, deterministic=True):
        current_score = np.mean(obs[:4]) if len(obs) >= 4 else 0.0
        
        # Update baseline
        self.baseline_score = (self.ewma_alpha * current_score) + ((1 - self.ewma_alpha) * self.baseline_score)
        
        anomaly_score = current_score - self.baseline_score
        
        if anomaly_score > self.anomaly_threshold:
            self.current_action += self.action_step_up
        else:
            self.current_action -= self.action_step_down
            
        self.current_action = np.clip(self.current_action, self.min_action, self.max_action)
        return self.current_action.copy(), None


class ShieldOnlyPolicy:
    def __init__(self, base_policy, shield):
        self.base_policy = base_policy
        self.shield = shield
        self.policy_name = "Shield-only Policy"

    def predict(self, obs, deterministic=True):
        # Returns the raw action. The actual shielding happens inside the environment or the evaluation loop.
        # But wait, if this policy needs to evaluate the shield itself to log it, or does the env log it?
        # The prompt says: "Passes that action through the Instantaneous or Temporal Runtime Shield. Logs: raw_action, final_action..."
        # But the environment `step` function already calls the shield and logs `shield_repair` if a shield is passed to it.
        # To strictly comply, we can just return the base policy action and let the evaluate function pass it to the shield, 
        # which evaluates and logs properly via `info` dict.
        return self.base_policy.predict(obs, deterministic)
