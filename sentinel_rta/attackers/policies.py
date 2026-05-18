import numpy as np

class AttackerPolicy:
    def predict(self, obs):
        raise NotImplementedError

class FixedAttackerPolicy(AttackerPolicy):
    def __init__(self, action):
        self.action = action
        
    def predict(self, obs):
        return np.array(self.action), None

class RandomAttackerPolicy(AttackerPolicy):
    def __init__(self, action_space):
        self.action_space = action_space
        
    def predict(self, obs):
        return self.action_space.sample(), None

class HeuristicAdaptiveAttackerPolicy(AttackerPolicy):
    def __init__(self):
        self.current_action = [2, 0, 0, 1, 0] # High intensity, SYN, sustained, normal entropy, uniform
        
    def predict(self, obs):
        # obs: [rolling_sq, rolling_leak, rolling_action, rolling_repairs, rolling_sla, time]
        rolling_sq = obs[0]
        rolling_leak = obs[1]
        rolling_action = obs[2]
        rolling_repairs = obs[3]
        
        # Adaptive rules
        if rolling_leak < 0.05 and rolling_action < 0.5:
            # Low leakage and low defense? Increase intensity
            self.current_action[0] = min(3, self.current_action[0] + 1)
        elif rolling_repairs > 0.1:
            # Shield is repairing a lot, meaning defense is maxed out and constrained
            # Shift pattern to evade or change protocol
            self.current_action[2] = (self.current_action[2] + 1) % 5
            self.current_action[1] = (self.current_action[1] + 1) % 5
        elif rolling_action > 0.8 and rolling_leak < 0.1:
            # Defense is extremely high, we are getting mitigated hard
            # Increase entropy and change protocol
            self.current_action[3] = 2 # high entropy
            self.current_action[1] = (self.current_action[1] + 1) % 5
            
        return np.array(self.current_action), None

class PPOWrapperPolicy(AttackerPolicy):
    def __init__(self, ppo_model):
        self.ppo_model = ppo_model
        
    def predict(self, obs):
        return self.ppo_model.predict(obs, deterministic=True)
