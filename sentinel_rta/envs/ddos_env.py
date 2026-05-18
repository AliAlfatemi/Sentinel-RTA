import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import evaluator

class DDoSEnv(gym.Env):
    """
    Simulation Environment for Sentinel-RTA DDoS Defense.
    Supports configurable reward modes to prevent passive policy collapse.
    """
    def __init__(self, max_steps=1000, base_legitimate_rate=0.5, reward_mode="A", reward_weights=None, scenario="periodic", oracle_mode=False):
        super().__init__()
        self.max_steps = max_steps
        self.base_legitimate_rate = base_legitimate_rate
        self.current_step = 0
        self.reward_mode = reward_mode
        self.reward_weights = reward_weights or {}
        self.scenario = scenario
        self.oracle_mode = oracle_mode
        self.prev_action = 0.0
        self.rolling_window = self.reward_weights.get("rolling_window", 25)
        
        from collections import deque
        self.history_sq = deque(maxlen=self.rolling_window)
        self.history_cd = deque(maxlen=self.rolling_window)
        self.history_leak = deque(maxlen=self.rolling_window)
        self.history_sla = deque(maxlen=self.rolling_window)
        self.history_action = deque(maxlen=self.rolling_window)
        
        self.cum_legit_dropped = 0.0
        self.cum_legit_served = 0.0
        self.cum_attack_dropped = 0.0
        self.cum_attack_passed = 0.0
        
        # 15-dimensional observation space (normalized 0 to 1)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(15,), dtype=np.float32)
        
        # Action space: Continuous [drop_intensity]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.prev_action = 0.0
        
        self.history_sq.clear()
        self.history_cd.clear()
        self.history_leak.clear()
        self.history_sla.clear()
        self.history_action.clear()
        
        self.cum_legit_dropped = 0.0
        self.cum_legit_served = 0.0
        self.cum_attack_dropped = 0.0
        self.cum_attack_passed = 0.0
        return self._get_obs(), {}
        
    def _get_obs(self):
        # Determine attack and volume based on scenario
        is_attack = 0.0
        self.current_intensity = 0.0
        legit_volume = self.base_legitimate_rate
        
        syn_rate, udp_rate, http_rate, icmp_rate = 0.1, 0.1, 0.1, 0.0
        src_ip_ent = 0.5
        
        if self.scenario == "periodic":
            is_attack = 1.0 if (self.current_step % 200 > 100) else 0.0
            self.current_intensity = np.random.uniform(0.3, 0.8) if is_attack else 0.0
            syn_rate = 0.8 if is_attack else 0.1
            src_ip_ent = 0.9 if is_attack else 0.2
            
        elif self.scenario == "long_sustained_attack":
            is_attack = 1.0 if self.current_step >= 100 else 0.0
            self.current_intensity = np.random.uniform(0.5, 0.9) if is_attack else 0.0
            syn_rate = 0.8 if is_attack else 0.1
            src_ip_ent = 0.9 if is_attack else 0.2
            
        elif self.scenario == "flash_crowd":
            is_attack = 0.0
            self.current_intensity = 0.0
            if 200 <= self.current_step <= 400:
                legit_volume = min(1.0, self.base_legitimate_rate + 0.6) # Huge spike
                
        elif self.scenario == "low_and_slow":
            is_attack = 1.0 if self.current_step >= 50 else 0.0
            self.current_intensity = 0.15 if is_attack else 0.0 # Just enough to leak, but low signal
            http_rate = 0.4 if is_attack else 0.1
            src_ip_ent = 0.5
            
        elif self.scenario == "mixed_attack_shift":
            if self.current_step >= 100:
                is_attack = 1.0
                phase = (self.current_step - 100) // 200
                self.current_intensity = np.random.uniform(0.4, 0.7)
                src_ip_ent = 0.9
                if phase % 4 == 0: syn_rate = 0.8
                elif phase % 4 == 1: udp_rate = 0.8
                elif phase % 4 == 2: http_rate = 0.8
                elif phase % 4 == 3: icmp_rate = 0.8
                
        elif self.scenario == "clean_benign":
            is_attack = 0.0
            self.current_intensity = 0.0
            
        self._current_is_attack = is_attack
        self._current_legit_volume = legit_volume
        
        vol = min(1.0, legit_volume + self.current_intensity)
        pkt_rate = min(1.0, legit_volume + self.current_intensity * 1.5)
        
        # Simulated metrics
        obs = np.array([
            vol,               # Normalized traffic volume
            pkt_rate,          # Packet rate
            syn_rate,          # SYN rate
            udp_rate,          # UDP rate
            http_rate,         # HTTP rate
            icmp_rate,         # ICMP rate
            src_ip_ent,        # Source IP entropy
            0.5,               # Dest port entropy
            0.5,               # Protocol distribution
            vol * 0.8,         # Queue occupancy
            vol * 0.9,         # Service latency
            max(0, vol - 0.8), # Packet loss
            self.prev_action,  # Current mitigation
            legit_volume / max(1e-5, vol), # Est legit ratio
            is_attack if self.oracle_mode else min(1.0, max(0.0, (vol - self.base_legitimate_rate) / 0.5)) # Est attack indicator (Volatility)
        ], dtype=np.float32)
        
        return np.clip(obs, 0.0, 1.0)
        
    def get_shield_context(self):
        """Returns the temporal safety state *before* the next action is applied."""
        rolling_sq = np.mean(self.history_sq) if self.history_sq else 1.0
        rolling_cd = np.mean(self.history_cd) if self.history_cd else 0.0
        rolling_leak = np.mean(self.history_leak) if self.history_leak else 0.0
        rolling_sla_rate = np.mean(self.history_sla) if self.history_sla else 0.0
        recent_action_mean = np.mean(self.history_action) if self.history_action else 0.0
        
        return {
            'current_step': self.current_step,
            'rolling_service_quality': float(rolling_sq),
            'rolling_collateral_damage': float(rolling_cd),
            'rolling_attack_leakage': float(rolling_leak),
            'rolling_sla_violation_rate': float(rolling_sla_rate),
            'cumulative_legitimate_dropped': float(self.cum_legit_dropped),
            'cumulative_legitimate_served': float(self.cum_legit_served),
            'cumulative_attack_passed': float(self.cum_attack_passed),
            'cumulative_attack_dropped': float(self.cum_attack_dropped),
            'remaining_episode_steps': max(0, self.max_steps - self.current_step),
            'recent_action_mean': float(recent_action_mean),
            'recent_action_sum': float(np.sum(self.history_action) if self.history_action else 0.0),
            'recent_sla_violations': int(np.sum(self.history_sla) if self.history_sla else 0)
        }
        
    def step(self, action):
        obs = self._get_obs()
        drop_intensity = np.clip(action[0], 0.0, 1.0)
        
        is_attack = self._current_is_attack
        
        # Mitigation math
        legit_sent = self._current_legit_volume
        attack_sent = self.current_intensity
        
        # Even during attacks, dropping traffic causes some collateral damage
        legit_dropped = drop_intensity * (0.2 + (1.0 - is_attack) * 0.2) * legit_sent
        attack_dropped = drop_intensity * (0.7 + is_attack * 0.2) * attack_sent
        
        # Metrics
        sq = evaluator.service_quality(legit_sent, legit_dropped)
        collateral = evaluator.collateral_damage(legit_sent, legit_dropped)
        sla_violation = evaluator.safety_violation_count(sq, sla_threshold=0.95)
        me = evaluator.mitigation_efficiency(attack_sent, attack_dropped)
        leakage = evaluator.attack_leakage(attack_sent, attack_dropped)
        
        # Update tracking
        self.cum_legit_dropped += legit_dropped
        self.cum_legit_served += (legit_sent - legit_dropped)
        self.cum_attack_dropped += attack_dropped
        self.cum_attack_passed += (attack_sent - attack_dropped)
        
        self.history_sq.append(sq)
        self.history_cd.append(collateral)
        self.history_leak.append(leakage)
        self.history_sla.append(sla_violation)
        self.history_action.append(drop_intensity)
        
        # Temporal metrics
        rolling_sla_rate = np.mean(self.history_sla) if self.history_sla else 0.0
        
        # Reward Component Calculation
        w_sq = self.reward_weights.get("w_sq", 10.0 if self.reward_mode == "A" else 0.0)
        w_me = self.reward_weights.get("w_me", 5.0 if self.reward_mode == "A" else 0.0)
        w_sla = self.reward_weights.get("w_sla", 20.0 if self.reward_mode == "A" else 0.0)
        w_cd = self.reward_weights.get("w_cd", 0.0)
        w_leak = self.reward_weights.get("w_leak", 0.0)
        w_cost = self.reward_weights.get("w_cost", 0.0)
        w_action_change = self.reward_weights.get("w_action_change", 0.0)
        w_temporal_sla = self.reward_weights.get("w_temporal_sla", 0.0)
        
        r_sq = w_sq * sq
        r_me = w_me * me
        p_sla = -w_sla * sla_violation
        p_cd = -w_cd * collateral
        p_leak = -w_leak * leakage
        p_cost = -w_cost * drop_intensity
        p_change = -w_action_change * abs(drop_intensity - self.prev_action)
        p_temp = -w_temporal_sla * rolling_sla_rate
        
        if self.reward_mode == "A":
            total_reward = r_sq + r_me + p_sla
        else:
            total_reward = r_sq + r_me + p_sla + p_cd + p_leak + p_cost + p_change + p_temp
            
        reward = float(total_reward)
        self.prev_action = drop_intensity
        
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        info = {
            'sq': sq,
            'me': me,
            'collateral': collateral,
            'sla_violation': sla_violation,
            'leakage': leakage,
            'action': drop_intensity,
            'is_attack': is_attack,
            'rolling_sla_violation_rate': float(rolling_sla_rate),
            # Reward Decomposition
            'reward_service_quality': float(r_sq),
            'reward_mitigation': float(r_me),
            'penalty_attack_leakage': float(p_leak),
            'penalty_collateral_damage': float(p_cd),
            'penalty_sla_violation': float(p_sla),
            'penalty_action_cost': float(p_cost),
            'penalty_action_change': float(p_change),
            'penalty_temporal_sla_risk': float(p_temp),
            'total_reward': reward
        }
        
        return obs, reward, terminated, truncated, info
