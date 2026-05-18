import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import evaluator
from shields.rta_shield import RTAShield

class CoEvolutionDDoSEnv:
    """
    Core simulation state for Co-Evolution.
    Maintains dual state for both Defender and Attacker.
    """
    def __init__(self, max_steps=1000, base_legitimate_rate=0.5, 
                 defender_reward_config=None, attacker_reward_config=None, attacker_objective="stealthy"):
        self.max_steps = max_steps
        self.base_legitimate_rate = base_legitimate_rate
        self.current_step = 0
        
        self.defender_reward_config = defender_reward_config or {}
        self.attacker_reward_config = attacker_reward_config or {}
        self.attacker_objective = attacker_objective
        
        self.rolling_window = 25
        self.history_sq = deque(maxlen=self.rolling_window)
        self.history_cd = deque(maxlen=self.rolling_window)
        self.history_leak = deque(maxlen=self.rolling_window)
        self.history_sla = deque(maxlen=self.rolling_window)
        self.history_action = deque(maxlen=self.rolling_window)
        self.history_repairs = deque(maxlen=self.rolling_window)
        
        self.prev_defender_action = 0.0
        
        # State vectors
        self.current_attacker_intent = {
            'intensity': 0.0,
            'is_attack': 0.0,
            'syn_rate': 0.1,
            'udp_rate': 0.1,
            'http_rate': 0.1,
            'icmp_rate': 0.0,
            'src_ip_ent': 0.5,
            'cost': 0.0,
            'obviousness': 0.0
        }
        
    def reset(self):
        self.current_step = 0
        self.prev_defender_action = 0.0
        
        self.history_sq.clear()
        self.history_cd.clear()
        self.history_leak.clear()
        self.history_sla.clear()
        self.history_action.clear()
        self.history_repairs.clear()
        
        self._apply_attacker_action(np.array([0, 0, 0, 1, 0])) # Benign reset
        return self.get_defender_obs(), self.get_attacker_obs()
        
    def _apply_attacker_action(self, attacker_action):
        # Decode MultiDiscrete([4, 5, 5, 3, 3])
        intensity_idx = attacker_action[0]
        protocol_idx = attacker_action[1]
        pattern_idx = attacker_action[2]
        entropy_idx = attacker_action[3]
        size_idx = attacker_action[4]
        
        # 1. Intensity (0: low, 1: med, 2: high, 3: burst)
        # Note: if it's benign, we need a way to say "no attack". We assume intensity 0 could be benign or low.
        # Actually, if the attacker is always attacking, low is just low.
        intensity_map = {0: 0.2, 1: 0.5, 2: 0.8, 3: 1.0}
        base_int = intensity_map[intensity_idx]
        is_attack = 1.0 if base_int > 0.05 else 0.0
        
        # Temporal pattern (0: sustained, 1: bursty, 2: low_slow, 3: pulse, 4: shift)
        if pattern_idx == 1: # bursty
            base_int = base_int if self.current_step % 20 < 10 else 0.0
            is_attack = 1.0 if base_int > 0.05 else 0.0
        elif pattern_idx == 2: # low_slow
            base_int = min(0.3, base_int)
        elif pattern_idx == 3: # pulse
            base_int = base_int if self.current_step % 50 < 5 else 0.0
            is_attack = 1.0 if base_int > 0.05 else 0.0
            
        self.current_attacker_intent['intensity'] = base_int
        self.current_attacker_intent['is_attack'] = is_attack
        
        # Protocol (0: syn, 1: udp, 2: http, 3: icmp, 4: mixed)
        syn, udp, http, icmp = 0.1, 0.1, 0.1, 0.0
        if is_attack:
            if protocol_idx == 0: syn = 0.8
            elif protocol_idx == 1: udp = 0.8
            elif protocol_idx == 2: http = 0.8
            elif protocol_idx == 3: icmp = 0.8
            elif protocol_idx == 4:
                phase = (self.current_step // 50) % 4
                if phase == 0: syn = 0.8
                elif phase == 1: udp = 0.8
                elif phase == 2: http = 0.8
                else: icmp = 0.8
        self.current_attacker_intent['syn_rate'] = syn
        self.current_attacker_intent['udp_rate'] = udp
        self.current_attacker_intent['http_rate'] = http
        self.current_attacker_intent['icmp_rate'] = icmp
        
        # Entropy (0: low, 1: normal, 2: high)
        if entropy_idx == 0: src_ent = 0.1
        elif entropy_idx == 1: src_ent = 0.5
        else: src_ent = 0.9
        self.current_attacker_intent['src_ip_ent'] = src_ent
        
        # Cost and obviousness tracking
        self.current_attacker_intent['cost'] = base_int * (1.5 if entropy_idx == 2 else 1.0)
        self.current_attacker_intent['obviousness'] = base_int * (1.0 if pattern_idx == 0 else 0.5)

    def get_defender_obs(self):
        intent = self.current_attacker_intent
        legit_volume = self.base_legitimate_rate
        
        vol = min(1.0, legit_volume + intent['intensity'])
        pkt_rate = min(1.0, legit_volume + intent['intensity'] * 1.5)
        
        # Est attack indicator (Volatility)
        est_attack = min(1.0, max(0.0, (vol - self.base_legitimate_rate) / 0.5))
        
        obs = np.array([
            vol,               
            pkt_rate,          
            intent['syn_rate'],          
            intent['udp_rate'],          
            intent['http_rate'],         
            intent['icmp_rate'],         
            intent['src_ip_ent'],        
            0.5,               
            0.5,               
            vol * 0.8,         
            vol * 0.9,         
            max(0, vol - 0.8), 
            self.prev_defender_action,  
            legit_volume / max(1e-5, vol), 
            est_attack
        ], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)
        
    def get_attacker_obs(self):
        # Non-oracle features
        rolling_sq = np.mean(self.history_sq) if self.history_sq else 1.0
        rolling_leak = np.mean(self.history_leak) if self.history_leak else 0.0
        rolling_action = np.mean(self.history_action) if self.history_action else 0.0
        rolling_repairs = np.mean(self.history_repairs) if self.history_repairs else 0.0
        rolling_sla = np.mean(self.history_sla) if self.history_sla else 0.0
        
        obs = np.array([
            rolling_sq,
            rolling_leak,
            rolling_action,
            rolling_repairs,
            rolling_sla,
            float(self.current_step) / self.max_steps
        ], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)
        
    def get_shield_context(self):
        rolling_sq = np.mean(self.history_sq) if self.history_sq else 1.0
        rolling_sla_rate = np.mean(self.history_sla) if self.history_sla else 0.0
        return {
            'rolling_service_quality': float(rolling_sq),
            'rolling_sla_violation_rate': float(rolling_sla_rate)
        }

    def step(self, defender_action, attacker_action, shield=None):
        self._apply_attacker_action(attacker_action)
        
        defender_obs = self.get_defender_obs()
        drop_intensity = np.clip(defender_action[0], 0.0, 1.0)
        
        # Shielding
        shield_repaired = 0.0
        if shield:
            context = self.get_shield_context()
            safe_action, s_info = shield.repair_action(defender_obs, defender_action, context)
            drop_intensity = safe_action[0]
            if s_info['total_shield_repair']:
                shield_repaired = 1.0
                
        # Mitigation math
        is_attack = self.current_attacker_intent['is_attack']
        legit_sent = self.base_legitimate_rate
        attack_sent = self.current_attacker_intent['intensity']
        
        legit_dropped = drop_intensity * (0.2 + (1.0 - is_attack) * 0.2) * legit_sent
        attack_dropped = drop_intensity * (0.7 + is_attack * 0.2) * attack_sent
        
        # Metrics
        sq = evaluator.service_quality(legit_sent, legit_dropped)
        collateral = evaluator.collateral_damage(legit_sent, legit_dropped)
        sla_violation = evaluator.safety_violation_count(sq, sla_threshold=0.95)
        me = evaluator.mitigation_efficiency(attack_sent, attack_dropped)
        leakage = evaluator.attack_leakage(attack_sent, attack_dropped)
        
        self.history_sq.append(sq)
        self.history_cd.append(collateral)
        self.history_leak.append(leakage)
        self.history_sla.append(sla_violation)
        self.history_action.append(drop_intensity)
        self.history_repairs.append(shield_repaired)
        
        # Calculate Rewards
        def_reward = self._calc_defender_reward(sq, me, sla_violation, collateral, leakage, drop_intensity)
        att_reward = self._calc_attacker_reward(leakage, sla_violation, collateral, shield_repaired)
        
        self.prev_defender_action = drop_intensity
        self.current_step += 1
        
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        info = {
            'sq': sq, 'me': me, 'collateral': collateral, 'sla_violation': sla_violation,
            'leakage': leakage, 'action': drop_intensity, 'shield_repair': shield_repaired,
            'is_attack': is_attack
        }
        
        return defender_obs, self.get_attacker_obs(), def_reward, att_reward, terminated, truncated, info

    def _calc_defender_reward(self, sq, me, sla_violation, collateral, leakage, drop_intensity):
        w_sq = self.defender_reward_config.get("w_sq", 1.0)
        w_me = self.defender_reward_config.get("w_me", 0.0)
        w_sla = self.defender_reward_config.get("w_sla", 5.0)
        w_cd = self.defender_reward_config.get("w_cd", 2.0)
        w_leak = self.defender_reward_config.get("w_leak", 2.0)
        w_cost = self.defender_reward_config.get("w_cost", 0.05)
        w_action_change = self.defender_reward_config.get("w_action_change", 0.01)
        w_temp_sla = self.defender_reward_config.get("w_temporal_sla", 0.0)
        
        r_sq = w_sq * sq
        r_me = w_me * me
        p_sla = -w_sla * sla_violation
        p_cd = -w_cd * collateral
        p_leak = -w_leak * leakage
        p_cost = -w_cost * drop_intensity
        p_change = -w_action_change * abs(drop_intensity - self.prev_defender_action)
        
        rolling_sla_rate = np.mean(self.history_sla) if self.history_sla else 0.0
        p_temp = -w_temp_sla * rolling_sla_rate
        
        return float(r_sq + r_me + p_sla + p_cd + p_leak + p_cost + p_change + p_temp)
        
    def _calc_attacker_reward(self, leakage, sla_violation, collateral, shield_repaired):
        w_leak = self.attacker_reward_config.get("w_leak", 2.0)
        w_sla = self.attacker_reward_config.get("w_sla", 2.0)
        w_cd = self.attacker_reward_config.get("w_cd", 1.0)
        w_repair = self.attacker_reward_config.get("w_repair", 0.5)
        w_cost = self.attacker_reward_config.get("w_cost", 0.2)
        w_obvious = self.attacker_reward_config.get("w_obvious", 0.1)
        
        cost = self.current_attacker_intent['cost']
        obvious = self.current_attacker_intent['obviousness']
        
        if self.attacker_objective == "stealthy":
            return float((w_leak * leakage) - (w_cost * cost) - (w_obvious * obvious))
        elif self.attacker_objective == "stress":
            return float((w_sla * sla_violation) + (w_cd * collateral) + (w_repair * shield_repaired) - (w_cost * cost * 0.1))
        else: # mixed
            return float((w_leak * leakage) + (w_sla * sla_violation) + (w_cd * collateral) + (w_repair * shield_repaired) - (w_cost * cost) - (w_obvious * obvious))

class DefenderTrainingEnv(gym.Env):
    def __init__(self, coevo_env, attacker_policy, shield=None):
        super().__init__()
        self.coevo_env = coevo_env
        self.attacker_policy = attacker_policy
        self.shield = shield
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(15,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        def_obs, att_obs = self.coevo_env.reset()
        self.last_att_obs = att_obs
        return def_obs, {}
        
    def step(self, action):
        att_action, _ = self.attacker_policy.predict(self.last_att_obs)
        def_obs, att_obs, def_reward, att_reward, term, trunc, info = self.coevo_env.step(action, att_action, self.shield)
        self.last_att_obs = att_obs
        return def_obs, def_reward, term, trunc, info

class AttackerTrainingEnv(gym.Env):
    def __init__(self, coevo_env, defender_policy, shield=None):
        super().__init__()
        self.coevo_env = coevo_env
        self.defender_policy = defender_policy
        self.shield = shield
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([4, 5, 5, 3, 3])
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        def_obs, att_obs = self.coevo_env.reset()
        self.last_def_obs = def_obs
        return att_obs, {}
        
    def step(self, action):
        def_action, _ = self.defender_policy.predict(self.last_def_obs)
        def_obs, att_obs, def_reward, att_reward, term, trunc, info = self.coevo_env.step(def_action, action, self.shield)
        self.last_def_obs = def_obs
        return att_obs, att_reward, term, trunc, info
