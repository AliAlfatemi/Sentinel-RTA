import numpy as np
import random
import uuid

class HallOfFame:
    def __init__(self, config=None):
        if config is None:
            config = {}
            
        self.max_size = config.get('max_size', 10)
        self.diversity_threshold = config.get('diversity_threshold', 0.15)
        self.reward_margin = config.get('reward_margin', 0.05)
        self.leakage_threshold = config.get('leakage_threshold', 0.60)
        self.sla_threshold = config.get('sla_threshold', 10)
        self.repair_threshold = config.get('repair_threshold', 25)
        self.admission_mode = config.get('admission_mode', 'diversity_aware')
        
        self.archive = []
        
    def _is_diverse(self, new_dist):
        if not self.archive:
            return True
        for _, existing_m in self.archive:
            existing_dist = existing_m.get('attack_type_distribution', [0]*5)
            # Add padding if needed
            while len(existing_dist) < 5: existing_dist.append(0)
            while len(new_dist) < 5: new_dist.append(0)
            dist_diff = np.sum(np.abs(np.array(new_dist) - np.array(existing_dist))) / 2.0
            if dist_diff < self.diversity_threshold:
                return False
        return True

    def _is_pareto_nondominated(self, new_meta):
        if not self.archive:
            return True
            
        # We want to MAXIMIZE: reward, leakage, sla, diversity, repairs
        new_vec = np.array([
            new_meta.get('average_attacker_reward', 0.0),
            new_meta.get('average_attack_leakage', 0.0),
            new_meta.get('average_sla_violations', 0.0),
            new_meta.get('average_shield_repairs', 0.0)
        ])
        
        for _, ex_meta in self.archive:
            ex_vec = np.array([
                ex_meta.get('average_attacker_reward', 0.0),
                ex_meta.get('average_attack_leakage', 0.0),
                ex_meta.get('average_sla_violations', 0.0),
                ex_meta.get('average_shield_repairs', 0.0)
            ])
            
            # If existing vector is >= in all dimensions and > in at least one, new is dominated
            if np.all(ex_vec >= new_vec) and np.any(ex_vec > new_vec):
                return False
        return True
        
    def add(self, attacker_policy, metadata):
        att_reward = metadata.get('average_attacker_reward', 0.0)
        leakage = metadata.get('average_attack_leakage', 0.0)
        sla_violation = metadata.get('average_sla_violations', 0.0)
        repairs = metadata.get('average_shield_repairs', 0.0)
        att_dist = metadata.get('attack_type_distribution', [0]*5)
        
        # Diversity score
        is_diverse = self._is_diverse(att_dist)
        
        if len(self.archive) == 0:
            median_reward = 0.0
        else:
            rewards = [m['average_attacker_reward'] for _, m in self.archive]
            median_reward = np.median(rewards)
            
        reason = None
        
        if self.admission_mode == 'reward_only':
            if att_reward > median_reward + self.reward_margin:
                reason = "reward_margin"
        elif self.admission_mode == 'leakage_pressure':
            if leakage > self.leakage_threshold:
                reason = "high_leakage"
        elif self.admission_mode == 'sla_pressure':
            if sla_violation > self.sla_threshold:
                reason = "high_sla"
        elif self.admission_mode == 'diversity_aware':
            if (att_reward > median_reward + self.reward_margin or leakage > self.leakage_threshold) and is_diverse:
                reason = "diverse_and_high_threat"
        elif self.admission_mode == 'pareto':
            if self._is_pareto_nondominated(metadata):
                reason = "pareto_nondominated"
                
        if reason is None:
            return False, f"Failed admission: mode={self.admission_mode}"
            
        attacker_id = str(uuid.uuid4())[:8]
        metadata['attacker_id'] = attacker_id
        metadata['archive_reason'] = reason
        metadata['diversity_score_to_archive'] = 1.0 if is_diverse else 0.0 # simplified tracking
        
        self.archive.append((attacker_policy, metadata))
        
        # Sort and trim
        self.archive.sort(key=lambda x: x[1]['average_attacker_reward'], reverse=True)
        if len(self.archive) > self.max_size:
            self.archive = self.archive[:self.max_size]
            
        return True, reason
        
    def sample(self, k=1):
        if not self.archive:
            return []
        sampled = random.choices(self.archive, k=k)
        return [policy for policy, _ in sampled]
        
    def get_metadata(self):
        return [m for _, m in self.archive]
