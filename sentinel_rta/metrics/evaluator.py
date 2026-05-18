import numpy as np

def service_quality(legitimate_sent, legitimate_dropped):
    return max(0.0, 1.0 - (legitimate_dropped / max(1e-5, legitimate_sent)))

def mitigation_efficiency(attack_sent, attack_dropped):
    if attack_sent <= 0: return 1.0
    return max(0.0, min(1.0, attack_dropped / attack_sent))

def safety_violation_count(sq, sla_threshold=0.95):
    return 1 if sq < sla_threshold else 0

def shield_repair_count(was_repaired):
    return 1 if was_repaired else 0

def attack_leakage(attack_sent, attack_dropped):
    if attack_sent <= 0: return 0.0
    return max(0.0, (attack_sent - attack_dropped) / attack_sent)

def collateral_damage(legitimate_sent, legitimate_dropped):
    return min(1.0, legitimate_dropped / max(1e-5, legitimate_sent))

def sla_violation_rate(total_violations, total_steps):
    return total_violations / max(1, total_steps)

def recovery_time(sq_history, attack_shift_index, target_sq=0.95):
    for i, sq in enumerate(sq_history[attack_shift_index:]):
        if sq >= target_sq:
            return i
    return len(sq_history) - attack_shift_index

def average_reward(rewards):
    return np.mean(rewards) if len(rewards) > 0 else 0.0
