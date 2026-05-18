import numpy as np

def calculate_sla_norm(sla_violation_count, total_eval_steps, sla_budget_rate=0.05):
    """
    Calculates the normalized SLA violation score bounded at 1.0.
    """
    sla_violation_rate = sla_violation_count / max(total_eval_steps, 1)
    return min(1.0, sla_violation_rate / sla_budget_rate)

def calculate_robustness_raw(service_quality, mitigation_efficiency, attack_leakage, collateral_damage, sla_norm):
    """
    Multi-objective robustness formulation.
    robustness_raw = 0.35 * service_quality + 0.25 * mitigation_efficiency - 0.20 * attack_leakage - 0.10 * collateral_damage - 0.10 * sla_norm
    """
    return (0.35 * service_quality) + (0.25 * mitigation_efficiency) - (0.20 * attack_leakage) - (0.10 * collateral_damage) - (0.10 * sla_norm)

def calculate_robustness_score_01(robustness_raw):
    """
    Scales the raw robustness into a 0-1 range for plotting bounds.
    """
    return min(1.0, max(0.0, (robustness_raw + 0.40) / 1.00))

def calculate_forgetting_score(final_leakage, early_leakage):
    """
    Forgetting score = final_leakage_against_old_attackers - early_leakage_against_old_attackers
    Interpretation:
    - positive = forgetting / worse retention
    - zero = no change
    - negative = improved retention
    """
    return final_leakage - early_leakage
