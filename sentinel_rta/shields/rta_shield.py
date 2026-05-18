import numpy as np

class RTAShield:
    """
    Runtime Assurance (RTA) Shield for Sentinel-RTA.
    Projects unsafe DRL actions into an operator-defined safe action set,
    supporting both instantaneous and temporal constraints.
    """
    def __init__(self, max_collateral_tolerance=0.05, temporal_enabled=False, temporal_recovery_max_action=0.05):
        self.max_collateral_tolerance = max_collateral_tolerance
        self.temporal_enabled = temporal_enabled
        self.temporal_recovery_max_action = temporal_recovery_max_action
        
    def check_safety(self, obs, raw_action):
        """
        Check if the action violates safety constraints.
        Returns True if safe, False if unsafe.
        """
        est_attack = obs[14]
        drop_intensity = raw_action[0]
        
        # Constraint: Do not drop heavily if attack confidence is zero/low
        if est_attack < 0.2 and drop_intensity > 0.1:
            return False
            
        return True
        
    def repair_action(self, obs, raw_action, temporal_state=None):
        """
        Projects an unsafe action to the nearest safe action.
        Returns:
            safe_action (np.array): The bounded safe action.
            shield_info (dict): Diagnostic info mapping repair types.
        """
        est_attack = obs[14]
        safe_action = raw_action.copy()
        drop_intensity = raw_action[0]
        
        is_inst_repair = False
        is_temp_repair = False
        repair_reason = "None"
        dynamic_max_action = 1.0
        
        # 1. Instantaneous Constraint
        if est_attack < 0.2 and drop_intensity > 0.1:
            safe_action[0] = 0.1
            is_inst_repair = True
            repair_reason = "Instantaneous: High drop during benign"
            dynamic_max_action = 0.1
            
        # 2. Temporal Constraint
        if self.temporal_enabled and temporal_state is not None:
            rolling_sq = temporal_state.get('rolling_service_quality', 1.0)
            rolling_sla_rate = temporal_state.get('rolling_sla_violation_rate', 0.0)
            
            # If rolling SQ drops below SLA threshold, tighten the max action
            if rolling_sq < 0.95 or rolling_sla_rate > 0.05:
                # Strict temporal bounding to reduce SLA-risk under configured simulator rules
                # If SLA is at risk, we limit the drop intensity to a safe threshold
                temp_max_action = self.temporal_recovery_max_action
                dynamic_max_action = min(dynamic_max_action, temp_max_action)
                
                if safe_action[0] > temp_max_action:
                    safe_action[0] = temp_max_action
                    if not is_inst_repair:
                        is_temp_repair = True
                        repair_reason = f"Temporal | recovery_bound={temp_max_action} | rolling_sla_risk={rolling_sla_rate:.3f} | rolling_sq={rolling_sq:.3f} | dynamic_max_action={dynamic_max_action}"
                        
        shield_info = {
            'instantaneous_shield_repair': is_inst_repair,
            'temporal_shield_repair': is_temp_repair,
            'total_shield_repair': is_inst_repair or is_temp_repair,
            'dynamic_max_action': dynamic_max_action,
            'repair_reason': repair_reason,
            'rolling_service_quality': temporal_state.get('rolling_service_quality', 1.0) if temporal_state else 1.0,
            'rolling_collateral_damage': temporal_state.get('rolling_collateral_damage', 0.0) if temporal_state else 0.0,
            'rolling_sla_violation_rate': temporal_state.get('rolling_sla_violation_rate', 0.0) if temporal_state else 0.0,
            'safety_budget_remaining': 1.0 - (temporal_state.get('rolling_sla_violation_rate', 0.0) if temporal_state else 0.0)
        }
            
        return safe_action, shield_info
