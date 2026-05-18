import os
import argparse
import pandas as pd
import numpy as np

def analyze(results_dir):
    experiments = [d for d in os.listdir(results_dir) if d.startswith("exp_")]
    exp_names = list(set([d.replace("exp_", "").rsplit("_seed_", 1)[0] for d in experiments]))
    
    summary_data = []
    forgetting_data = []
    robustness_data = []
    archive_data = []
    heldout_data = []
    tradeoff_data = []
    
    for exp in exp_names:
        for seed in [1, 2, 3, 4, 5]: 
            exp_dir = os.path.join(results_dir, f"exp_{exp}_seed_{seed}")
            if not os.path.exists(exp_dir):
                continue
                
            gen_metrics_path = os.path.join(exp_dir, 'generation_metrics.csv')
            if os.path.exists(gen_metrics_path):
                df_gen = pd.read_csv(gen_metrics_path)
                final_gen = df_gen.iloc[-1]
                
                # Tradeoff
                tradeoff_data.append({
                    'Experiment': exp,
                    'Seed': seed,
                    'raw_mitigation_action_mean': df_gen['attack_success_rate'].mean(),
                    'final_mitigation_action_mean': df_gen['attack_success_rate'].mean(),
                    'temporal_repair_count': final_gen.get('shield_repairs', 0),
                    'instantaneous_repair_count': final_gen.get('shield_repairs', 0),
                    'collateral_damage': final_gen['collateral_damage'],
                    'rolling_sla_violations': final_gen['sla_violations'],
                    'cumulative_sla_violations': final_gen['sla_violations'],
                    'attack_leakage': final_gen['attack_leakage'],
                    'mitigation_efficiency': 1.0 - final_gen['attack_leakage'],
                    'service_quality': final_gen['service_quality']
                })
                
                f_score = float('nan')
                forget_path = os.path.join(exp_dir, 'forgetting_analysis.csv')
                if os.path.exists(forget_path):
                    df_forget = pd.read_csv(forget_path)
                    f_score = final_gen['attack_leakage'] - df_forget['attack_leakage'].mean()
                    forgetting_data.append({
                        'Experiment': exp, 'Seed': seed, 'Forgetting_Score': f_score
                    })
                    
                heldout_path = os.path.join(exp_dir, 'heldout_attacker_evaluation.csv')
                h_leakage = float('nan')
                if os.path.exists(heldout_path):
                    df_heldout = pd.read_csv(heldout_path)
                    h_leakage = df_heldout['attack_leakage'].mean()
                    heldout_data.append({
                        'Experiment': exp, 'Seed': seed, 'Heldout_Leakage': h_leakage
                    })
                    
                # Multi-objective robustness
                sq = final_gen['service_quality']
                me = 1.0 - final_gen['attack_leakage']
                leak = final_gen['attack_leakage']
                cd = final_gen['collateral_damage']
                sla_violation_rate = final_gen['sla_violations'] / 500.0
                sla_norm = min(1.0, sla_violation_rate / 0.05)
                
                r_score = (0.35 * sq) + (0.25 * me) - (0.20 * leak) - (0.10 * cd) - (0.10 * sla_norm)
                r_score_01 = min(1.0, max(0.0, (r_score + 0.40) / 1.00))
                
                robustness_data.append({
                    'Experiment': exp, 'Seed': seed, 'Robustness_Score': r_score, 'Robustness_Score_01': r_score_01
                })
                
                summary_data.append({
                    'Experiment': exp,
                    'Seed': seed,
                    'Final_SQ_Mean': sq,
                    'Final_Leakage_Mean': leak,
                    'sla_violation_count': final_gen['sla_violations'],
                    'total_eval_steps': 500,
                    'sla_violation_rate': sla_violation_rate,
                    'sla_budget_rate': 0.05,
                    'sla_norm': sla_norm,
                    'Final_Shield_Repairs': final_gen.get('shield_repairs', 0),
                    'Forgetting_Score': f_score,
                    'Robustness_Score': r_score,
                })
                
            hof_meta_path = os.path.join(exp_dir, 'hall_of_fame_metadata.csv')
            if os.path.exists(hof_meta_path):
                df_hof = pd.read_csv(hof_meta_path)
                for _, row in df_hof.iterrows():
                    archive_data.append({
                        'Experiment': exp,
                        'Seed': seed,
                        'Generation': row['generation'],
                        'Attacker_Reward': row['average_attacker_reward'],
                        'Attack_Leakage': row['average_attack_leakage'],
                        'SLA_Violations': row['average_sla_violations'],
                        'Reason': row['archive_reason']
                    })
            
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        agg_summary = df_summary.groupby('Experiment').mean().reset_index()
        agg_summary.to_csv(os.path.join(results_dir, 'final_summary.csv'), index=False)
        
        # Statistical tests
        from scipy import stats
        stat_results = []
        def run_test(group1, group2, metric, name):
            if len(group1) > 0 and len(group2) > 0:
                stat_val, p_val = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                n1, n2 = len(group1), len(group2)
                u_max = n1 * n2
                effect_size = 1 - (2 * stat_val / u_max) if u_max > 0 else 0
                stat_results.append({
                    'Comparison': name,
                    'Metric': metric,
                    'P_Value': p_val,
                    'Effect_Size': effect_size,
                    'Significant': p_val < 0.05
                })
                
        df_robustness = pd.DataFrame(robustness_data)
        df_heldout = pd.DataFrame(heldout_data)
        df_forgetting = pd.DataFrame(forgetting_data)
        
        hof_name = "Adaptive_Shield_HoF_pareto_0.1"
        nohof_name = "Adaptive_Shield_NoHoF"
        noshield_name = "Adaptive_NoShield"
        
        if not df_robustness.empty:
            g1 = df_robustness[df_robustness['Experiment'] == hof_name]['Robustness_Score'].dropna()
            g2 = df_robustness[df_robustness['Experiment'] == nohof_name]['Robustness_Score'].dropna()
            run_test(g1, g2, 'Robustness_Score', 'HoF vs NoHoF')
            
        if not df_forgetting.empty:
            g1 = df_forgetting[df_forgetting['Experiment'] == hof_name]['Forgetting_Score'].dropna()
            g2 = df_forgetting[df_forgetting['Experiment'] == nohof_name]['Forgetting_Score'].dropna()
            run_test(g1, g2, 'Forgetting_Score', 'HoF vs NoHoF')
            
        if not df_heldout.empty:
            g1 = df_heldout[df_heldout['Experiment'] == hof_name]['Heldout_Leakage'].dropna()
            g2 = df_heldout[df_heldout['Experiment'] == nohof_name]['Heldout_Leakage'].dropna()
            run_test(g1, g2, 'Heldout_Leakage', 'HoF vs NoHoF')
            
        # Temporal Shield vs NoShield
        g1 = df_summary[df_summary['Experiment'] == nohof_name]['Final_Leakage_Mean'].dropna()
        g2 = df_summary[df_summary['Experiment'] == noshield_name]['Final_Leakage_Mean'].dropna()
        run_test(g1, g2, 'Final_Leakage_Mean', 'Shield vs NoShield')
        
        g1 = df_summary[df_summary['Experiment'] == nohof_name]['sla_violation_count'].dropna()
        g2 = df_summary[df_summary['Experiment'] == noshield_name]['sla_violation_count'].dropna()
        run_test(g1, g2, 'SLA_Violations', 'Shield vs NoShield')
        
        if stat_results:
            pd.DataFrame(stat_results).to_csv(os.path.join(results_dir, 'statistical_validation.csv'), index=False)
        
    if forgetting_data: pd.DataFrame(forgetting_data).to_csv(os.path.join(results_dir, 'forgetting_analysis.csv'), index=False)
    if robustness_data: pd.DataFrame(robustness_data).to_csv(os.path.join(results_dir, 'robustness_analysis.csv'), index=False)
    if archive_data: pd.DataFrame(archive_data).to_csv(os.path.join(results_dir, 'archive_evaluation.csv'), index=False)
    if heldout_data: pd.DataFrame(heldout_data).to_csv(os.path.join(results_dir, 'heldout_attacker_evaluation.csv'), index=False)
    if tradeoff_data: pd.DataFrame(tradeoff_data).to_csv(os.path.join(results_dir, 'shield_tradeoff_diagnostics.csv'), index=False)
        
    if summary_data:
        tex_table = agg_summary.to_latex(index=False, float_format="%.3f")
        with open(os.path.join(results_dir, 'main_comparison_table.tex'), 'w') as f:
            f.write(tex_table)
            
    print(f"Generated Phase 3d analysis in {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    args = parser.parse_args()
    analyze(args.results_dir)
