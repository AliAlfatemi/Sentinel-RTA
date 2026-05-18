import pandas as pd
import numpy as np
import os

out_dir = 'results/manuscript_results_package/tables'

def clean_name(name):
    if not isinstance(name, str):
        return name
    
    mapping = {
        'Adaptive_NoShield': 'Adaptive No Shield',
        'Static_NoShield': 'Static No Shield',
        'Adaptive_Shield_HoF_pareto_0.1': 'Adaptive Shield HoF (Pareto, 0.1)',
        'Adaptive_Shield_NoHoF': 'Adaptive Shield NoHoF'
    }
    
    if name in mapping:
        return mapping[name]
        
    return name.replace('_', ' ')

# Table 1: Baseline comparison
try:
    src1 = 'results/manuscript_results_package/source_csv/heuristic_baselines_summary.csv'
    df_heur = pd.read_csv(src1)
    df_heur['Experiment'] = df_heur['Experiment'].apply(clean_name)
    df_heur_sub = df_heur[['Experiment', 'Final_SQ_Mean', 'Final_Leakage_Mean', 'sla_violation_count']]
    
    # Calculate mean and std
    grouped = df_heur_sub.groupby('Experiment').agg(['mean', 'std'])
    
    formatted = pd.DataFrame()
    formatted['Method'] = grouped.index
    formatted['Service Quality $\\uparrow$'] = grouped['Final_SQ_Mean'].apply(lambda x: f"{x['mean']:.3f} $\\pm$ {x['std']:.3f}", axis=1).values
    formatted['Attack Leakage $\\downarrow$'] = grouped['Final_Leakage_Mean'].apply(lambda x: f"{x['mean']:.3f} $\\pm$ {x['std']:.3f}", axis=1).values
    formatted['SLA Violations $\\downarrow$'] = grouped['sla_violation_count'].apply(lambda x: f"{x['mean']:.3f} $\\pm$ {x['std']:.3f}", axis=1).values
    
    tex1 = f"% Source: {src1}\n" + formatted.to_latex(index=False, escape=False)
    with open(os.path.join(out_dir, 'table1_baseline_comparison.tex'), 'w') as f:
        f.write(tex1)
except Exception as e:
    print("Table 1 Error:", e)

# Table 2: Temporal Stress
try:
    src2 = 'results/manuscript_results_package/source_csv/shield_mode_comparison.csv'
    df_p2f = pd.read_csv(src2)
    df_p2f['shield_mode'] = df_p2f['shield_mode'].apply(clean_name)
    
    formatted = pd.DataFrame()
    formatted['Method'] = df_p2f['shield_mode']
    formatted['Cumulative SLA Violations $\\downarrow$'] = df_p2f['cumulative_sla_violation_count']
    formatted['Temporal Repairs'] = df_p2f['temporal_shield_repair_count']
    formatted['Attack Leakage $\\downarrow$'] = df_p2f['attack_leakage'].apply(lambda x: f"{x:.3f}")
    
    tex2 = f"% Source: {src2}\n" + formatted.to_latex(index=False, escape=False)
    with open(os.path.join(out_dir, 'table2_temporal_stress.tex'), 'w') as f:
        f.write(tex2)
except Exception as e:
    print("Table 2 Error:", e)

# Coevolution summary path
archive_dir = 'results/manuscript_results_package/source_csv'
src3 = os.path.join(archive_dir, 'final_summary.csv')

try:
    df_coev = pd.read_csv(src3)
    df_coev['Experiment'] = df_coev['Experiment'].apply(clean_name)
    
    def format_mean_std(x):
        if pd.isna(x['std']):
            return f"{x['mean']:.3f}"
        return f"{x['mean']:.3f} $\\pm$ {x['std']:.3f}"

    # Table 3: Adaptive Attacker
    grouped3 = df_coev.groupby('Experiment').agg(['mean', 'std'])
    formatted3 = pd.DataFrame()
    formatted3['Method (Extended Preliminary)'] = grouped3.index
    formatted3['Service Quality $\\uparrow$'] = grouped3['Final_SQ_Mean'].apply(format_mean_std, axis=1).values
    formatted3['Attack Leakage $\\downarrow$'] = grouped3['Final_Leakage_Mean'].apply(format_mean_std, axis=1).values
    formatted3['SLA Violations $\\downarrow$'] = grouped3['sla_violation_count'].apply(format_mean_std, axis=1).values
    formatted3['Shield Repairs'] = grouped3['Final_Shield_Repairs'].apply(format_mean_std, axis=1).values
    
    tex3 = f"% Source: {src3}\n% Note: Extended-preliminary aggregate; seed-level dispersion unavailable for some rows.\n" + formatted3.to_latex(index=False, escape=False)
    with open(os.path.join(out_dir, 'table3_adaptive_attacker.tex'), 'w') as f:
        f.write(tex3)
        
    # Table 4: HoF Ablation
    src4_1 = os.path.join(archive_dir, 'robustness_analysis.csv')
    src4_2 = os.path.join(archive_dir, 'forgetting_analysis.csv')
    
    df_rob = pd.read_csv(src4_1)
    df_for = pd.read_csv(src4_2)
    
    # Aggregate before merging
    df_rob_agg = df_rob.groupby('Experiment', as_index=False)['Robustness_Score'].mean()
    df_for_agg = df_for.groupby('Experiment', as_index=False)['Forgetting_Score'].mean()
    
    df_merged = pd.merge(df_rob_agg, df_for_agg, on='Experiment')
    df_merged['Experiment'] = df_merged['Experiment'].apply(clean_name)
    
    formatted4 = pd.DataFrame()
    formatted4['Method (Extended Preliminary)'] = df_merged['Experiment']
    formatted4['Robustness Score $\\uparrow$'] = df_merged['Robustness_Score'].apply(lambda x: f"{x:.3f}")
    formatted4['Forgetting Score $\\downarrow$'] = df_merged['Forgetting_Score'].apply(lambda x: f"{x:.3f}")
    
    def get_interp(x):
        return "Regression" if x > 0 else "Improved Retention"
        
    formatted4['Interpretation'] = df_merged['Forgetting_Score'].apply(get_interp)
    
    tex4 = f"% Source: {src4_1}, {src4_2}\n" + formatted4.to_latex(index=False, escape=False)
    with open(os.path.join(out_dir, 'table4_hof_ablation.tex'), 'w') as f:
        f.write(tex4)
        
    # Table 5: Heldout
    src5 = os.path.join(archive_dir, 'heldout_attacker_evaluation.csv')
    df_held = pd.read_csv(src5)
    df_held['Experiment'] = df_held['Experiment'].apply(clean_name)
    
    grouped5 = df_held.groupby('Experiment').agg(['mean', 'std'])
    formatted5 = pd.DataFrame()
    formatted5['Method (Extended Preliminary)'] = grouped5.index
    formatted5['Heldout Leakage $\\downarrow$'] = grouped5['Heldout_Leakage'].apply(lambda x: f"{x['mean']:.3f} $\\pm$ {x['std']:.3f}", axis=1).values
    
    tex5 = f"% Source: {src5}\n" + formatted5.to_latex(index=False, escape=False)
    with open(os.path.join(out_dir, 'table5_heldout_generalization.tex'), 'w') as f:
        f.write(tex5)
        
except Exception as e:
    print("Error parsing CoEvolution CSVs:", e)

