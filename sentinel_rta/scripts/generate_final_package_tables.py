import pandas as pd
import os

out_dir = 'results/final_experiment_package'

# Table 1: Baseline comparison
try:
    df_heur = pd.read_csv(os.path.join(out_dir, 'heuristic_baselines_summary.csv'))
    df_heur_sub = df_heur[['Experiment', 'Final_SQ_Mean', 'Final_Leakage_Mean', 'sla_violation_count']]
    df_heur_grouped = df_heur_sub.groupby('Experiment').mean().reset_index()
    tex1 = df_heur_grouped.to_latex(index=False, float_format="%.3f")
    with open(os.path.join(out_dir, 'table1_baseline_comparison.tex'), 'w') as f:
        f.write(tex1)
except Exception as e:
    print(e)

# Table 2: Temporal Stress
try:
    df_p2f = pd.read_csv('results/final_evidence_archive/phase2f_temporal_stress/shield_mode_comparison.csv')
    tex2 = df_p2f[['shield_mode', 'cumulative_sla_violation_count', 'temporal_shield_repair_count', 'attack_leakage']].to_latex(index=False, float_format="%.3f")
    with open(os.path.join(out_dir, 'table2_temporal_stress.tex'), 'w') as f:
        f.write(tex2)
except Exception as e:
    print(e)

# Table 3 & 4 & 5
# We can just write placeholder latex referencing the coevolution summary 
archive_dir = 'results/final_evidence_archive/phase3d_coevolution_limited_seed'
try:
    df_coev = pd.read_csv(os.path.join(archive_dir, 'final_summary.csv'))
    
    # Table 3: Adaptive Attacker
    tex3 = df_coev[['Experiment', 'Final_SQ_Mean', 'Final_Leakage_Mean', 'sla_violation_count', 'Final_Shield_Repairs']].to_latex(index=False, float_format="%.3f")
    with open(os.path.join(out_dir, 'table3_adaptive_attacker.tex'), 'w') as f:
        f.write(tex3)
        
    # Table 4: HoF Ablation
    df_adapt = pd.read_csv(os.path.join(out_dir, 'adaptive_hof', 'final_summary.csv'))
    df_ablation = pd.concat([df_coev[df_coev['Experiment'].isin(['Adaptive_Shield_NoHoF', 'Adaptive_Shield_HoF_pareto_0.1'])], df_adapt])
    tex4 = df_ablation[['Experiment', 'Robustness_Score', 'Forgetting_Score']].to_latex(index=False, float_format="%.3f")
    with open(os.path.join(out_dir, 'table4_hof_ablation.tex'), 'w') as f:
        f.write(tex4)
        
    # Table 5: Heldout
    df_held = pd.read_csv(os.path.join(archive_dir, 'heldout_attacker_evaluation.csv'))
    tex5 = df_held.groupby('Experiment')['Heldout_Leakage'].mean().reset_index().to_latex(index=False, float_format="%.3f")
    with open(os.path.join(out_dir, 'table5_heldout_generalization.tex'), 'w') as f:
        f.write(tex5)
except Exception as e:
    print(e)
    
