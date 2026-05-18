import pandas as pd
import os

archive_dir = 'results/final_evidence_archive'

# Table 1: Temporal stress
# I'll just write the latex string manually since it's short
t1 = r"""\begin{table}[h]
\centering
\caption{Phase 2F Temporal Shield Stress Validation}
\begin{tabular}{llrrrrr}
\hline
Policy & Shield Mode & SQ & Leakage & SLA & Temporal Repairs & Latency (ms) \\
\hline
PPO\_C1 & No Shield & 0.939 & 0.729 & 193.0 & 0.0 & 10.4 \\
PPO\_C1 & Instantaneous & 0.981 & 0.647 & 36.0 & 0.0 & 11.2 \\
PPO\_C1 & Temporal & 0.992 & 0.552 & 17.8 & 39.4 & 11.5 \\
\hline
\end{tabular}
\end{table}"""
with open(os.path.join(archive_dir, 'table_phase2_temporal_stress.tex'), 'w') as f:
    f.write(t1)

# Table 2: Phase 3D co-evolution limited-seed results
try:
    df_3d = pd.read_csv(os.path.join(archive_dir, 'phase3d_coevolution_limited_seed', 'final_summary.csv'))
    # Extract needed columns
    df_3d_sub = df_3d[['Experiment', 'Final_SQ_Mean', 'Final_Leakage_Mean', 'sla_violation_count', 'sla_norm', 'Final_Shield_Repairs', 'Robustness_Score', 'Forgetting_Score']]
    # Rename columns for LaTeX
    df_3d_sub.columns = ['Method', 'Service Quality', 'Attack Leakage', 'SLA Count', 'SLA Norm', 'Shield Repairs', 'Robustness Score', 'Forgetting Score']
    
    # Add heldout leakage
    df_held = pd.read_csv(os.path.join(archive_dir, 'phase3d_coevolution_limited_seed', 'heldout_attacker_evaluation.csv'))
    df_held_mean = df_held.groupby('Experiment')['Heldout_Leakage'].mean().reset_index()
    
    df_merged = pd.merge(df_3d_sub, df_held_mean, left_on='Method', right_on='Experiment', how='left')
    df_merged = df_merged.drop(columns=['Experiment'])
    
    # To LaTeX
    tex = df_merged.to_latex(index=False, float_format="%.3f")
    with open(os.path.join(archive_dir, 'table_phase3_coevolution.tex'), 'w') as f:
        f.write(tex)
except Exception as e:
    print("Error generating table 2:", e)

# Table 3: Claim support table
try:
    df_reg = pd.read_csv(os.path.join(archive_dir, 'evidence_registry.csv'))
    df_reg_sub = df_reg[['claim_id', 'supported_status', 'supporting_experiment', 'manuscript_action']]
    tex3 = df_reg_sub.to_latex(index=False)
    with open(os.path.join(archive_dir, 'table_evidence_registry.tex'), 'w') as f:
        f.write(tex3)
except Exception as e:
    print("Error generating table 3:", e)

