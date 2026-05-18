import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob

# IEEE configurations
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,
    "lines.markersize": 5
})

SINGLE_COLUMN = (3.5, 2.5)
DOUBLE_COLUMN = (7.16, 3.0)

out_dir = "results/manuscript_results_package/figures"
os.makedirs(out_dir, exist_ok=True)

def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{name}.png"), format="png", dpi=600, bbox_inches="tight")
    plt.close(fig)

def clean_name(x):
    if not isinstance(x, str): return x
    x_clean = x.replace('_', ' ')
    mapping = {
        'Random Defender': 'Random',
        'Static Threshold': 'Static Threshold',
        'Adaptive Threshold': 'Adaptive Threshold',
        'Shield Only': 'Shield-only',
        'No Shield': 'No Shield',
        'Instantaneous Runtime Shield': 'Instantaneous RTA',
        'Temporal Runtime Shield': 'Temporal RTA',
        'Adaptive Shield NoHoF': 'Adaptive Shield NoHoF',
        'Adaptive Shield HoF pareto 0.1': 'Adaptive Shield HoF',
        'Adaptive NoShield': 'Adaptive No Shield',
        'Static NoShield': 'Static No Shield'
    }
    return mapping.get(x_clean, x_clean)

# ==========================================
# Figure 2: Baseline Defender Comparison
# ==========================================
try:
    df_base = pd.read_csv('results/manuscript_results_package/source_csv/heuristic_baselines_summary.csv')
    df_base['Experiment'] = df_base['Experiment'].apply(clean_name)
    grouped = df_base.groupby('Experiment').agg({'Final_SQ_Mean': ['mean', 'std'], 
                                                 'Final_Leakage_Mean': ['mean', 'std'],
                                                 'sla_violation_count': ['mean', 'std']})
                                                 
    order = ['Random', 'Static Threshold', 'Adaptive Threshold', 'Shield-only']
    # Filter to only include available
    order = [o for o in order if o in grouped.index]
    grouped = grouped.loc[order]
    
    fig, axes = plt.subplots(1, 3, figsize=DOUBLE_COLUMN)
    x = np.arange(len(grouped))
    width = 0.5
    
    # Panel A: SQ
    axes[0].bar(x, grouped['Final_SQ_Mean']['mean'], width, yerr=grouped['Final_SQ_Mean']['std'], capsize=4, color='#4c72b0')
    axes[0].set_title('A. Service Quality (higher is better)')
    axes[0].set_ylabel('Service Quality')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(grouped.index, rotation=45, ha='right')
    
    # Panel B: Leakage
    axes[1].bar(x, grouped['Final_Leakage_Mean']['mean'], width, yerr=grouped['Final_Leakage_Mean']['std'], capsize=4, color='#dd8452')
    axes[1].set_title('B. Attack Leakage (lower is better)')
    axes[1].set_ylabel('Attack Leakage')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(grouped.index, rotation=45, ha='right')
    
    # Panel C: SLA
    axes[2].bar(x, grouped['sla_violation_count']['mean'], width, yerr=grouped['sla_violation_count']['std'], capsize=4, color='#c44e52')
    axes[2].set_title('C. SLA Violations (lower is better)')
    axes[2].set_ylabel('SLA Violations')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(grouped.index, rotation=45, ha='right')
    
    save_fig(fig, "fig2_baseline_comparison")
except Exception as e:
    print("Fig 2 error:", e)

# ==========================================
# Figure 3: Temporal Stress Summary
# ==========================================
try:
    df_stress = pd.read_csv('results/manuscript_results_package/source_csv/shield_mode_comparison.csv')
    df_stress['shield_mode'] = df_stress['shield_mode'].apply(clean_name)
    
    df_stress = df_stress.set_index('shield_mode')
    order = ['No Shield', 'Instantaneous RTA', 'Temporal RTA']
    order = [o for o in order if o in df_stress.index]
    df_stress = df_stress.loc[order]
    
    fig, axes = plt.subplots(1, 3, figsize=DOUBLE_COLUMN)
    x = np.arange(len(df_stress))
    width = 0.5
    
    axes[0].bar(x, df_stress['cumulative_sla_violation_count'], width, color='#4c72b0')
    axes[0].set_title('A. SLA Violations (lower is better)')
    axes[0].set_ylabel('SLA Violations')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_stress.index, rotation=45, ha='right')
    
    axes[1].bar(x, df_stress['attack_leakage'], width, color='#dd8452')
    axes[1].set_title('B. Attack Leakage (lower is better)')
    axes[1].set_ylabel('Attack Leakage')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_stress.index, rotation=45, ha='right')
    
    axes[2].bar(x, df_stress['temporal_shield_repair_count'], width, color='#55a868')
    axes[2].set_title('C. Temporal Repairs')
    axes[2].set_ylabel('Temporal Repairs')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(df_stress.index, rotation=45, ha='right')
    
    save_fig(fig, "fig3_temporal_stress_summary")
except Exception as e:
    print("Fig 3 error:", e)

# ==========================================
# Figure 5: Adaptive Leakage Generations
# ==========================================
try:
    gen_files = glob.glob("results/manuscript_results_package/source_csv/phase3d_generation_metrics/exp_*/generation_metrics.csv")
    if gen_files:
        gen_data = []
        for gf in gen_files:
            exp_name = gf.split('exp_')[1].split('_seed_')[0]
            exp_name = clean_name(exp_name)
            df = pd.read_csv(gf)
            df['Experiment'] = exp_name
            gen_data.append(df)
            
        df_all = pd.concat(gen_data)
        
        # Make full width
        fig, ax = plt.subplots(figsize=DOUBLE_COLUMN)
        final_gen_data = []
        
        for exp in df_all['Experiment'].unique():
            df_sub = df_all[df_all['Experiment'] == exp]
            grouped = df_sub.groupby('generation')['attack_leakage'].agg(['mean', 'std'])
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], label=exp, capsize=3)
            
            final_gen = grouped.index.max()
            final_gen_data.append({
                'Experiment': exp,
                'mean': grouped.loc[final_gen, 'mean'],
                'std': grouped.loc[final_gen, 'std']
            })
            
        ax.set_title('Attack Leakage Over Generations (lower is better)')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Attack Leakage')
        # Move legend below
        ax.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=4)
        fig.subplots_adjust(bottom=0.3)
        save_fig(fig, "fig5_adaptive_leakage_generations")
        
        # Fig 5b
        fig5b, ax5b = plt.subplots(figsize=SINGLE_COLUMN)
        df_final = pd.DataFrame(final_gen_data)
        x5b = np.arange(len(df_final))
        ax5b.bar(x5b, df_final['mean'], yerr=df_final['std'], capsize=4, color='#4c72b0')
        ax5b.set_title('Final Generation Leakage (lower is better)')
        ax5b.set_ylabel('Attack Leakage')
        ax5b.set_xticks(x5b)
        ax5b.set_xticklabels(df_final['Experiment'], rotation=45, ha='right')
        save_fig(fig5b, "fig5b_final_generation_leakage")
        
except Exception as e:
    print("Fig 5 error:", e)

# ==========================================
# Figure 6: HoF Ablation
# ==========================================
try:
    df_rob = pd.read_csv("results/manuscript_results_package/source_csv/robustness_analysis.csv")
    df_for = pd.read_csv("results/manuscript_results_package/source_csv/forgetting_analysis.csv")
    
    df_rob_agg = df_rob.groupby('Experiment', as_index=False)['Robustness_Score'].mean()
    df_for_agg = df_for.groupby('Experiment', as_index=False)['Forgetting_Score'].mean()
    
    df_merged = pd.merge(df_rob_agg, df_for_agg, on="Experiment")
    df_merged['Experiment'] = df_merged['Experiment'].apply(clean_name)
    
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COLUMN)
    y = np.arange(len(df_merged))
    height = 0.5
    
    axes[0].barh(y, df_merged['Robustness_Score'], height, color='#4c72b0')
    axes[0].set_title('A. Robustness Score (higher is better)')
    axes[0].set_xlabel('Robustness Score')
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df_merged['Experiment'])
    
    axes[1].barh(y, df_merged['Forgetting_Score'], height, color='#c44e52')
    axes[1].set_title('B. Forgetting Score (lower is better)')
    axes[1].set_xlabel('Forgetting Score')
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(df_merged['Experiment'])
    axes[1].axvline(0, color='black', linewidth=1)
    
    fig.subplots_adjust(bottom=0.3)
    save_fig(fig, "fig6_hof_ablation")
except Exception as e:
    print("Fig 6 error:", e)

# ==========================================
# Figure 7: Safety-Performance Trade-off
# ==========================================
# Note: Figure 7 is now generated exclusively by sentinel_rta/scripts/generate_fig7_ieee_v11.py

print("Figure generation completed.")
