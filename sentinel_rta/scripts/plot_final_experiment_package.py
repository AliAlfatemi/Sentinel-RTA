import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd
import seaborn as sns

def plot_final_metrics(results_dir):
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    
    # Load heuristic baselines
    df_heur = pd.read_csv(os.path.join(results_dir, 'heuristic_baselines_summary.csv'))
    df_heur['Source'] = 'Heuristic'
    
    # Load Co-evolution results
    # For a real run we'd load final_summary.csv from phase3d. We'll simulate reading the archived one.
    archive_dir = 'results/final_evidence_archive/phase3d_coevolution_limited_seed'
    df_coev = pd.read_csv(os.path.join(archive_dir, 'final_summary.csv'))
    df_coev['Source'] = 'CoEvolution'
    
    # 1. Baseline Comparison: Service Quality, Leakage, SLA Violations
    df_combined = pd.concat([df_heur, df_coev])
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_combined, x='Experiment', y='Final_SQ_Mean')
    plt.title('1. Baseline Comparison: Service Quality')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plots', '1_baseline_sq.png'))
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_combined, x='Experiment', y='Final_Leakage_Mean')
    plt.title('1. Baseline Comparison: Attack Leakage')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plots', '1_baseline_leakage.png'))
    plt.close()
    
    # 2. Temporal Shield Stress (Use phase2f data)
    df_p2f = pd.read_csv('results/final_evidence_archive/phase2f_temporal_stress/shield_mode_comparison.csv')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_p2f, x='shield_mode', y='cumulative_sla_violation_count')
    plt.title('2. Temporal Shield Stress: SLA Violations')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plots', '2_temporal_stress_sla.png'))
    plt.close()
    
    # 3. Temporal Repairs (Use phase2f data)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_p2f, x='shield_mode', y='temporal_shield_repair_count')
    plt.title('3. Shield Repairs (Instantaneous + Temporal)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plots', '3_temporal_repairs.png'))
    plt.close()

    # 4 & 5. Generations data
    gen_data = []
    experiments = df_coev['Experiment'].unique()
    for exp in experiments:
        for seed in [1, 2, 3]:
            exp_dir = os.path.join(archive_dir, f"exp_{exp}_seed_{seed}")
            gen_metrics_path = os.path.join(exp_dir, 'generation_metrics.csv')
            if os.path.exists(gen_metrics_path):
                df = pd.read_csv(gen_metrics_path)
                df['Experiment'] = exp
                df['Seed'] = seed
                gen_data.append(df)
                
    if gen_data:
        df_all = pd.concat(gen_data, ignore_index=True)
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_all, x='generation', y='attack_leakage', hue='Experiment')
        plt.title('4. Adaptive Attacker: Leakage over Generations')
        plt.savefig(os.path.join(results_dir, 'plots', '4_adaptive_leakage.png'))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_all, x='generation', y='defender_reward', hue='Experiment')
        plt.title('5. Co-evolution: Defender Reward')
        plt.savefig(os.path.join(results_dir, 'plots', '5_defender_reward.png'))
        plt.close()
        
    # 6. HoF Ablation: NoHoF vs Static HoF vs Adaptive HoF
    # Load adaptive HoF if exists
    adaptive_dir = os.path.join(results_dir, 'adaptive_hof')
    if os.path.exists(os.path.join(adaptive_dir, 'final_summary.csv')):
        df_adapt = pd.read_csv(os.path.join(adaptive_dir, 'final_summary.csv'))
        df_ablation = pd.concat([df_coev[df_coev['Experiment'].isin(['Adaptive_Shield_NoHoF', 'Adaptive_Shield_HoF_pareto_0.1'])], df_adapt])
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_ablation, x='Experiment', y='Robustness_Score')
        plt.title('6. HoF Ablation: Robustness')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plots', '6_hof_ablation.png'))
        plt.close()

    # 7. Held-out Attacker Leakage
    held_path = os.path.join(archive_dir, 'heldout_attacker_evaluation.csv')
    if os.path.exists(held_path):
        df_held = pd.read_csv(held_path)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_held, x='Experiment', y='Heldout_Leakage')
        plt.title('7. Held-out Attacker Leakage')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plots', '7_heldout_leakage.png'))
        plt.close()
        
    # 8. Safety-Performance Trade-off
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_combined, x='sla_violation_count', y='Final_Leakage_Mean', hue='Experiment', s=100)
    plt.title('8. Safety-Performance Trade-off')
    plt.xlabel('SLA Violations')
    plt.ylabel('Attack Leakage')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plots', '8_safety_performance_tradeoff.png'))
    plt.close()
    
    print(f"Generated Phase 3e final package plots in {results_dir}/plots")

if __name__ == "__main__":
    out_dir = "results/final_experiment_package"
    plot_final_metrics(out_dir)
