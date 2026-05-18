import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd
import seaborn as sns

def plot_metrics(results_dir):
    summary_path = os.path.join(results_dir, 'final_summary.csv')
    if not os.path.exists(summary_path):
        print("No summary data to plot.")
        return
        
    df_sum = pd.read_csv(summary_path)
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    
    experiments = df_sum['Experiment'].unique()
    gen_data = []
    archive_data = []
    
    for exp in experiments:
        for seed in [1, 2, 3, 4, 5]:
            exp_dir = os.path.join(results_dir, f"exp_{exp}_seed_{seed}")
            gen_metrics_path = os.path.join(exp_dir, 'generation_metrics.csv')
            if os.path.exists(gen_metrics_path):
                df = pd.read_csv(gen_metrics_path)
                df['Experiment'] = exp
                df['Seed'] = seed
                gen_data.append(df)
            
            hof_meta_path = os.path.join(exp_dir, 'hall_of_fame_metadata.csv')
            if os.path.exists(hof_meta_path):
                df_hof = pd.read_csv(hof_meta_path)
                df_hof['Experiment'] = exp
                df_hof['Seed'] = seed
                archive_data.append(df_hof)
                
    if gen_data:
        df_all = pd.concat(gen_data, ignore_index=True)
        
        # 1. Defender Service Quality
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_all, x='generation', y='service_quality', hue='Experiment')
        plt.title('1. Defender Service Quality over Generations')
        plt.savefig(os.path.join(results_dir, 'plots', '1_service_quality.png'))
        plt.close()
        
        # 2. Attack Leakage
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_all, x='generation', y='attack_leakage', hue='Experiment')
        plt.title('2. Attack Leakage over Generations')
        plt.savefig(os.path.join(results_dir, 'plots', '2_attack_leakage.png'))
        plt.close()
        
        # 3. Attacker Reward
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_all, x='generation', y='attacker_reward', hue='Experiment')
        plt.title('3. Attacker Reward over Generations')
        plt.savefig(os.path.join(results_dir, 'plots', '3_attacker_reward.png'))
        plt.close()
        
        # 4. Defender Reward
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_all, x='generation', y='defender_reward', hue='Experiment')
        plt.title('4. Defender Reward over Generations')
        plt.savefig(os.path.join(results_dir, 'plots', '4_defender_reward.png'))
        plt.close()
        
        # 9. Shield repairs
        if 'shield_repairs' in df_all.columns:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df_all, x='generation', y='shield_repairs', hue='Experiment')
            plt.title('9. Shield Repairs over Generations')
            plt.savefig(os.path.join(results_dir, 'plots', '9_shield_repairs.png'))
            plt.close()
            
    # 5. Hall-of-Fame Archive Growth
    if archive_data:
        df_arc = pd.concat(archive_data, ignore_index=True)
        df_arc['count'] = 1
        growth = df_arc.groupby(['Experiment', 'Seed', 'generation']).sum()['count'].groupby(level=[0,1]).cumsum().reset_index(name='archive_size')
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=growth, x='generation', y='archive_size', hue='Experiment')
        plt.title('5. Hall-of-Fame Archive Growth')
        plt.savefig(os.path.join(results_dir, 'plots', '5_archive_growth.png'))
        plt.close()
        
    rob_path = os.path.join(results_dir, 'robustness_analysis.csv')
    if os.path.exists(rob_path):
        df_rob = pd.read_csv(rob_path)
        # 7. Robustness score comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_rob, x='Experiment', y='Robustness_Score')
        plt.title('7. Robustness Score Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plots', '7_robustness_score.png'))
        plt.close()
        
        # 12. HoF vs NoHoF Robustness
        hof_vs = df_rob[df_rob['Experiment'].isin(['Adaptive_Shield_HoF_pareto_0.1', 'Adaptive_Shield_NoHoF'])]
        if not hof_vs.empty:
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=hof_vs, x='Experiment', y='Robustness_Score')
            plt.title('12. HoF vs NoHoF Robustness')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'plots', '12_hof_vs_nohof_robustness.png'))
            plt.close()

    forg_path = os.path.join(results_dir, 'forgetting_analysis.csv')
    if os.path.exists(forg_path):
        df_forg = pd.read_csv(forg_path)
        # 6. Forgetting score over generations (currently computing final score only in the script)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_forg, x='Experiment', y='Forgetting_Score')
        plt.title('6. Final Forgetting Score (Negative = Improved)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plots', '6_forgetting_score.png'))
        plt.close()
        
    held_path = os.path.join(results_dir, 'heldout_attacker_evaluation.csv')
    if os.path.exists(held_path):
        df_held = pd.read_csv(held_path)
        # 11. Held-out attacker leakage comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_held, x='Experiment', y='Heldout_Leakage')
        plt.title('11. Held-out Attacker Leakage')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plots', '11_heldout_leakage.png'))
        plt.close()
        
    print(f"Generated Phase 3d plots in {results_dir}/plots")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    args = parser.parse_args()
    plot_metrics(args.results_dir)
