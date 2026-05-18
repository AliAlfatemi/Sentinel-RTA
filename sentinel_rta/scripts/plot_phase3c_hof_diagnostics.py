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
    
    # Extract Prob and Mode
    def extract_prob(exp):
        if 'NoHoF' in exp: return 0.0
        try: return float(exp.split('prob')[1].split('_')[0])
        except: return 0.0
        
    def extract_mode(exp):
        if 'NoHoF' in exp: return 'none'
        try: return exp.split('_', 2)[2]
        except: return 'none'

    df_sum['Prob'] = df_sum['Experiment'].apply(extract_prob)
    df_sum['Mode'] = df_sum['Experiment'].apply(extract_mode)
    
    # 1. Robustness score by HoF variant.
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_sum, x='Mode', y='Robustness_Score', hue='Prob')
    plt.title('1. Robustness Score by HoF Variant')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plots', '1_robustness_by_variant.png'))
    plt.close()

    # 2. Forgetting score by HoF variant.
    if 'Forgetting_Score' in df_sum.columns:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_sum, x='Mode', y='Forgetting_Score', hue='Prob')
        plt.title('2. Forgetting Score (Worse against Gen0) by HoF Variant')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plots', '2_forgetting_by_variant.png'))
        plt.close()

    # 8. Shield repairs vs SLA violations.
    # We plot from the tradeoff diagnostics
    tradeoff_path = os.path.join(results_dir, 'shield_tradeoff_diagnostics.csv')
    if os.path.exists(tradeoff_path):
        df_tradeoff = pd.read_csv(tradeoff_path)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_tradeoff, x='rolling_sla_violations', y='temporal_repair_count', hue='Experiment')
        plt.title('8. Shield Repairs vs SLA Violations')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plots', '8_repairs_vs_sla.png'))
        plt.close()
        
    print(f"Generated Phase 3c plots in {results_dir}/plots")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    args = parser.parse_args()
    plot_metrics(args.results_dir)
