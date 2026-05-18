import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def plot_bar_chart(df, metric_col, title, ylabel, filename, results_dir, color='tab:blue'):
    plt.figure(figsize=(10, 6))
    x = np.arange(len(df))
    plt.bar(x, df[metric_col], color=color)
    plt.xticks(x, df["Policy"], rotation=20, ha='right')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, filename), dpi=300)
    plt.close()

def plot_results(results_dir):
    summary_path = os.path.join(results_dir, "final_summary.csv")
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found.")
        return
        
    df = pd.read_csv(summary_path)
    
    # 1. Main Comparison (SQ vs SLA Violations)
    plt.figure(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Policy')
    ax1.set_ylabel('Service Quality (SQ)', color=color)
    ax1.bar(x - width/2, df["Avg_SQ"], width, color=color, label='Service Quality')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.1)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('SLA Violations', color=color)
    ax2.bar(x + width/2, df["SLA_Violations"], width, color=color, label='SLA Violations')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.xticks(x, df["Policy"], rotation=20, ha='right')
    plt.title("Phase 2B Benchmark: SQ vs SLA Violations")
    fig.tight_layout()
    plt.savefig(os.path.join(results_dir, "phase2_main_comparison_plot.png"), dpi=300)
    plt.close()
    
    # 2. Service Quality Plot
    plot_bar_chart(df, "Avg_SQ", "Average Service Quality by Policy", "Service Quality", "service_quality_plot.png", results_dir, 'mediumseagreen')
    
    # 3. Mitigation Efficiency Plot
    plot_bar_chart(df, "Avg_ME", "Mitigation Efficiency by Policy", "Mitigation Efficiency", "mitigation_efficiency_plot.png", results_dir, 'steelblue')
    
    # 4. Safety Violations Plot
    plot_bar_chart(df, "Safety_Violations", "Safety Violations by Policy", "Total Violations", "safety_violations_plot.png", results_dir, 'crimson')
    
    # 5. Latency Plot
    plot_bar_chart(df, "Avg_Total_Control_Latency_ms", "Total Control Latency (ms)", "Latency (ms)", "latency_plot.png", results_dir, 'darkorange')

    print("Successfully generated all Phase 2B empirical plots.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()
    plot_results(args.results_dir)
