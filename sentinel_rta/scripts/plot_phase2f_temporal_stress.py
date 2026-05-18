import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os

def plot_phase2f_stress(results_dir):
    summary_path = os.path.join(results_dir, "temporal_stress_summary.csv")
    per_step_path = os.path.join(results_dir, "per_step_logs.csv")
    sweep_path = os.path.join(results_dir, "temporal_parameter_sweep.csv")
    
    if not os.path.exists(summary_path):
        print("Summary not found.")
        return
        
    df = pd.read_csv(summary_path)
    
    # 1. Temporal repairs by policy and scenario
    plt.figure(figsize=(14, 6))
    temp_df = df[df["shield_mode"] == "Temporal Runtime Shield"]
    if not temp_df.empty:
        sns.barplot(data=temp_df, x="scenario", y="temporal_shield_repair_count", hue="policy_name")
        plt.title("Temporal Repairs by Scenario and Policy")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "temporal_repairs_by_scenario.png"), dpi=300)
    plt.close()
    
    # 2. SLA Violations (No Shield vs Inst vs Temp)
    plt.figure(figsize=(14, 6))
    # We'll plot total SLA violations aggregated over scenarios
    sla_agg = df.groupby(["policy_name", "shield_mode"])["cumulative_sla_violation_count"].mean().reset_index()
    sns.barplot(data=sla_agg, x="policy_name", y="cumulative_sla_violation_count", hue="shield_mode")
    plt.title("Cumulative SLA Violations across Shield Modes")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "sla_violations_by_shield_mode.png"), dpi=300)
    plt.close()
    
    # Time-series plots from per_step_logs
    if os.path.exists(per_step_path):
        df_step = pd.read_csv(per_step_path)
        
        # We'll isolate the Edge-Riding Policy under long_sustained_attack for a single seed to show the dynamics
        sub_df = df_step[(df_step["policy_name"] == "Edge-Riding Stress Policy") & 
                         (df_step["scenario"] == "long_sustained_attack") & 
                         (df_step["seed"] == df_step["seed"].iloc[0])]
                         
        if not sub_df.empty:
            for mode in ["Instantaneous Runtime Shield", "Temporal Runtime Shield"]:
                mode_df = sub_df[sub_df["shield_mode"] == mode]
                if mode_df.empty: continue
                
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
                
                # Plot 3: Rolling service quality
                ax1.plot(mode_df["step"], mode_df["rolling_service_quality"], color='green', label='Rolling SQ')
                ax1.axhline(0.95, color='red', linestyle='--', label='SLA Bound')
                ax1.set_title(f"Rolling SQ ({mode})")
                ax1.legend()
                
                # Plot 4: Action bounds
                ax2.plot(mode_df["step"], mode_df["raw_action"], color='grey', alpha=0.5, label='Raw Action')
                ax2.plot(mode_df["step"], mode_df["final_action"], color='blue', label='Final Action')
                ax2.plot(mode_df["step"], mode_df["dynamic_max_action"], color='orange', linestyle='--', label='Dynamic Max Action')
                ax2.set_title("Action vs Dynamic Bounds")
                ax2.legend()
                
                # Plot SLA cumulative
                ax3.plot(mode_df["step"], mode_df["cumulative_sla_violation"], color='purple', label='Cumulative SLA Violations')
                ax3.set_title("SLA Violations Over Time")
                ax3.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f"timeseries_edgeriding_{mode.replace(' ', '_')}.png"), dpi=300)
                plt.close()
                
    # Parameter Sweep Heatmap
    if os.path.exists(sweep_path):
        df_sweep = pd.read_csv(sweep_path)
        if not df_sweep.empty:
            # Aggregate by window and threshold to make it 2D
            agg_sweep = df_sweep.groupby(["rolling_window", "rolling_sla_threshold"])["cumulative_sla_violations"].mean().unstack()
            plt.figure(figsize=(8, 6))
            sns.heatmap(agg_sweep, annot=True, cmap="YlOrRd")
            plt.title("Parameter Sweep: Mean Cumulative SLA Violations")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "parameter_sweep_heatmap.png"), dpi=300)
            plt.close()
            
    print("Successfully generated Phase 2F plots.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()
    plot_phase2f_stress(args.results_dir)
