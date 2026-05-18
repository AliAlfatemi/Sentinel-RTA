import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# IEEE configurations
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linestyle": '--'
})

DOUBLE_COLUMN = (7.16, 3.2)

def clean_name(x):
    if not isinstance(x, str): return x
    x_clean = x.replace('_', ' ')
    mapping = {
        'No Shield': 'No Shield',
        'Instantaneous Runtime Shield': 'Instantaneous RTA',
        'Temporal Runtime Shield': 'Temporal RTA',
        'Adaptive Shield NoHoF': 'Adaptive Shield NoHoF',
        'Adaptive Shield HoF pareto 0.1': 'Adaptive Shield HoF',
        'Adaptive NoShield': 'Adaptive No Shield',
        'Static NoShield': 'Static No Shield'
    }
    return mapping.get(x_clean, x_clean)

style_logic = {
    'No Shield': {'color': '#6B7280', 'marker': 'o', 'label': 'No Shield'},
    'Instantaneous RTA': {'color': '#2563EB', 'marker': 's', 'label': 'Inst. RTA'},
    'Temporal RTA': {'color': '#16A34A', 'marker': 'D', 'label': 'Temporal RTA'},
    'Adaptive No Shield': {'color': '#6B7280', 'marker': 'o', 'label': 'Adaptive No Shield'},
    'Static No Shield': {'color': '#6B7280', 'marker': '^', 'label': 'Static No Shield'},
    'Adaptive Shield HoF': {'color': '#D97706', 'marker': 'X', 'label': 'Adaptive Shield HoF'},
    'Adaptive Shield NoHoF': {'color': '#16A34A', 'marker': 'D', 'label': 'Adaptive Shield NoHoF'}
}

# We load both Phase 2F and Phase 3D
df_p2f = pd.read_csv('results/manuscript_results_package/source_csv/shield_mode_comparison.csv')
df_coev = pd.read_csv('results/manuscript_results_package/source_csv/final_summary.csv')

df_p2f['Shield_Mode'] = df_p2f['shield_mode'].apply(clean_name)
df_coev['Experiment'] = df_coev['Experiment'].apply(clean_name)

# Calculate SLA violation rate dynamically
if 'total_eval_steps' in df_p2f.columns:
    df_p2f['sla_rate'] = df_p2f['cumulative_sla_violation_count'] / df_p2f['total_eval_steps'].clip(lower=1)
else:
    df_p2f['sla_rate'] = df_p2f['cumulative_sla_violation_count'] / 500.0

if 'total_eval_steps' in df_coev.columns:
    df_coev['sla_rate'] = df_coev['sla_violation_count'] / df_coev['total_eval_steps'].clip(lower=1)
elif 'evaluation_episodes' in df_coev.columns and 'max_steps' in df_coev.columns:
    total = df_coev['evaluation_episodes'] * df_coev['max_steps']
    df_coev['sla_rate'] = df_coev['sla_violation_count'] / total.clip(lower=1)
else:
    df_coev['sla_rate'] = df_coev['sla_violation_count'] / 2500.0

fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COLUMN)

# Add preferred region shading (no text)
for ax in axes:
    ax.axvspan(-0.01, 0.002, ymin=-0.01, ymax=0.3, color='#DCFCE7', alpha=0.08, zorder=0)

# Panel A: Phase 2F
for exp in ['No Shield', 'Instantaneous RTA', 'Temporal RTA']:
    df_sub = df_p2f[df_p2f['Shield_Mode'] == exp]
    if df_sub.empty: continue
    
    style = style_logic.get(exp, {'color': 'black', 'marker': 'o', 'label': exp})
    
    x_val = df_sub['sla_rate'].values
    y_val = df_sub['attack_leakage'].values
    repairs = df_sub['temporal_shield_repair_count'].values
    
    axes[0].scatter(x_val, y_val, 
                    s=55, alpha=0.9, color=style['color'], marker=style['marker'], zorder=3)
                    
    for i in range(len(x_val)):
        xytext = (6, 6)
        ha = 'left'
        va = 'bottom'
        
        if exp == 'Temporal RTA':
            xytext = (6, 0)
            ha = 'left'
            va = 'center'
        elif exp == 'Instantaneous RTA':
            xytext = (0, 6)
            ha = 'center'
            va = 'bottom'
        elif exp == 'No Shield':
            xytext = (6, -6)
            ha = 'left'
            va = 'top'

        axes[0].annotate(style['label'], (x_val[i], y_val[i]), 
                         xytext=xytext, textcoords='offset points', fontsize=7, color='#111827', 
                         ha=ha, va=va, zorder=4)
                         
        r_val = repairs[i]
        if not pd.isna(r_val):
            r_int = int(float(r_val))
            if r_int > 0:
                axes[0].annotate(f"{r_int} repairs", (x_val[i], y_val[i]), 
                                 xytext=(xytext[0], xytext[1]-9), 
                                 textcoords='offset points', fontsize=6.5, color=style['color'], 
                                 ha=ha, va=va, zorder=4)

axes[0].set_title('A. Temporal stress setting', loc='left')
axes[0].set_xlabel('SLA violation rate (lower is better)')
axes[0].set_ylabel('Attack leakage (lower is better)')

# Panel B: Phase 3D
panel_b_mapping = {
    'Adaptive No Shield': '1',
    'Static No Shield': '2',
    'Adaptive Shield HoF': '3',
    'Adaptive Shield NoHoF': '4'
}

for exp in ['Static No Shield', 'Adaptive No Shield', 'Adaptive Shield NoHoF', 'Adaptive Shield HoF']:
    df_sub = df_coev[df_coev['Experiment'] == exp]
    if df_sub.empty: continue
    
    style = style_logic.get(exp, {'color': 'black', 'marker': 'o', 'label': exp})
    num_label = panel_b_mapping.get(exp, '')
    
    x_val = df_sub['sla_rate'].values
    y_val = df_sub['Final_Leakage_Mean'].values
    
    repairs = df_sub['Final_Shield_Repairs'].values if 'Final_Shield_Repairs' in df_sub.columns else [0]*len(x_val)
    
    axes[1].scatter(x_val, y_val, 
                    s=55, alpha=0.9, color=style['color'], marker=style['marker'], zorder=3)
                    
    for i in range(len(x_val)):
        xytext = (7, 7)
        ha = 'left'
        va = 'bottom'
        
        if exp == 'Adaptive No Shield':
            xytext = (6, 6)
            ha = 'left'
            va = 'bottom'
        elif exp == 'Static No Shield':
            xytext = (6, -6)
            ha = 'left'
            va = 'top'
        elif exp == 'Adaptive Shield NoHoF':
            xytext = (-6, -6)
            ha = 'right'
            va = 'top'
        elif exp == 'Adaptive Shield HoF':
            xytext = (7, -3)
            ha = 'left'
            va = 'center'

        # Annotate Number
        axes[1].annotate(num_label, (x_val[i], y_val[i]), 
                         xytext=xytext, textcoords='offset points', fontsize=8, weight='bold', color='black', 
                         ha=ha, va=va, zorder=4)
                         
        # Annotate Repairs (only if > 0)
        r_val = repairs[i]
        if not pd.isna(r_val):
            r_int = int(float(r_val))
            if r_int > 0:
                axes[1].annotate(f"{r_int} repair" if r_int == 1 else f"{r_int} repairs", 
                                 (x_val[i], y_val[i]), 
                                 xytext=(xytext[0], xytext[1]-12 if va in ['top','center'] else xytext[1]-12), 
                                 textcoords='offset points', fontsize=6.5, color=style['color'], 
                                 ha=ha, va=va, zorder=4)

# Panel B compact key
key_text = "1 Adaptive No Shield\n2 Static No Shield\n3 Shield HoF\n4 Shield NoHoF"
axes[1].text(0.95, 0.95, key_text, transform=axes[1].transAxes, fontsize=6.5,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray', lw=0.5))

axes[1].set_title('B. Adaptive co-evolution setting', loc='left')
axes[1].set_xlabel('SLA violation rate (lower is better)')
axes[1].set_ylabel('Attack leakage (lower is better)')

# Adjust axes limits to make sure points are not clipped and there is room for annotations
axes[0].margins(x=0.25, y=0.35) # Increased y margin for Panel A padding
axes[1].margins(x=0.25, y=0.25)

# Fix axis bottom limit for preferred region
for ax in axes:
    ax.set_ylim(bottom=-0.01)
    ax.set_xlim(left=-0.0002)

fig.tight_layout()
fig.savefig("results/manuscript_results_package/figures/fig7_safety_performance_tradeoff.pdf", format="pdf", bbox_inches="tight")
plt.close(fig)

print("Figure generated successfully.")
