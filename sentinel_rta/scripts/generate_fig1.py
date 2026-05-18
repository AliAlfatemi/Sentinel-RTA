import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

out_dir = "results/manuscript_results_package/figures"
os.makedirs(out_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.axis('off')

def draw_box(ax, x, y, width, height, text, facecolor='#eeeeee', edgecolor='black'):
    box = patches.Rectangle((x, y), width, height, linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, horizontalalignment='center', verticalalignment='center', fontsize=9, zorder=3, weight='bold')
    return (x, y, width, height)

def draw_arrow(ax, x1, y1, x2, y2, text=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='black'), zorder=1)
    if text:
        # mid point
        mx = (x1 + x2)/2
        my = (y1 + y2)/2
        ax.text(mx, my+0.1, text, horizontalalignment='center', verticalalignment='bottom', fontsize=8)

# Positions
# Left side: Attacker
b_attacker = draw_box(ax, 0.5, 3.5, 2, 1, "Adaptive\nAttacker", facecolor='#f8cecc', edgecolor='#b85450')

# Top: Environment
b_env = draw_box(ax, 3.5, 3.5, 3, 1, "Network / Service\nEnvironment", facecolor='#e1d5e7', edgecolor='#9673a6')

# Right side: Metrics
b_metrics = draw_box(ax, 7.5, 3.5, 2, 1, "Metrics Feedback\n(SQ, Leakage, SLA)", facecolor='#fff2cc', edgecolor='#d6b656')

# Middle left: PPO
b_ppo = draw_box(ax, 0.5, 1.0, 2, 1, "PPO Defender\n(RL Agent)", facecolor='#dae8fc', edgecolor='#6c8ebf')

# Middle center: Shields
b_inst_shield = draw_box(ax, 3.5, 1.5, 3, 0.8, "Instantaneous\nRuntime Assurance", facecolor='#d5e8d4', edgecolor='#82b366')
b_temp_shield = draw_box(ax, 3.5, 0.5, 3, 0.8, "Temporal\nRuntime Assurance\n(Rolling SLA Context)", facecolor='#d5e8d4', edgecolor='#82b366')

# Optional HoF
b_hof = draw_box(ax, 0.5, 2.5, 2, 0.5, "Hall-of-Fame\n(Optional replay ablation)", facecolor='#eeeeee', edgecolor='gray')

# Arrows
# Env -> Metrics
draw_arrow(ax, 6.5, 4.0, 7.5, 4.0, "Telemetry")

# Metrics -> PPO
# Draw a path: right down left
ax.plot([8.5, 8.5, 1.5, 1.5], [3.5, 0.2, 0.2, 1.0], color='black', lw=1.5, zorder=1)
ax.annotate("", xy=(1.5, 1.0), xytext=(1.5, 0.9), arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))
ax.text(5.0, 0.3, "Rewards & Observations", horizontalalignment='center', verticalalignment='bottom', fontsize=8)

# Env -> Attacker
draw_arrow(ax, 3.5, 4.0, 2.5, 4.0, "State")

# Attacker -> Env
draw_arrow(ax, 1.5, 4.5, 3.5, 4.5, "Attack Traffic")

# PPO -> Inst Shield
draw_arrow(ax, 2.5, 1.5, 3.5, 1.9, "Proposed Action")

# Inst Shield -> Temp Shield
draw_arrow(ax, 5.0, 1.5, 5.0, 1.3, "Validated")

# Temp Shield -> Env
draw_arrow(ax, 6.5, 0.9, 7.0, 0.9)
ax.plot([7.0, 7.0, 5.0, 5.0], [0.9, 3.0, 3.0, 3.5], color='black', lw=1.5, zorder=1)
ax.annotate("", xy=(5.0, 3.5), xytext=(5.0, 3.4), arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))
ax.text(7.2, 2.0, "Final Mitigation Action", horizontalalignment='left', verticalalignment='center', fontsize=8, rotation=90)

ax.set_xlim(0, 10)
ax.set_ylim(0, 5)

fig.savefig(os.path.join(out_dir, "fig1_architecture.pdf"), format="pdf", bbox_inches="tight")
fig.savefig(os.path.join(out_dir, "fig1_architecture.png"), format="png", dpi=600, bbox_inches="tight")
plt.close(fig)
print("Figure 1 generated.")
