import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

# ── Configuration ─────────────────────────────────────────────────────────────
LOG_DIR = 'logs'
PLOT_DIR = 'plots'
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Load data (Experiment 2: Mixed N=3,4,5,6) ──────────────────────────────────
# Note: These are the "Original" ablation logs without the "_new" suffix.
try:
    b0 = pd.read_csv(os.path.join(LOG_DIR, 'grpo_ablation_B0_G8_metrics.csv'))
    b1 = pd.read_csv(os.path.join(LOG_DIR, 'grpo_ablation_B1_G8_metrics.csv'))
    b2 = pd.read_csv(os.path.join(LOG_DIR, 'grpo_ablation_B2_G8_metrics.csv'))
    b3 = pd.read_csv(os.path.join(LOG_DIR, 'grpo_ablation_B3_G8_metrics.csv'))
    # B4 uses the FINAL version found in logs
    b4 = pd.read_csv(os.path.join(LOG_DIR, 'grpo_ablation_B4_FINAL_G8_metrics.csv'))
except Exception as e:
    print(f"Error loading CSV files: {e}")
    raise e

dfs = {
    'B0 Vanilla GRPO':       b0,
    'B1 +Length Penalty':    b1,
    'B2 +Reward Annealing':  b2,
    'B3 +Adv Clipping':      b3,
    'B4 Full LAGRPO (Ours)': b4,
}

# ── Palette (Matching v2 style) ────────────────────────────────────────────────
PALETTE = {
    'B0 Vanilla GRPO':       '#7f7f7f',
    'B1 +Length Penalty':    '#E07B39',
    'B2 +Reward Annealing':  '#5B8DB8',
    'B3 +Adv Clipping':      '#6BAF6B',
    'B4 Full LAGRPO (Ours)': '#C94040',
}
LW   = {k: 1.5 for k in dfs}; LW['B4 Full LAGRPO (Ours)'] = 2.4
LS   = {
    'B0 Vanilla GRPO':       '--',
    'B1 +Length Penalty':    ':',
    'B2 +Reward Annealing':  '-.',
    'B3 +Adv Clipping':      (0,(3,1,1,1)),
    'B4 Full LAGRPO (Ours)': '-',
}
ZO   = {k: 2 for k in dfs}; ZO['B4 Full LAGRPO (Ours)'] = 5

def ema_smooth(series, alpha=0.12):
    return series.ewm(alpha=alpha, adjust=False).mean()

# ── Figure 1: Main 1×3 (Mixed Difficulty) ─────────────────────────────────────
sns.set_theme(style='whitegrid', font='DejaVu Sans')
fig1, axes = plt.subplots(1, 3, figsize=(15, 4.4),
                          gridspec_kw={'wspace': 0.38})
fig1.patch.set_facecolor('white')

# (a) EMA Success Rate
ax = axes[0]
for name, df in dfs.items():
    steps = df['step'].values
    vals  = df['ema_success_rate'].values * 100
    ax.plot(steps, vals, color=PALETTE[name], lw=LW[name],
            ls=LS[name], zorder=ZO[name], alpha=0.93)

ax.set_xlabel('Update Step', fontsize=10)
ax.set_ylabel('EMA Success Rate (%)', fontsize=10)
ax.set_title('(a) EMA Success Rate (Mixed Difficulty)', fontsize=11, fontweight='bold', pad=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.1f}%'))
ax.tick_params(labelsize=9)

# Clean Legend (No SNR)
handles = [mpatches.Patch(color=PALETTE[n], label=n) for n in dfs]
ax.legend(handles=handles, fontsize=7.8, framealpha=0.9,
          loc='upper left', handlelength=2.2)

# (b) Output Length — Boxplot
ax = axes[1]
box_data  = [df['mean_response_length'].values for df in dfs.values()]
box_names = list(dfs.keys())

bp = ax.boxplot(box_data, patch_artist=True, notch=False, widths=0.52,
                medianprops=dict(color='black', lw=2.2),
                whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
                flierprops=dict(marker='o', markersize=3, alpha=0.4, lw=0))

for patch, name in zip(bp['boxes'], box_names):
    patch.set_facecolor(PALETTE[name]); patch.set_alpha(0.80)
for flier, name in zip(bp['fliers'], box_names):
    flier.set(markerfacecolor=PALETTE[name], markeredgecolor=PALETTE[name])

short = ['B0\nVanilla', 'B1\n+Len', 'B2\n+Anneal', 'B3\n+Clip', 'B4\nLAGRPO']
ax.set_xticks(range(1, 6)); ax.set_xticklabels(short, fontsize=8.5)
ax.set_ylabel('Mean Response Length (tokens)', fontsize=10)
ax.set_title('(b) Output Length Distribution', fontsize=11, fontweight='bold', pad=8)
ax.tick_params(axis='y', labelsize=9)

# (c) Exploration Efficiency η
ax = axes[2]
for name, df in dfs.items():
    steps = df['step'].values
    raw_eta = df['ema_success_rate'] / df['mean_response_length'] * 1000
    eta = ema_smooth(raw_eta, alpha=0.15).values
    ax.plot(steps, eta, color=PALETTE[name], lw=LW[name],
            ls=LS[name], zorder=ZO[name], alpha=0.93)

ax.set_xlabel('Update Step', fontsize=10)
ax.set_ylabel(r'$\eta = \frac{\mathrm{EMA\;SR}}{\mathrm{Length}}\times10^{3}$', fontsize=10)
ax.set_title(r'(c) Exploration Efficiency ($\eta$)', fontsize=11,
             fontweight='bold', pad=8)
ax.tick_params(labelsize=9)

# Clean up axes
for ax_ in axes:
    ax_.spines['top'].set_visible(False)
    ax_.spines['right'].set_visible(False)
    ax_.grid(True, linestyle='--', linewidth=0.45, alpha=0.55)

# Save plot
fig1.savefig(os.path.join(PLOT_DIR, 'ablation_mixed_main.png'),
             dpi=200, bbox_inches='tight', facecolor='white')
fig1.savefig(os.path.join(PLOT_DIR, 'ablation_mixed_main.pdf'),
             dpi=300, bbox_inches='tight', facecolor='white')

print("Mixed difficulty main figure saved to plots/ablation_mixed_main.png")

# ── Summary Table (Table 5-4) ─────────────────────────────────────────────────
print("\n=== Table 5-4: LAGRPO Ablation Study (Mixed Difficulty N=3,4,5,6) ===")
print(f"{'Config':<24} | {'Stable SR':>10} | {'Peak SR':>10} | {'Med Length':>10} | {'Halluc':>8} | {'KL':>8} | {'Eta':>8}")
print("-" * 100)

for name, df in dfs.items():
    s50 = df.tail(50)
    sr_st = s50['ema_success_rate'].mean()
    sr_mx = df['ema_success_rate'].max()
    len_med = df['mean_response_length'].median()
    len_st_med = s50['mean_response_length'].median()
    halluc = df['hallucination_rate'].mean()
    kl = df['kl_div'].mean()
    
    # Calculate eta based on stable metrics
    # Using stable mean success and stable median length for Eta
    eta = (sr_st / len_st_med) * 1000 if len_st_med > 0 else 0
    
    print(f"{name:<24} | {sr_st:>10.4f} | {sr_mx:>10.4f} | {len_med:>10.1f} | {halluc:>8.3f} | {kl:>8.2f} | {eta:>8.3f}")
