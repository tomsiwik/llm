"""Pierre Research Program - Comprehensive Visualization."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# --- Style ---
plt.style.use("seaborn-v0_8-whitegrid")
KILLED = "#e74c3c"
SUPPORTED = "#27ae60"
PROVISIONAL = "#f39c12"
CONCLUSIVE = "#2980b9"
SFT_LINE = "#7f8c8d"

fig, axes = plt.subplots(2, 2, figsize=(18, 14), gridspec_kw={"hspace": 0.35, "wspace": 0.3})
fig.suptitle(
    "Pierre Research Program: M2P Distillation Results",
    fontsize=18,
    fontweight="bold",
    y=0.98,
)

# ============================================================
# Plot 1 (top-left): M2P Quality Evolution
# ============================================================
ax1 = axes[0, 0]

exps = [
    "distill\ntoy",
    "domain\ncond.",
    "loss\nnorm",
    "comp.\nn=5",
    "tfidf\nrouting",
    "data\nscale",
    "macro\nd=512",
]
quality = [21.9, 47.3, 50.7, 93.3, 92.2, 97.6, 100.6]
# Status: toy=killed, domain_cond=supported (K856 fail but K855 pass),
# loss_norm=supported, comp_n5=supported (K852 fail but quality good),
# tfidf=supported, data_scale=supported, macro=supported
statuses = [KILLED, SUPPORTED, SUPPORTED, SUPPORTED, SUPPORTED, SUPPORTED, SUPPORTED]

x = np.arange(len(exps))
bars = ax1.bar(x, quality, color=statuses, edgecolor="white", linewidth=0.8, width=0.7)

# SFT ceiling
ax1.axhline(y=100, color=SFT_LINE, linestyle="--", linewidth=1.5, label="SFT ceiling (100%)")

# Value labels
for i, (xi, q) in enumerate(zip(x, quality)):
    ax1.text(xi, q + 2, f"{q:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Trend line
ax1.plot(x, quality, color="#2c3e50", linewidth=1.2, alpha=0.5, zorder=0)

ax1.set_xticks(x)
ax1.set_xticklabels(exps, fontsize=8)
ax1.set_ylabel("Median Quality (% of SFT)", fontsize=11)
ax1.set_title("M2P Quality: From 22% to 101% of SFT", fontsize=13, fontweight="bold")
ax1.set_ylim(0, 115)
ax1.legend(fontsize=9, loc="upper left")

# Legend patches
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor=KILLED, label="Killed"),
    Patch(facecolor=SUPPORTED, label="Supported"),
    Patch(facecolor=SFT_LINE, label="SFT ceiling", linestyle="--"),
]
ax1.legend(handles=legend_elements, fontsize=9, loc="upper left")

# ============================================================
# Plot 2 (top-right): What Was NOT the Bottleneck
# ============================================================
ax2 = axes[0, 1]

categories = ["Width ($d_{M2P}$)", "Depth (L)", "Steps (T)"]
labels_per = [
    ["64", "128", "256"],
    ["1", "2", "4"],
    ["500", "1000", "2000"],
]
values_per = [
    [95.1, 93.1, 95.4],
    [88.0, 91.9, 91.9],
    [89.4, 84.7, 83.0],
]

n_groups = 3
n_bars = 3
bar_width = 0.22
group_positions = np.arange(n_groups)

for i in range(n_bars):
    positions = group_positions + (i - 1) * bar_width
    vals = [values_per[g][i] for g in range(n_groups)]
    bars = ax2.bar(
        positions,
        vals,
        bar_width,
        color=KILLED,
        alpha=0.6 + 0.15 * i,
        edgecolor="white",
        linewidth=0.5,
    )
    for pos, val in zip(positions, vals):
        ax2.text(pos, val + 0.8, f"{val:.0f}%", ha="center", va="bottom", fontsize=7.5)

# Add labels below each group
for g in range(n_groups):
    for i in range(n_bars):
        pos = group_positions[g] + (i - 1) * bar_width
        ax2.text(
            pos, -4.5, labels_per[g][i], ha="center", va="top", fontsize=7.5, color="#555"
        )

ax2.set_xticks(group_positions)
ax2.set_xticklabels(categories, fontsize=10)
ax2.set_ylabel("Quality (% of SFT)", fontsize=11)
ax2.set_ylim(70, 105)
ax2.set_title("Architecture Sweeps: All Killed", fontsize=13, fontweight="bold", color=KILLED)
ax2.axhline(y=100, color=SFT_LINE, linestyle="--", linewidth=1, alpha=0.5)

# Annotation
ax2.annotate(
    "Flat or declining\n= not the bottleneck",
    xy=(2, 83),
    fontsize=9,
    fontstyle="italic",
    color="#666",
    ha="center",
)

# ============================================================
# Plot 3 (bottom-left): Cross-Domain Transfer Heatmap
# ============================================================
ax3 = axes[1, 0]

domains = ["arith.", "sort", "parity", "reverse", "repeat"]
n = len(domains)

# Build 5x5 matrix from Option A cross_quality data
# diagonal = 0 (self), off-diagonal from results
cross_data = {
    (0, 1): 60.55,
    (0, 2): -110.45,
    (0, 3): 63.38,
    (0, 4): 79.21,
    (1, 2): -226.74,
    (1, 3): 62.56,
    (1, 4): 76.44,
    (2, 3): 62.86,
    (2, 4): 72.12,
    (3, 4): 78.72,
}

matrix = np.full((n, n), np.nan)
for (i, j), v in cross_data.items():
    matrix[i, j] = v
# Fill lower triangle symmetrically for display (or leave as directional)
# Keep it as directional (source -> target)

# For display: mask diagonal and missing
masked = np.ma.masked_invalid(matrix)

# Custom colormap: red for negative, green for positive
from matplotlib.colors import TwoSlopeNorm

vmin = -230
vmax = 80
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

cmap = plt.cm.RdYlGn
im = ax3.imshow(masked, cmap=cmap, norm=norm, aspect="auto")

# Text annotations
for i in range(n):
    for j in range(n):
        if i == j:
            ax3.text(j, i, "--", ha="center", va="center", fontsize=9, color="#999")
        elif not np.isnan(matrix[i, j]):
            val = matrix[i, j]
            color = "white" if abs(val) > 100 else "black"
            ax3.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=8, color=color, fontweight="bold")
        else:
            ax3.text(j, i, "", ha="center", va="center", fontsize=8, color="#ccc")

ax3.set_xticks(range(n))
ax3.set_xticklabels(domains, fontsize=9)
ax3.set_yticks(range(n))
ax3.set_yticklabels(domains, fontsize=9)
ax3.set_xlabel("Target domain", fontsize=10)
ax3.set_ylabel("Source adapter", fontsize=10)
ax3.set_title("Cross-Domain Transfer (% improvement)", fontsize=13, fontweight="bold")

cb = fig.colorbar(im, ax=ax3, shrink=0.8, pad=0.02)
cb.set_label("Improvement %", fontsize=9)

# ============================================================
# Plot 4 (bottom-right): Scale Trajectory
# ============================================================
ax4 = axes[1, 1]

d_models = [256, 512]
qualities = [97.6, 100.6]

ax4.plot(d_models, qualities, "o-", color=SUPPORTED, markersize=10, linewidth=2.5, zorder=5, label="Measured")

# Dotted projection
proj_d = [512, 1024, 3584]
# Linear extrapolation from 256->512 trend
slope = (100.6 - 97.6) / (512 - 256)
proj_q = [100.6, 100.6 + slope * (1024 - 512), 100.6 + slope * (3584 - 512)]
# Cap projections reasonably (quality ratio can exceed 100%)
ax4.plot(proj_d, proj_q, "o--", color=PROVISIONAL, markersize=7, linewidth=1.5, alpha=0.7, label="Projected", zorder=4)

# 85% threshold
ax4.axhline(y=85, color=KILLED, linestyle=":", linewidth=1.5, alpha=0.7, label="Kill threshold (85%)")
# SFT ceiling
ax4.axhline(y=100, color=SFT_LINE, linestyle="--", linewidth=1, alpha=0.5, label="SFT ceiling")

# Labels
for d, q in zip(d_models, qualities):
    ax4.annotate(
        f"{q:.1f}%\nd={d}",
        (d, q),
        textcoords="offset points",
        xytext=(0, 15),
        ha="center",
        fontsize=9,
        fontweight="bold",
    )

# Projection labels
ax4.annotate(
    f"d=1024\n(proj.)",
    (1024, proj_q[1]),
    textcoords="offset points",
    xytext=(0, 15),
    ha="center",
    fontsize=8,
    color=PROVISIONAL,
    fontstyle="italic",
)
ax4.annotate(
    f"d=3584\nQwen3-4B",
    (3584, proj_q[2]),
    textcoords="offset points",
    xytext=(0, 15),
    ha="center",
    fontsize=8,
    color=PROVISIONAL,
    fontstyle="italic",
)

ax4.set_xlabel("$d_{model}$", fontsize=12)
ax4.set_ylabel("M2P Quality (% of SFT)", fontsize=11)
ax4.set_title("M2P Quality vs Model Scale", fontsize=13, fontweight="bold")
ax4.set_xscale("log", base=2)
ax4.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax4.set_xticks([256, 512, 1024, 2048, 3584])
ax4.set_xticklabels(["256", "512", "1024", "2048", "3584"])
ax4.set_ylim(80, 115)
ax4.legend(fontsize=9, loc="lower right")

# --- Final layout ---
plt.subplots_adjust(top=0.93)
out_path = "/Users/tom/Code/tomsiwik/llm/pierre/research_progress.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved to {out_path}")
plt.close()
