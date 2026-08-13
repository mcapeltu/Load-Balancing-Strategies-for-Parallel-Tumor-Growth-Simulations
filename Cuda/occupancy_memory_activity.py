#!/usr/bin/env python3

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Output directory
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "poster_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# Data from the paper
# ============================================================

crm_occupancy = 48.83
hwd_occupancy = 97.80

crm_warps = 15.63
hwd_warps = 31.30

crm_l1 = 435.89
hwd_l1 = 432.96

crm_l2 = 410.79
hwd_l2 = 407.15

crm_dram = 621.93
hwd_dram = 631.61

# Relative changes, HWD vs CRM
l1_change = 100.0 * (hwd_l1 - crm_l1) / crm_l1
l2_change = 100.0 * (hwd_l2 - crm_l2) / crm_l2
dram_change = 100.0 * (hwd_dram - crm_dram) / crm_dram

warps_ratio = hwd_warps / crm_warps

# ============================================================
# General style
# ============================================================

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 17,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.dpi": 150,
    "savefig.dpi": 600
})

# ============================================================
# Figure 1
# Achieved Occupancy + Active Warps per SM
# ============================================================

labels = ["CRM", "HWD"]
occupancy = [crm_occupancy, hwd_occupancy]
warps = [crm_warps, hwd_warps]

fig, ax1 = plt.subplots(figsize=(9.0, 5.5))

x = np.arange(len(labels))
bar_width = 0.55

bars = ax1.bar(
    x,
    occupancy,
    width=bar_width,
    alpha=0.88
)

ax1.set_ylabel("Achieved occupancy (%)")
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylim(0, 110)

ax1.grid(
    axis="y",
    linestyle="--",
    linewidth=0.6,
    alpha=0.25
)

# ------------------------------------------------------------
# Occupancy labels INSIDE the bars
# ------------------------------------------------------------

for bar, value in zip(bars, occupancy):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        value - 5.0,
        f"{value:.2f}%",
        ha="center",
        va="top",
        fontsize=17,
        fontweight="bold"
    )

# ------------------------------------------------------------
# Secondary axis: active warps / SM
# ------------------------------------------------------------

ax2 = ax1.twinx()

ax2.plot(
    x,
    warps,
    marker="o",
    markersize=9,
    linewidth=2.2,
    linestyle="--",
    label="Active warps / SM"
)

ax2.set_ylabel("Active warps / SM")
ax2.set_ylim(0, 38)

# ------------------------------------------------------------
# Warp-value labels ABOVE the markers
# ------------------------------------------------------------

for xi, value in zip(x, warps):
    ax2.annotate(
        f"{value:.2f}",
        xy=(xi, value),
        xytext=(0, 14),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold"
    )

# ------------------------------------------------------------
# 2x annotation
# Place it in the central free region, not over HWD bar
# ------------------------------------------------------------

ax1.text(
    0.50,
    0.58,
    f"≈ {warps_ratio:.2f}× active warps / SM",
    transform=ax1.transAxes,
    ha="center",
    va="center",
    fontsize=15,
    fontweight="bold",
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="0.7",
        alpha=0.90
    )
)

ax1.set_title(
    "HWD nearly doubles GPU occupancy",
    pad=12
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "occupancy_and_active_warps.pdf",
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "occupancy_and_active_warps.png",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# Figure 2
# Memory activity change: HWD relative to CRM
# ============================================================

subsystems = ["L1", "L2", "DRAM"]
changes = [l1_change, l2_change, dram_change]

fig, ax = plt.subplots(figsize=(8.5, 4.8))

y = np.arange(len(subsystems))

bars = ax.barh(
    y,
    changes,
    height=0.55,
    alpha=0.88
)

ax.axvline(
    0,
    linewidth=1.2
)

ax.set_yticks(y)
ax.set_yticklabels(subsystems)

ax.set_xlabel(
    "Change in active cycles: HWD vs CRM (%)",
    labelpad=10
)

ax.set_xlim(-2.2, 2.2)

# More vertical room above the bars
ax.set_ylim(-0.6, 2.85)

ax.grid(
    axis="x",
    linestyle="--",
    linewidth=0.6,
    alpha=0.25
)

# ------------------------------------------------------------
# Numeric labels
# ------------------------------------------------------------

for bar, value in zip(bars, changes):

    y_pos = bar.get_y() + bar.get_height() / 2

    if value >= 0:
        x_pos = value + 0.08
        ha = "left"
    else:
        x_pos = value - 0.08
        ha = "right"

    ax.text(
        x_pos,
        y_pos,
        f"{value:+.2f}%",
        va="center",
        ha=ha,
        fontsize=15,
        fontweight="bold"
    )

# ------------------------------------------------------------
# Main takeaway INSIDE the plot, above the bars
# ------------------------------------------------------------

ax.text(
    0.5,
    0.94,
    "All changes remain within ±1.6%",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=15,
    fontweight="bold",
    bbox=dict(
        boxstyle="round,pad=0.30",
        facecolor="white",
        edgecolor="0.7",
        alpha=0.90
    )
)

ax.set_title(
    "Memory-system activity remains nearly unchanged",
    pad=12
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "memory_activity_change.pdf",
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "memory_activity_change.png",
    bbox_inches="tight"
)

plt.close(fig)
# ============================================================
# Print numerical summary
# ============================================================

print("Generated figures:")
print("  occupancy_and_active_warps.pdf")
print("  occupancy_and_active_warps.png")
print("  memory_activity_change.pdf")
print("  memory_activity_change.png")
print()
print("Computed values:")
print(f"Active warps ratio: {warps_ratio:.4f}x")
print(f"L1 change:   {l1_change:+.2f}%")
print(f"L2 change:   {l2_change:+.2f}%")
print(f"DRAM change: {dram_change:+.2f}%")
