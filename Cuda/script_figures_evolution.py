#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Directory containing this Python script: .../Cuda/
SCRIPT_DIR = Path(__file__).resolve().parent

# RTX 3090 benchmark directory
DATA_DIR = SCRIPT_DIR / "benchmarks" / "RTX_3090"

TUMOR_FILE = DATA_DIR / "crm_tumor_evolution.csv"
KERNEL_FILE = DATA_DIR / "evolution_vs_kernel.csv"

# ------------------------------------------------------------
# General plotting parameters
# ------------------------------------------------------------
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 600
})

# ============================================================
# 1. Tumour cells vs days
# ============================================================

tumor = pd.read_csv(TUMOR_FILE)

# We want one tumour-size value per simulated day.
# Each day has 24 steps, so take the final step of each day.
tumor_daily = (
    tumor.sort_values(["day", "step"])
         .groupby("day", as_index=False)
         .tail(1)
)

fig, ax = plt.subplots(figsize=(8.5, 5.2))

ax.plot(
    tumor_daily["day"],
    tumor_daily["tumor_cells"],
    linewidth=2.8
)

# Mark the three days used in the tumour montage
selected_days = [20, 75, 150]

for day in selected_days:
    row = tumor_daily[tumor_daily["day"] == day]

    if not row.empty:
        cells = row.iloc[0]["tumor_cells"]

        ax.scatter(
            day,
            cells,
            s=70,
            zorder=5
        )

        ax.annotate(
            f"Day {day}\n{int(cells):,} cells",
            xy=(day, cells),
            xytext=(0, 13),
            textcoords="offset points",
            ha="center",
            fontsize=11
        )

ax.set_xlabel("Simulation day")
ax.set_ylabel("Tumour cells")
ax.set_xlim(1, 150)
ax.set_ylim(bottom=0)

ax.grid(
    True,
    linestyle="--",
    linewidth=0.6,
    alpha=0.35
)

fig.tight_layout()

fig.savefig(
    "tumour_cells_vs_days.pdf",
    bbox_inches="tight"
)

fig.savefig(
    "tumour_cells_vs_days.png",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 2. Transition kernel execution time vs iterations
# ============================================================

kernel = pd.read_csv(KERNEL_FILE)

fig, ax = plt.subplots(figsize=(8.5, 5.2))

ax.plot(
    kernel["iteration"],
    kernel["crm_ms"],
    linewidth=2.0,
    label="CRM"
)

ax.plot(
    kernel["iteration"],
    kernel["frm_ms"],
    linewidth=2.0,
    label="FRM"
)

# Correspondence:
# day 20  -> iteration 480
# day 75  -> iteration 1800
# day 150 -> iteration 3600

selected_iterations = {
    480: "Day 20",
    1800: "Day 75",
    3600: "Day 150"
}

for iteration, day_label in selected_iterations.items():
    ax.axvline(
        iteration,
        linestyle="--",
        linewidth=1.0,
        alpha=0.5
    )

    ax.text(
        iteration,
        ax.get_ylim()[1] * 0.93,
        day_label,
        rotation=90,
        va="top",
        ha="right",
        fontsize=10
    )

ax.set_xlabel("Simulation iteration")
ax.set_ylabel("Transition-kernel time (ms)")
ax.set_xlim(1, 3600)
ax.set_ylim(bottom=0)

ax.legend(
    frameon=False,
    loc="upper left"
)

ax.grid(
    True,
    linestyle="--",
    linewidth=0.6,
    alpha=0.35
)

fig.tight_layout()

fig.savefig(
    "transition_kernel_vs_iterations.pdf",
    bbox_inches="tight"
)

fig.savefig(
    "transition_kernel_vs_iterations.png",
    bbox_inches="tight"
)

plt.close(fig)

print("Generated:")
print("  tumour_cells_vs_days.pdf")
print("  tumour_cells_vs_days.png")
print("  transition_kernel_vs_iterations.pdf")
print("  transition_kernel_vs_iterations.png")
