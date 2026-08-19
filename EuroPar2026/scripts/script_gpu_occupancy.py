#!/usr/bin/env python3

from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "poster_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

CRM_OCC = 48.83
HWD_OCC = 97.80

ratio = HWD_OCC / CRM_OCC

# ============================================================
# Style
# ============================================================

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 600
})

# ============================================================
# Figure
# ============================================================

fig, ax = plt.subplots(figsize=(5.5, 5.2))

labels = ["CRM", "HWD"]
values = [CRM_OCC, HWD_OCC]

bars = ax.bar(
    labels,
    values,
    width=0.55
)

# ------------------------------------------------------------
# Percentages
# ------------------------------------------------------------

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 2.5,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=20,
        fontweight="bold"
    )

# ------------------------------------------------------------
# Main ≈2x message
# ------------------------------------------------------------

ax.text(
    0.5,
    0.42,
    f"≈ {ratio:.2f}×",
    transform=ax.transAxes,
    ha="center",
    va="center",
    fontsize=23,
    fontweight="bold",
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="0.6",
        alpha=0.95
    )
)

ax.text(
    0.5,
    0.32,
    "achieved occupancy",
    transform=ax.transAxes,
    ha="center",
    va="center",
    fontsize=13
)

# ============================================================
# Formatting
# ============================================================

ax.set_ylim(0, 110)
ax.set_ylabel("Achieved occupancy (%)")

ax.set_title(
    "GPU OCCUPANCY",
    fontweight="bold",
    pad=14
)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.5,
    alpha=0.25
)

# Clean poster-style frame
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

# ============================================================
# Save
# ============================================================

fig.savefig(
    OUTPUT_DIR / "gpu_occupancy_crm_vs_hwd.pdf",
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "gpu_occupancy_crm_vs_hwd.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)

print("Generated:")
print(OUTPUT_DIR / "gpu_occupancy_crm_vs_hwd.pdf")
print(OUTPUT_DIR / "gpu_occupancy_crm_vs_hwd.png")
