#!/usr/bin/env python3

from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

IMAGE_FILE = SCRIPT_DIR / "day075.png"

OUTPUT_DIR = SCRIPT_DIR / "poster_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

N_REGIONS = 64

# ============================================================
# Load tumour image
# ============================================================

img = Image.open(IMAGE_FILE).convert("RGB")

width, height = img.size

print(f"Image size: {width} x {height}")

# ============================================================
# Draw FRM partition
# ============================================================

fig, ax = plt.subplots(figsize=(6.0, 6.0))

ax.imshow(
    img,
    extent=[0, width, height, 0],
    interpolation="nearest"
)

region_height = height / N_REGIONS

# ------------------------------------------------------------
# Horizontal boundaries for 64 regions
# ------------------------------------------------------------

for i in range(N_REGIONS + 1):

    y = i * region_height

    # Outer border slightly thicker
    if i in (0, N_REGIONS):
        linewidth = 1.8
        alpha = 0.85
    else:
        linewidth = 0.45
        alpha = 0.55

    ax.axhline(
        y,
        linewidth=linewidth,
        color="black",
        alpha=alpha
    )

# ------------------------------------------------------------
# External vertical borders
# ------------------------------------------------------------

ax.axvline(
    0,
    linewidth=1.8,
    color="black",
    alpha=0.85
)

ax.axvline(
    width,
    linewidth=1.8,
    color="black",
    alpha=0.85
)

# ------------------------------------------------------------
# Label only selected regions
# Avoid cluttering the figure with 64 labels
# ------------------------------------------------------------

selected_regions = {
    0: "R0",
    15: "R15",
    31: "R31",
    32: "R32",
    47: "R47",
    63: "R63"
}

for region, label in selected_regions.items():

    y_center = (region + 0.5) * region_height

    ax.text(
        width * 0.025,
        y_center,
        label,
        ha="left",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.12",
            facecolor="white",
            edgecolor="none",
            alpha=0.72
        )
    )

# ============================================================
# Formatting
# ============================================================

ax.set_xlim(0, width)
ax.set_ylim(height, 0)

ax.set_aspect("equal")

ax.set_xticks([])
ax.set_yticks([])

ax.set_title(
    "FRM — Fine-Grained Region Mapping",
    fontsize=16,
    fontweight="bold",
    pad=12
)

ax.text(
    width / 2,
    height + height * 0.055,
    "64 horizontal regions",
    ha="center",
    va="top",
    fontsize=12
)

fig.tight_layout()

# ============================================================
# Save
# ============================================================

fig.savefig(
    OUTPUT_DIR / "frm_day075_64_regions.pdf",
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "frm_day075_64_regions.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)

print("Generated:")
print(OUTPUT_DIR / "frm_day075_64_regions.pdf")
print(OUTPUT_DIR / "frm_day075_64_regions.png")
