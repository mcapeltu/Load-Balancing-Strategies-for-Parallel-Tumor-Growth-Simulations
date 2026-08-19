#!/usr/bin/env python3

from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Cambia el nombre si guardas la imagen con otro nombre
IMAGE_FILE = SCRIPT_DIR / "day075.png"

OUTPUT_DIR = SCRIPT_DIR / "poster_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

N_BLOCKS = 16

# ============================================================
# Load tumour image
# ============================================================

img = Image.open(IMAGE_FILE).convert("RGB")

width, height = img.size

print(f"Image size: {width} x {height}")

# ============================================================
# Draw CRM partition
# ============================================================

fig, ax = plt.subplots(figsize=(6.0, 6.0))

ax.imshow(
    img,
    extent=[0, width, height, 0],
    interpolation="nearest"
)

# Height of each CRM region
region_height = height / N_BLOCKS

# ------------------------------------------------------------
# Horizontal boundaries
# ------------------------------------------------------------

for i in range(N_BLOCKS + 1):

    y = i * region_height

    # Outer borders slightly thicker
    linewidth = 2.0 if i in (0, N_BLOCKS) else 1.0

    ax.axhline(
        y,
        linewidth=linewidth,
        color="black",
        alpha=0.75
    )

# ------------------------------------------------------------
# Left/right external borders
# ------------------------------------------------------------

ax.axvline(
    0,
    linewidth=2.0,
    color="black",
    alpha=0.75
)

ax.axvline(
    width,
    linewidth=2.0,
    color="black",
    alpha=0.75
)

# ------------------------------------------------------------
# Block labels
# ------------------------------------------------------------

for block in range(N_BLOCKS):

    y_center = (block + 0.5) * region_height

    ax.text(
        width * 0.025,
        y_center,
        f"B{block}",
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.15",
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

# No conventional axes: this is an architectural diagram
ax.set_xticks([])
ax.set_yticks([])

ax.set_title(
    "CRM — Contiguous Row Mapping",
    fontsize=16,
    fontweight="bold",
    pad=12
)

# Caption inside figure
ax.text(
    width / 2,
    height + height * 0.055,
    "16 contiguous horizontal regions",
    ha="center",
    va="top",
    fontsize=12
)

fig.tight_layout()

# ============================================================
# Save
# ============================================================

fig.savefig(
    OUTPUT_DIR / "crm_day075_16_regions.pdf",
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "crm_day075_16_regions.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)

print("Generated:")
print(OUTPUT_DIR / "crm_day075_16_regions.pdf")
print(OUTPUT_DIR / "crm_day075_16_regions.png")
