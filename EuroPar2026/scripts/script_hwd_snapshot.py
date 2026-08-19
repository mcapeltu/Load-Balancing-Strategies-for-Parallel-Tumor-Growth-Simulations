#!/usr/bin/env python3

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image

# ============================================================
# Configuration
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

IMAGE_FILE = SCRIPT_DIR / "day075.png"

OUTPUT_DIR = SCRIPT_DIR / "poster_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

N_REGIONS = 64
N_BLOCKS = 16

# ============================================================
# Load tumour image
# ============================================================

img = Image.open(IMAGE_FILE).convert("RGB")

width, height = img.size

print(f"Image size: {width} x {height}")

region_height = height / N_REGIONS

# ============================================================
# Figure
# ============================================================

fig, ax = plt.subplots(figsize=(6.4, 6.4))

ax.imshow(
    img,
    extent=[0, width, height, 0],
    interpolation="nearest"
)

# ============================================================
# HWD mapping
#
# 64 fine regions
# region r -> CUDA block (r mod 16)
#
# Therefore:
# B0  processes regions 0,16,32,48
# B1  processes regions 1,17,33,49
# ...
# B15 processes regions 15,31,47,63
# ============================================================

# Use a qualitative palette for the 16 CUDA blocks
cmap = plt.get_cmap("tab20")

block_colors = [
    cmap(i) for i in range(N_BLOCKS)
]

# ------------------------------------------------------------
# Draw lightly coloured region overlays
# ------------------------------------------------------------

for region in range(N_REGIONS):

    block = region % N_BLOCKS

    y = region * region_height

    rect = Rectangle(
        (0, y),
        width,
        region_height,
        facecolor=block_colors[block],
        edgecolor="none",
        alpha=0.12
    )

    ax.add_patch(rect)

# ------------------------------------------------------------
# Draw the 64 region boundaries
# ------------------------------------------------------------

for i in range(N_REGIONS + 1):

    y = i * region_height

    if i in (0, N_REGIONS):
        linewidth = 1.8
        alpha = 0.90
    elif i % 16 == 0:
        # Highlight groups of 16 regions
        linewidth = 1.0
        alpha = 0.65
    else:
        linewidth = 0.35
        alpha = 0.35

    ax.axhline(
        y,
        color="black",
        linewidth=linewidth,
        alpha=alpha
    )

# External vertical borders
ax.axvline(
    0,
    color="black",
    linewidth=1.8
)

ax.axvline(
    width,
    color="black",
    linewidth=1.8
)

# ============================================================
# Highlight representative CUDA blocks
#
# Each highlighted block owns four spatially separated regions.
# ============================================================

highlighted_blocks = [0, 5, 10, 15]

label_x = width * 1.07

for block in highlighted_blocks:

    regions = [
        block,
        block + 16,
        block + 32,
        block + 48
    ]

    color = block_colors[block]

    # Block label on the right
    label_y = (
        sum((r + 0.5) * region_height for r in regions)
        / len(regions)
    )

    ax.text(
        label_x,
        label_y,
        f"B{block}",
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=color
    )

    # Mark each region assigned to this block
    for region in regions:

        y_center = (region + 0.5) * region_height

        # Small marker on right edge
        ax.plot(
            width * 0.985,
            y_center,
            marker="o",
            markersize=4.5,
            color=color,
            zorder=5
        )

        # Arrow from region towards the block label
        arrow = FancyArrowPatch(
            (width * 0.985, y_center),
            (label_x - width * 0.015, label_y),
            arrowstyle="-",
            linewidth=0.7,
            color=color,
            alpha=0.65,
            connectionstyle="arc3,rad=0.05"
        )

        ax.add_patch(arrow)

# ============================================================
# Add a compact conceptual annotation
# ============================================================

ax.text(
    width * 0.50,
    -height * 0.055,
    "spatial dispersion  +  contiguous accesses within each region",
    ha="center",
    va="bottom",
    fontsize=11,
    fontweight="bold"
)

# ============================================================
# Formatting
# ============================================================

ax.set_xlim(
    -width * 0.01,
    width * 1.18
)

ax.set_ylim(
    height,
    -height * 0.12
)

ax.set_aspect("equal")

ax.set_xticks([])
ax.set_yticks([])

ax.set_title(
    "HWD — Hybrid Workload Dispersion",
    fontsize=16,
    fontweight="bold",
    pad=12
)

# Remove ordinary matplotlib frame
for spine in ax.spines.values():
    spine.set_visible(False)

# Bottom takeaway
ax.text(
    width * 0.50,
    height * 1.04,
    "Distributed hotspot work → more balanced SM utilisation",
    ha="center",
    va="top",
    fontsize=11.5,
    fontweight="bold"
)

fig.tight_layout()

# ============================================================
# Save
# ============================================================

fig.savefig(
    OUTPUT_DIR / "hwd_day075_schematic.pdf",
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "hwd_day075_schematic.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# Console description
# ============================================================

print()
print("HWD schematic mapping:")
print()

for block in range(N_BLOCKS):
    regions = [
        block,
        block + 16,
        block + 32,
        block + 48
    ]

    print(
        f"B{block:2d}: regions "
        + ", ".join(str(r) for r in regions)
    )

print()
print("Generated:")
print(OUTPUT_DIR / "hwd_day075_schematic.pdf")
print(OUTPUT_DIR / "hwd_day075_schematic.png")
