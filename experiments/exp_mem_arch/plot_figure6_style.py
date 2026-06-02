#!/usr/bin/env python3
"""Plot Figure 6 style: Attention block latency comparison between w/ and w/o reordering."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.sim_utils import attention_breakdown_from_csv, read_csv_rows

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
PLOT_DIR = EXP_DIR / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_PER_GPU = [32, 64, 128, 256]
SEQ_LENGTHS = [2048, 4096, 8192]


def _paper_positions():
    positions = []
    seq_labels = []
    group_centers = []
    inner_step = 0.68
    offsets = (-inner_step, 0.0, inner_step)
    group_gap = 2.55
    for group_idx, batch in enumerate(BATCH_PER_GPU):
        center = group_idx * group_gap
        batch_positions = [center + offset for offset in offsets]
        positions.extend(batch_positions)
        seq_labels.extend(str(seq_len) for seq_len in SEQ_LENGTHS)
        group_centers.append(center)
    return positions, seq_labels, group_centers


def _add_paper_xlabels(ax, positions, seq_labels, group_centers, group_labels):
    ax.set_xticks(positions)
    ax.set_xticklabels(seq_labels, rotation=90, fontsize=7)
    ax.tick_params(axis="x", length=0, pad=1)
    for center, label in zip(group_centers, group_labels):
        ax.text(center, -0.20, str(label), transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.5)
    ax.text(-0.17, -0.11, "Sequence\nlength", transform=ax.transAxes, ha="left", va="center", fontsize=8)
    ax.text(-0.17, -0.26, "Batch per\nGPU", transform=ax.transAxes, ha="left", va="center", fontsize=8)
    for idx in range(1, len(group_labels)):
        split_x = (group_centers[idx - 1] + group_centers[idx]) / 2
        ax.axvline(split_x, ymin=0.0, ymax=0.06, color="black", lw=0.8, clip_on=False)


# Load data (Ramulator ON)
data = {}
for reorder in ["on", "off"]:
    for batch in BATCH_PER_GPU:
        for seq_len in SEQ_LENGTHS:
            filename = f"result_hbm3e_b{batch}_l{seq_len}_reorder_{reorder}_ramul_on.csv"
            filepath = DATA_DIR / filename
            if filepath.exists():
                data[(reorder, seq_len, batch)] = attention_breakdown_from_csv(filepath)

# Plot
positions, seq_tick_labels, group_centers = _paper_positions()
seq_offsets = dict(zip(SEQ_LENGTHS, (-0.68, 0.0, 0.68)))
x_for = {
    (batch, seq_len): group_centers[group_idx] + seq_offsets[seq_len]
    for group_idx, batch in enumerate(BATCH_PER_GPU)
    for seq_len in SEQ_LENGTHS
}

plt.rcParams.update({
    "font.size": 9,
    "axes.linewidth": 0.8,
    "legend.handlelength": 0.8,
    "legend.handletextpad": 0.3,
})

fig = plt.figure(figsize=(13.8, 4.9))
grid = GridSpec(2, 3, figure=fig, height_ratios=[10, 3.0], hspace=0.30, wspace=0.24)
axes = [fig.add_subplot(grid[0, idx]) for idx in range(3)]
caption_axes = [fig.add_subplot(grid[1, idx]) for idx in range(3)]

# Panel (a): Normalized latency comparison
ax = axes[0]
comp_width = 0.28
for batch in BATCH_PER_GPU:
    for seq_len in SEQ_LENGTHS:
        on_key = ("on", seq_len, batch)
        off_key = ("off", seq_len, batch)
        if on_key not in data or off_key not in data or not data[on_key]["total"]:
            continue
        x = x_for[(batch, seq_len)]
        ratio = float(data[off_key]["total"]) / float(data[on_key]["total"])
        ax.bar(x - comp_width / 2, 1.0, comp_width, color="#d96b27", edgecolor="black", linewidth=0.5,
               label="w/ reordering" if batch == BATCH_PER_GPU[0] and seq_len == SEQ_LENGTHS[0] else "")
        ax.bar(x + comp_width / 2, ratio, comp_width, color="#0b4a1f", edgecolor="black", linewidth=0.5,
               label="w/o reordering" if batch == BATCH_PER_GPU[0] and seq_len == SEQ_LENGTHS[0] else "")

ax.set_yscale("log")
ax.set_ylim(0.1, 1000)
ax.set_ylabel("Normalized latency")
_add_paper_xlabels(ax, positions, seq_tick_labels, group_centers, BATCH_PER_GPU)
ax.grid(axis="y", alpha=0.30, linewidth=0.6)
comp_handles, comp_labels = ax.get_legend_handles_labels()

# Panel (b) and (c): Breakdown
categories = ["kv_decompress", "score_context", "out_proj", "etc"]
cat_labels = ["KV decompress", "Score + Context", "Out projection", "Etc"]
cat_colors = ["#d7191c", "#f6c744", "#a8cf8d", "#d9d9d9"]
width = 0.52

for panel, reorder in [(1, "off"), (2, "on")]:
    ax = axes[panel]
    for batch in BATCH_PER_GPU:
        for seq_len in SEQ_LENGTHS:
            key = (reorder, seq_len, batch)
            if key not in data:
                continue
            bottom = 0.0
            bar_x = x_for[(batch, seq_len)]
            for color, cat in zip(cat_colors, categories):
                val = float(data[key][cat]) / 1e6
                ax.bar(bar_x, val, width * 0.85, bottom=bottom, color=color, edgecolor="black", linewidth=0.3)
                bottom += val
    ax.set_ylabel("Attention block latency (ms)")
    _add_paper_xlabels(ax, positions, seq_tick_labels, group_centers, BATCH_PER_GPU)
    ax.grid(axis="y", alpha=0.30, linewidth=0.6)
    if panel == 1:
        ax.set_ylim(0, max(100.0, ax.get_ylim()[1]))

# Legends
patches = [plt.Rectangle((0, 0), 1, 1, color=c, ec="black", lw=0.5) for c in cat_colors]
fig.legend(comp_handles, comp_labels, loc="upper center", bbox_to_anchor=(0.22, 0.835), ncol=2, frameon=False, fontsize=8)
fig.legend(patches, cat_labels, loc="upper center", bbox_to_anchor=(0.64, 0.835), ncol=4, frameon=False, fontsize=8)

for ax in axes:
    ax.set_xlim(group_centers[0] - 1.0, group_centers[-1] + 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.subplots_adjust(left=0.075, right=0.985, bottom=0.09, top=0.71)

# Captions
captions = [
    "(a) Attention block latency comparison\nbetween w/ and w/o reordering",
    "(b) Attention block breakdown w/o reordering",
    "(c) Attention block breakdown w/ reordering",
]
for ax, caption in zip(caption_axes, captions):
    ax.axis("off")
    ax.text(0.5, 0.14, caption, ha="center", va="center", fontsize=9.5, transform=ax.transAxes)

fig.savefig(PLOT_DIR / "hbm3e_figure6_attention_breakdown.png", dpi=200)
print(f"Saved {PLOT_DIR / 'hbm3e_figure6_attention_breakdown.png'}")

# Print summary
print("\n=== Normalized Latency Summary (Ramulator ON) ===")
print(f"{'Batch':<10} {'Seq Len':<10} {'Reorder ON (ms)':<18} {'Reorder OFF (ms)':<18} {'Ratio':<10}")
print("-" * 66)

for batch in BATCH_PER_GPU:
    for seq_len in SEQ_LENGTHS:
        on_key = ("on", seq_len, batch)
        off_key = ("off", seq_len, batch)
        if on_key in data and off_key in data:
            on_val = data[on_key]["total"] / 1e6
            off_val = data[off_key]["total"] / 1e6
            ratio = off_val / on_val if on_val > 0 else 0
            print(f"{batch:<10} {seq_len:<10} {on_val:<18.2f} {off_val:<18.2f} {ratio:<10.2f}x")
    print()
