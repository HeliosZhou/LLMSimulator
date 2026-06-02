#!/usr/bin/env python3
"""Plot HBM3E energy breakdown comparison."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.sim_utils import average_rows, read_csv_rows

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
PLOT_DIR = EXP_DIR / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_PER_GPU = [32, 64, 128, 256]

# Load all data
data = {}
for reorder in ["on", "off"]:
    for ramul in ["on", "off"]:
        for batch in BATCH_PER_GPU:
            filename = f"result_hbm3e_b{batch}_l4096_reorder_{reorder}_ramul_{ramul}.csv"
            rows = read_csv_rows(DATA_DIR / filename)
            data[(reorder, ramul, batch)] = average_rows(rows, "t2t")

# Energy categories
energy_categories = [
    "act_energy",
    "read_energy",
    "write_energy",
    "ref_energy",
    "background_energy",
    "mac_energy",
]
energy_labels = [
    "ACT Energy",
    "READ Energy",
    "WRITE Energy",
    "REF Energy",
    "Background Energy",
    "MAC Energy",
]
energy_colors = ["#d7191c", "#2196F3", "#4CAF50", "#7B1FA2", "#607D8B", "#FF9800"]

# Figure: Energy breakdown by configuration (2x2 subplots)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for idx, (reorder, ramul) in enumerate([("on", "off"), ("on", "on"), ("off", "off"), ("off", "on")]):
    ax = axes[idx // 2][idx % 2]
    x = np.arange(len(BATCH_PER_GPU))
    width = 0.6

    bottoms = np.zeros(len(BATCH_PER_GPU))
    for ci, cat in enumerate(energy_categories):
        vals = []
        for batch in BATCH_PER_GPU:
            key = (reorder, ramul, batch)
            v = float(data.get(key, {}).get(cat, 0.0)) / 1e9  # Convert to nJ
            vals.append(v)
        ax.bar(x, vals, width, bottom=bottoms, color=energy_colors[ci],
               edgecolor="black", lw=0.3, label=energy_labels[ci] if idx == 0 else "")
        bottoms += np.array(vals)

    ax.set_xlabel("Batch per GPU")
    ax.set_ylabel("Energy (nJ)")
    ax.set_title(f"Reorder={reorder.upper()}, Ramulator={ramul.upper()} (L=4096)")
    ax.set_xticks(x)
    ax.set_xticklabels(BATCH_PER_GPU)
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")
    if idx == 0:
        ax.legend(fontsize=8, loc="upper left", ncol=2)

fig.suptitle("HBM3E: Energy Breakdown by Configuration", fontsize=14)
fig.tight_layout()
fig.savefig(PLOT_DIR / "hbm3e_energy_breakdown.png", dpi=200, bbox_inches="tight")
print(f"Saved {PLOT_DIR / 'hbm3e_energy_breakdown.png'}")

# Print summary
print("\n=== Energy Breakdown Summary (nJ) ===")
print(f"{'Config':<30} {'ACT':<12} {'READ':<12} {'WRITE':<12} {'REF':<12} {'BG':<12} {'MAC':<12} {'Total':<12}")
print("-" * 116)

for reorder in ["on", "off"]:
    for ramul in ["on", "off"]:
        for batch in BATCH_PER_GPU:
            key = (reorder, ramul, batch)
            d = data.get(key, {})
            act = d.get("act_energy", 0) / 1e9
            read = d.get("read_energy", 0) / 1e9
            write = d.get("write_energy", 0) / 1e9
            ref = d.get("ref_energy", 0) / 1e9
            bg = d.get("background_energy", 0) / 1e9
            mac = d.get("mac_energy", 0) / 1e9
            total = act + read + write + ref + bg + mac
            config = f"Reorder={reorder.upper()}, Ramul={ramul.upper()}, B={batch}"
            print(f"{config:<30} {act:<12.2f} {read:<12.2f} {write:<12.2f} {ref:<12.2f} {bg:<12.2f} {mac:<12.2f} {total:<12.2f}")
        print()
