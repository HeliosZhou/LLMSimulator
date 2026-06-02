#!/usr/bin/env python3
"""Plot HBM3E energy consumption comparison."""

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

# Load data
def load_data(reorder, ramul):
    filename = f"result_hbm3e_b32_l4096_reorder_{reorder}_ramul_{ramul}.csv"
    rows = read_csv_rows(DATA_DIR / filename)
    return average_rows(rows, "t2t")

# Reordering OFF
off_ideal = load_data("off", "off")
off_ramul = load_data("off", "on")

# Reordering ON
on_ideal = load_data("on", "off")
on_ramul = load_data("on", "on")

# Energy fields
energy_fields = [
    ("act_energy", "ACT Energy"),
    ("read_energy", "READ Energy"),
    ("write_energy", "WRITE Energy"),
    ("ref_energy", "REF Energy"),
    ("background_energy", "Background Energy"),
    ("mac_energy", "MAC Energy"),
]

# Figure 1: Energy comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Reordering OFF
ax = axes[0]
x = np.arange(len(energy_fields))
width = 0.35

ideal_vals = [off_ideal[f] / 1e9 for f, _ in energy_fields]
ramul_vals = [off_ramul[f] / 1e9 for f, _ in energy_fields]

bars1 = ax.bar(x - width/2, ideal_vals, width, label="Ideal", color="#2196F3", edgecolor="black", lw=0.5)
bars2 = ax.bar(x + width/2, ramul_vals, width, label="Ramulator", color="#FF5722", edgecolor="black", lw=0.5)

ax.set_xlabel("Energy Component")
ax.set_ylabel("Energy (nJ)")
ax.set_title("Reordering OFF")
ax.set_xticks(x)
ax.set_xticklabels([name for _, name in energy_fields], rotation=15)
ax.legend()
ax.grid(axis="y", alpha=0.3)
ax.set_yscale("log")

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height * 1.02, f'{height:.1f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height * 1.02, f'{height:.1f}', ha='center', va='bottom', fontsize=8)

# Reordering ON
ax = axes[1]
ideal_vals2 = [on_ideal[f] / 1e9 for f, _ in energy_fields]
ramul_vals2 = [on_ramul[f] / 1e9 for f, _ in energy_fields]

bars3 = ax.bar(x - width/2, ideal_vals2, width, label="Ideal", color="#2196F3", edgecolor="black", lw=0.5)
bars4 = ax.bar(x + width/2, ramul_vals2, width, label="Ramulator", color="#FF5722", edgecolor="black", lw=0.5)

ax.set_xlabel("Energy Component")
ax.set_ylabel("Energy (nJ)")
ax.set_title("Reordering ON")
ax.set_xticks(x)
ax.set_xticklabels([name for _, name in energy_fields], rotation=15)
ax.legend()
ax.grid(axis="y", alpha=0.3)
ax.set_yscale("log")

# Add value labels
for bar in bars3:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height * 1.02, f'{height:.1f}', ha='center', va='bottom', fontsize=8)
for bar in bars4:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height * 1.02, f'{height:.1f}', ha='center', va='bottom', fontsize=8)

fig.suptitle("HBM3E Energy Consumption Comparison (B=32, L=4096)", fontsize=14)
fig.tight_layout()
fig.savefig(PLOT_DIR / "hbm3e_energy_comparison.png", dpi=200, bbox_inches="tight")
print(f"Saved {PLOT_DIR / 'hbm3e_energy_comparison.png'}")

# Figure 2: Energy increase ratio
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(energy_fields))
width = 0.35

# Calculate increase ratios
off_ratios = [off_ramul[f] / off_ideal[f] if off_ideal[f] > 0 else 0 for f, _ in energy_fields]
on_ratios = [on_ramul[f] / on_ideal[f] if on_ideal[f] > 0 else 0 for f, _ in energy_fields]

bars1 = ax.bar(x - width/2, off_ratios, width, label="Reordering OFF", color="#4CAF50", edgecolor="black", lw=0.5)
bars2 = ax.bar(x + width/2, on_ratios, width, label="Reordering ON", color="#9C27B0", edgecolor="black", lw=0.5)

ax.set_xlabel("Energy Component")
ax.set_ylabel("Ramulator / Ideal Ratio")
ax.set_title("HBM3E Energy Increase Ratio (B=32, L=4096)")
ax.set_xticks(x)
ax.set_xticklabels([name for _, name in energy_fields], rotation=15)
ax.legend()
ax.grid(axis="y", alpha=0.3)
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No change")

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height * 1.02, f'{height:.2f}x', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height * 1.02, f'{height:.2f}x', ha='center', va='bottom', fontsize=9)

fig.tight_layout()
fig.savefig(PLOT_DIR / "hbm3e_energy_increase_ratio.png", dpi=200, bbox_inches="tight")
print(f"Saved {PLOT_DIR / 'hbm3e_energy_increase_ratio.png'}")

# Print summary
print("\n=== 能量消耗对比 ===")
print(f"{'组件':<15} {'Reorder OFF (Ideal)':<20} {'Reorder OFF (Ramulator)':<25} {'Reorder ON (Ideal)':<20} {'Reorder ON (Ramulator)':<25}")
print("-" * 105)
for field, name in energy_fields:
    print(f"{name:<15} {off_ideal[field]/1e9:<20.2f} {off_ramul[field]/1e9:<25.2f} {on_ideal[field]/1e9:<20.2f} {on_ramul[field]/1e9:<25.2f}")
