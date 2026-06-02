#!/usr/bin/env python3
"""HBM3E Memory Architecture Analysis.

Analyzes HBM3E data comparing:
- Reordering ON vs OFF
- Different Batch sizes (32, 64, 128, 256)
- Different Sequence lengths (2048, 4096, 8192)
- Ramulator ON vs OFF
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.sim_utils import (
    attention_breakdown_from_csv,
    read_csv_rows,
    write_summary_csv,
)

EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
PLOT_DIR = EXP_DIR / "plots"

BATCH_PER_GPU = [32, 64, 128, 256]
SEQ_LENGTHS = [2048, 4096, 8192]


def collect_hbm3e_results():
    """Collect HBM3E experiment results."""
    rows = []
    for csv_file in sorted(DATA_DIR.glob("result_hbm3e_*.csv")):
        parts = csv_file.stem.split("_")
        try:
            batch = int(next(p[1:] for p in parts if p.startswith("b")))
            seq_len = int(next(p[1:] for p in parts if p.startswith("l")))
            reorder_idx = parts.index("reorder") + 1
            reorder = parts[reorder_idx]
            ramul_idx = parts.index("ramul") + 1
            ramul = parts[ramul_idx]
        except (StopIteration, ValueError):
            continue
        rows.append({
            "reorder": reorder,
            "ramulator": ramul,
            "batch_size": batch,
            "seq_len": seq_len,
            **attention_breakdown_from_csv(csv_file),
        })
    return rows


def print_comparison_table(rows):
    """Print detailed comparison table."""
    data = {(r["reorder"], r["ramulator"], r["seq_len"], r["batch_size"]): r for r in rows}

    print("\n" + "=" * 100)
    print("HBM3E 内存架构仿真对比分析")
    print("=" * 100)

    # 1. Ramulator 对比
    print("\n### 1. Ramulator 启用前后对比 (Reordering ON)")
    print(f"\n{'Batch/GPU':<12} {'Seq Len':<10} {'Ideal (ms)':<14} {'Ramulator (ms)':<16} {'Overhead':<12} {'Ratio':<8}")
    print("-" * 72)

    for seq_len in SEQ_LENGTHS:
        for batch in BATCH_PER_GPU:
            key_off = ("on", "off", seq_len, batch)
            key_on = ("on", "on", seq_len, batch)
            val_off = data.get(key_off, {}).get("total", 0.0) / 1e6
            val_on = data.get(key_on, {}).get("total", 0.0) / 1e6
            if val_off > 0 and val_on > 0:
                overhead = ((val_on / val_off) - 1) * 100
                ratio = val_on / val_off
                print(f"{batch:<12} {seq_len:<10} {val_off:<14.2f} {val_on:<16.2f} +{overhead:<11.1f}% {ratio:<8.2f}x")
        print()

    # 2. Reordering 对比
    print("\n### 2. Reordering 效果对比 (Ramulator OFF)")
    print(f"\n{'Batch/GPU':<12} {'Seq Len':<10} {'Reorder ON':<14} {'Reorder OFF':<14} {'Improvement':<12} {'Ratio':<8}")
    print("-" * 70)

    for seq_len in SEQ_LENGTHS:
        for batch in BATCH_PER_GPU:
            key_on = ("on", "off", seq_len, batch)
            key_off = ("off", "off", seq_len, batch)
            val_on = data.get(key_on, {}).get("total", 0.0) / 1e6
            val_off = data.get(key_off, {}).get("total", 0.0) / 1e6
            if val_on > 0 and val_off > 0:
                improvement = ((val_off - val_on) / val_off) * 100
                ratio = val_off / val_on
                print(f"{batch:<12} {seq_len:<10} {val_on:<14.2f} {val_off:<14.2f} {improvement:<11.1f}% {ratio:<8.2f}x")
        print()

    # 3. Reordering 对比 (Ramulator ON)
    print("\n### 3. Reordering 效果对比 (Ramulator ON)")
    print(f"\n{'Batch/GPU':<12} {'Seq Len':<10} {'Reorder ON':<14} {'Reorder OFF':<14} {'Improvement':<12} {'Ratio':<8}")
    print("-" * 70)

    for seq_len in SEQ_LENGTHS:
        for batch in BATCH_PER_GPU:
            key_on = ("on", "on", seq_len, batch)
            key_off = ("off", "on", seq_len, batch)
            val_on = data.get(key_on, {}).get("total", 0.0) / 1e6
            val_off = data.get(key_off, {}).get("total", 0.0) / 1e6
            if val_on > 0 and val_off > 0:
                improvement = ((val_off - val_on) / val_off) * 100
                ratio = val_off / val_on
                print(f"{batch:<12} {seq_len:<10} {val_on:<14.2f} {val_off:<14.2f} {improvement:<11.1f}% {ratio:<8.2f}x")
        print()

    # 4. 详细分解
    print("\n### 4. 注意力机制分解 (Reordering ON, Ramulator OFF)")
    print(f"\n{'Batch/GPU':<12} {'Seq Len':<10} {'KV Decomp':<12} {'Score+Ctx':<12} {'Out Proj':<12} {'Etc':<12} {'Total':<12}")
    print("-" * 70)

    for seq_len in SEQ_LENGTHS:
        for batch in BATCH_PER_GPU:
            key = ("on", "off", seq_len, batch)
            r = data.get(key, {})
            kv = r.get("kv_decompress", 0.0) / 1e6
            score = r.get("score_context", 0.0) / 1e6
            out = r.get("out_proj", 0.0) / 1e6
            etc = r.get("etc", 0.0) / 1e6
            total = r.get("total", 0.0) / 1e6
            print(f"{batch:<12} {seq_len:<10} {kv:<12.2f} {score:<12.2f} {out:<12.2f} {etc:<12.2f} {total:<12.2f}")
        print()


def plot_hbm3e_comparison(rows):
    """Generate HBM3E comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    data = {(r["reorder"], r["ramulator"], r["seq_len"], r["batch_size"]): r for r in rows}

    # Figure 1: Ramulator comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    # First pass: collect all values to determine y-axis range
    all_vals = []
    for seq_len in SEQ_LENGTHS:
        for batch in BATCH_PER_GPU:
            key_off = ("on", "off", seq_len, batch)
            key_on = ("on", "on", seq_len, batch)
            all_vals.append(data.get(key_off, {}).get("total", 0.0) / 1e6)
            all_vals.append(data.get(key_on, {}).get("total", 0.0) / 1e6)
    y_min = min(v for v in all_vals if v > 0) * 0.8
    y_max = max(all_vals) * 1.3

    for idx, seq_len in enumerate(SEQ_LENGTHS):
        ax = axes[idx]
        x = np.arange(len(BATCH_PER_GPU))
        width = 0.35

        ideal_vals = []
        ramul_vals = []
        for batch in BATCH_PER_GPU:
            key_off = ("on", "off", seq_len, batch)
            key_on = ("on", "on", seq_len, batch)
            ideal_vals.append(data.get(key_off, {}).get("total", 0.0) / 1e6)
            ramul_vals.append(data.get(key_on, {}).get("total", 0.0) / 1e6)

        ax.bar(x - width/2, ideal_vals, width, label="Ideal", color="#2196F3", edgecolor="black", lw=0.5)
        ax.bar(x + width/2, ramul_vals, width, label="Ramulator", color="#FF5722", edgecolor="black", lw=0.5)

        ax.set_xlabel("Batch per GPU")
        if idx == 0:
            ax.set_ylabel("Latency (ms)")
        ax.set_title(f"Seq Len = {seq_len}")
        ax.set_xticks(x)
        ax.set_xticklabels(BATCH_PER_GPU)
        if idx == 0:
            ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)

        # Add ratio labels
        for i, (iv, rv) in enumerate(zip(ideal_vals, ramul_vals)):
            if iv > 0:
                ratio = rv / iv
                ax.text(i, max(iv, rv) * 1.02, f"{ratio:.2f}x", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "hbm3e_ramulator_comparison.png", dpi=200, bbox_inches="tight")
    print(f"Saved {PLOT_DIR / 'hbm3e_ramulator_comparison.png'}")

    # Figure 1b: Ramulator comparison (Reordering OFF)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    # First pass: collect all values to determine y-axis range
    all_vals_off = []
    for seq_len in SEQ_LENGTHS:
        for batch in BATCH_PER_GPU:
            key_off = ("off", "off", seq_len, batch)
            key_on = ("off", "on", seq_len, batch)
            all_vals_off.append(data.get(key_off, {}).get("total", 0.0) / 1e6)
            all_vals_off.append(data.get(key_on, {}).get("total", 0.0) / 1e6)
    y_min_off = min(v for v in all_vals_off if v > 0) * 0.8
    y_max_off = max(all_vals_off) * 1.3

    for idx, seq_len in enumerate(SEQ_LENGTHS):
        ax = axes[idx]
        x = np.arange(len(BATCH_PER_GPU))
        width = 0.35

        ideal_vals = []
        ramul_vals = []
        for batch in BATCH_PER_GPU:
            key_off = ("off", "off", seq_len, batch)
            key_on = ("off", "on", seq_len, batch)
            ideal_vals.append(data.get(key_off, {}).get("total", 0.0) / 1e6)
            ramul_vals.append(data.get(key_on, {}).get("total", 0.0) / 1e6)

        ax.bar(x - width/2, ideal_vals, width, label="Ideal", color="#2196F3", edgecolor="black", lw=0.5)
        ax.bar(x + width/2, ramul_vals, width, label="Ramulator", color="#FF5722", edgecolor="black", lw=0.5)

        ax.set_xlabel("Batch per GPU")
        if idx == 0:
            ax.set_ylabel("Latency (ms)")
        ax.set_title(f"Seq Len = {seq_len}")
        ax.set_xticks(x)
        ax.set_xticklabels(BATCH_PER_GPU)
        if idx == 0:
            ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_yscale("log")
        ax.set_ylim(y_min_off, y_max_off)

        # Add ratio labels
        for i, (iv, rv) in enumerate(zip(ideal_vals, ramul_vals)):
            if iv > 0:
                ratio = rv / iv
                ax.text(i, max(iv, rv) * 1.02, f"{ratio:.2f}x", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "hbm3e_ramulator_comparison_no_reorder.png", dpi=200, bbox_inches="tight")
    print(f"Saved {PLOT_DIR / 'hbm3e_ramulator_comparison_no_reorder.png'}")

    # Figure 2: Reordering comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, seq_len in enumerate(SEQ_LENGTHS):
        ax = axes[idx]
        x = np.arange(len(BATCH_PER_GPU))
        width = 0.35

        reorder_on = []
        reorder_off = []
        for batch in BATCH_PER_GPU:
            key_on = ("on", "off", seq_len, batch)
            key_off = ("off", "off", seq_len, batch)
            reorder_on.append(data.get(key_on, {}).get("total", 0.0) / 1e6)
            reorder_off.append(data.get(key_off, {}).get("total", 0.0) / 1e6)

        ax.bar(x - width/2, reorder_on, width, label="Reorder ON", color="#4CAF50", edgecolor="black", lw=0.5)
        ax.bar(x + width/2, reorder_off, width, label="Reorder OFF", color="#F44336", edgecolor="black", lw=0.5)

        ax.set_xlabel("Batch per GPU")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"Seq Len = {seq_len}")
        ax.set_xticks(x)
        ax.set_xticklabels(BATCH_PER_GPU)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_yscale("log")

        # Add ratio labels
        for i, (on, off) in enumerate(zip(reorder_on, reorder_off)):
            if on > 0:
                ratio = off / on
                ax.text(i, max(on, off) * 1.02, f"{ratio:.1f}x", ha="center", fontsize=8)

    fig.suptitle("HBM3E: Reordering Impact on Latency (Ramulator OFF)", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "hbm3e_reordering_comparison.png", dpi=200, bbox_inches="tight")
    print(f"Saved {PLOT_DIR / 'hbm3e_reordering_comparison.png'}")

    # Figure 3: Breakdown comparison
    categories = ["kv_decompress", "score_context", "out_proj", "etc"]
    cat_labels = ["KV decompress", "Score + Context", "Out projection", "Etc"]
    cat_colors = ["#d7191c", "#f6c744", "#a8cf8d", "#d9d9d9"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for idx, (reorder, ramul) in enumerate([("on", "off"), ("on", "on"), ("off", "off"), ("off", "on")]):
        ax = axes[idx // 2][idx % 2]
        x = np.arange(len(BATCH_PER_GPU))
        width = 0.6

        bottoms = np.zeros(len(BATCH_PER_GPU))
        for ci, cat in enumerate(categories):
            vals = []
            for batch in BATCH_PER_GPU:
                key = (reorder, ramul, 4096, batch)
                v = float(data.get(key, {}).get(cat, 0.0)) / 1e6
                vals.append(v)
            ax.bar(x, vals, width, bottom=bottoms, color=cat_colors[ci],
                   edgecolor="black", lw=0.3, label=cat_labels[ci] if idx == 0 else "")
            bottoms += np.array(vals)

        ax.set_xlabel("Batch per GPU")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"Reorder={reorder.upper()}, Ramulator={ramul.upper()} (L=4096)")
        ax.set_xticks(x)
        ax.set_xticklabels(BATCH_PER_GPU)
        ax.grid(axis="y", alpha=0.3)
        ax.set_yscale("log")
        if idx == 0:
            ax.legend(fontsize=8, loc="upper left", ncol=2)

    fig.suptitle("HBM3E: Attention Breakdown by Configuration", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "hbm3e_attention_breakdown.png", dpi=200, bbox_inches="tight")
    print(f"Saved {PLOT_DIR / 'hbm3e_attention_breakdown.png'}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--table", action="store_true", help="Print comparison table")
    parser.add_argument("--all", action="store_true", help="Generate plots and table")
    args = parser.parse_args()

    if not (args.plot or args.table or args.all):
        args.all = True

    rows = collect_hbm3e_results()
    if not rows:
        print("No HBM3E results found.")
        return

    print(f"Found {len(rows)} HBM3E result entries")

    # Write summary
    write_summary_csv(DATA_DIR / "summary_hbm3e.csv", rows)
    print(f"Saved summary to {DATA_DIR / 'summary_hbm3e.csv'}")

    if args.table or args.all:
        print_comparison_table(rows)

    if args.plot or args.all:
        plot_hbm3e_comparison(rows)


if __name__ == "__main__":
    main()
