#!/usr/bin/env python3
"""Memory Type Comparison: HBM3E vs GDDR6 vs DDR5 on LLM inference latency."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.sim_utils import (
    SimPoint,
    add_common_args,
    attention_breakdown_from_csv,
    run_simulation,
    write_summary_csv,
)

EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
PLOT_DIR = EXP_DIR / "plots"

BATCH_PER_GPU = [32, 64, 128, 256]
SEQ_LENGTHS = [2048, 4096, 8192]
NUM_NODE = 4
NUM_DEVICE = 8
NUM_GPUS = NUM_NODE * NUM_DEVICE
PRECISION_BYTE = 2

# Memory configs: (bandwidth_B_per_s, capacity_bytes)
MEM_CONFIGS: dict[str, dict[str, float | str]] = {
    "hbm3e": {
        "label": "HBM3E (8 TB/s)",
        "bw": 8.0e12,
        "cap": 192 * 1024**3,
        "color": "#2166ac",
    },
    "gddr6": {
        "label": "GDDR6 (512 GB/s)",
        "bw": 512e9,
        "cap": 192 * 1024**3,
        "color": "#4dac26",
    },
    "ddr5": {
        "label": "DDR5 (64 GB/s)",
        "bw": 64e9,
        "cap": 192 * 1024**3,
        "color": "#d7191c",
    },
}

MEM_ORDER = ["hbm3e", "gddr6", "ddr5"]


def collect_results() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for csv_file in sorted(DATA_DIR.glob("result_*_b*_l*_absorb_*.csv")):
        parts = csv_file.stem.split("_")
        try:
            mem = next(p for p in parts if p in MEM_CONFIGS)
            batch = int(next(p[1:] for p in parts if p.startswith("b")))
            seq_len = int(next(p[1:] for p in parts if p.startswith("l")))
            absorb = parts[-1]
        except (StopIteration, ValueError):
            continue
        rows.append(
            {
                "mem_type": mem,
                "absorb": absorb,
                "batch_size": batch,
                "seq_len": seq_len,
                **attention_breakdown_from_csv(csv_file),
            }
        )
    return rows


def run_experiments(args) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    batches = [32] if args.quick else BATCH_PER_GPU
    seq_lens = [2048] if args.quick else SEQ_LENGTHS

    for mem_name, mem_cfg in MEM_CONFIGS.items():
        for absorb_label, use_absorb in [("on", True), ("off", False)]:
            for seq_len in seq_lens:
                for batch in batches:
                    name = f"result_{mem_name}_b{batch}_l{seq_len}_absorb_{absorb_label}.csv"
                    point = SimPoint(
                        gpu_gen="B200",
                        num_node=NUM_NODE,
                        num_device=NUM_DEVICE,
                        batch_size=batch * NUM_GPUS,
                        seq_len=seq_len,
                        precision_byte=PRECISION_BYTE,
                        use_absorb=use_absorb,
                        memory_bandwidth=mem_cfg["bw"],
                        memory_capacity=mem_cfg["cap"],
                    )
                    print(f"[{mem_name}] absorb={absorb_label} B/GPU={batch} L={seq_len}")
                    if run_simulation(point, DATA_DIR, name, timeout=args.timeout, skip_existing=not args.overwrite) is None:
                        print(f"  skipped: {name}")


def plot_results(rows: list[dict[str, float | int | str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    data = {(r["mem_type"], r["absorb"], r["seq_len"], r["batch_size"]): r for r in rows}

    categories = ["kv_decompress", "score_context", "out_proj", "etc"]
    cat_labels = ["KV decompress", "Score + Context", "Out projection", "Etc"]
    cat_colors = ["#d7191c", "#f6c744", "#a8cf8d", "#d9d9d9"]

    # ========== Figure 1: Memory Type Comparison ==========
    # Layout: 3 rows (one per seq_len) x 2 columns (total latency + breakdown)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex="col")

    for row_idx, seq_len in enumerate(SEQ_LENGTHS):
        # --- Left column: Total latency grouped bar ---
        ax = axes[row_idx, 0]
        x = np.arange(len(BATCH_PER_GPU))
        n_mem = len(MEM_ORDER)
        bar_width = 0.25
        gap = 0.05

        for mi, mem_name in enumerate(MEM_ORDER):
            totals = []
            for batch in BATCH_PER_GPU:
                key = (mem_name, "on", seq_len, batch)
                val = data.get(key, {}).get("total", 0.0)
                totals.append(val / 1e6)
            bar_x = x + mi * (bar_width + gap)
            ax.bar(bar_x, totals, bar_width,
                   color=MEM_CONFIGS[mem_name]["color"],
                   edgecolor="black", linewidth=0.4,
                   label=MEM_CONFIGS[mem_name]["label"])

        ax.set_yscale("log")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"L = {seq_len}", fontsize=11)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(BATCH_PER_GPU)
        if row_idx == 0:
            ax.legend(fontsize=8, loc="upper left")
        if row_idx == 2:
            ax.set_xlabel("Batch per GPU")

        # --- Right column: Breakdown stacked bar (absorb=on) ---
        ax = axes[row_idx, 1]
        stack_width = 0.7
        for mi, mem_name in enumerate(MEM_ORDER):
            bottoms = np.zeros(len(BATCH_PER_GPU))
            for ci, cat in enumerate(categories):
                vals = []
                for batch in BATCH_PER_GPU:
                    key = (mem_name, "on", seq_len, batch)
                    v = float(data.get(key, {}).get(cat, 0.0)) / 1e6
                    vals.append(v)
                ax.bar(x + mi * (bar_width + gap), vals, bar_width,
                       bottom=bottoms, color=cat_colors[ci],
                       edgecolor="black", linewidth=0.3,
                       label=cat_labels[ci] if row_idx == 0 and mi == 0 else "")
                bottoms += np.array(vals)

        ax.set_yscale("log")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"L = {seq_len} — Breakdown", fontsize=11)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(BATCH_PER_GPU)
        if row_idx == 0:
            ax.legend(fontsize=7, loc="upper left", ncol=2)
        if row_idx == 2:
            ax.set_xlabel("Batch per GPU")

    fig.suptitle("Impact of Memory Type on DeepSeek-V3 Decode (B200, 32 GPUs, absorb=on)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figure_memory_comparison.png", dpi=200, bbox_inches="tight")
    print(f"Saved {PLOT_DIR / 'figure_memory_comparison.png'}")

    # ========== Figure 2: Latency Ratio vs HBM3E ==========
    # Left: absorb=on, Right: absorb=off
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    markers = {"gddr6": "s", "ddr5": "^"}
    linestyles = {2048: "-", 4096: "--", 8192: ":"}

    for pi, absorb in enumerate(["on", "off"]):
        ax = axes2[pi]
        for mem_name in ["gddr6", "ddr5"]:
            for seq_len in SEQ_LENGTHS:
                ratios = []
                valid_batches = []
                for batch in BATCH_PER_GPU:
                    base = data.get(("hbm3e", absorb, seq_len, batch), {}).get("total", 0.0)
                    alt = data.get((mem_name, absorb, seq_len, batch), {}).get("total", 0.0)
                    if base > 0 and alt > 0:
                        ratios.append(alt / base)
                        valid_batches.append(batch)
                if valid_batches:
                    ax.plot(valid_batches, ratios,
                            marker=markers[mem_name],
                            color=MEM_CONFIGS[mem_name]["color"],
                            linestyle=linestyles.get(seq_len, "-"),
                            linewidth=1.5, markersize=7,
                            label=f"{MEM_CONFIGS[mem_name]['label']} L={seq_len}")

        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="HBM3E baseline")
        ax.set_xlabel("Batch per GPU")
        ax.set_ylabel("Latency ratio vs HBM3E")
        ax.set_title(f"absorb={absorb}")
        ax.set_xticks(BATCH_PER_GPU)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

    fig2.suptitle("Memory Type Slowdown vs HBM3E Baseline (B200, 32 GPUs)", fontsize=13)
    fig2.tight_layout()
    fig2.savefig(PLOT_DIR / "figure_memory_slowdown.png", dpi=200, bbox_inches="tight")
    print(f"Saved {PLOT_DIR / 'figure_memory_slowdown.png'}")

    # ========== Figure 3: Breakdown for absorb=on vs off (L=4096) ==========
    # Top: absolute (log), Bottom: normalized (linear)
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 9))
    bar_w = 0.25
    bar_gap = 0.05
    seq_len = 4096

    for pi, absorb in enumerate(["on", "off"]):
        # --- Top row: absolute values (log scale) ---
        ax = axes3[0, pi]
        x = np.arange(len(BATCH_PER_GPU))
        for mi, mem_name in enumerate(MEM_ORDER):
            bottoms = np.zeros(len(BATCH_PER_GPU))
            for ci, cat in enumerate(categories):
                vals = []
                for batch in BATCH_PER_GPU:
                    key = (mem_name, absorb, seq_len, batch)
                    v = float(data.get(key, {}).get(cat, 0.0)) / 1e6
                    vals.append(v)
                ax.bar(x + mi * (bar_w + bar_gap), vals, bar_w,
                       bottom=bottoms, color=cat_colors[ci],
                       edgecolor="black", linewidth=0.3,
                       label=cat_labels[ci] if mi == 0 else "")
                bottoms += np.array(vals)
        ax.set_yscale("log")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"absorb={absorb}, L={seq_len} (absolute)")
        ax.set_xticks(x + bar_w)
        ax.set_xticklabels(BATCH_PER_GPU)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        if pi == 0:
            ax.legend(fontsize=7, loc="upper left", ncol=2)

        # --- Bottom row: normalized (100% stacked, linear) ---
        ax = axes3[1, pi]
        for mi, mem_name in enumerate(MEM_ORDER):
            totals = []
            comp_vals = {cat: [] for cat in categories}
            for batch in BATCH_PER_GPU:
                key = (mem_name, absorb, seq_len, batch)
                total = float(data.get(key, {}).get("total", 0.0))
                totals.append(total)
                for cat in categories:
                    comp_vals[cat].append(float(data.get(key, {}).get(cat, 0.0)))
            bottoms = np.zeros(len(BATCH_PER_GPU))
            for ci, cat in enumerate(categories):
                pcts = []
                for bi in range(len(BATCH_PER_GPU)):
                    if totals[bi] > 0:
                        pcts.append(comp_vals[cat][bi] / totals[bi] * 100)
                    else:
                        pcts.append(0)
                ax.bar(x + mi * (bar_w + bar_gap), pcts, bar_w,
                       bottom=bottoms, color=cat_colors[ci],
                       edgecolor="black", linewidth=0.3)
                bottoms += np.array(pcts)
        ax.set_ylabel("Proportion (%)")
        ax.set_title(f"absorb={absorb}, L={seq_len} (normalized)")
        ax.set_xticks(x + bar_w)
        ax.set_xticklabels(BATCH_PER_GPU)
        ax.set_xlabel("Batch per GPU")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        if pi == 1:
            cat_patches = [plt.Rectangle((0, 0), 1, 1, color=c, ec="black", lw=0.5)
                           for c in cat_colors]
            ax.legend(handles=cat_patches, labels=cat_labels,
                      fontsize=8, loc="center right")

    # Memory type legend at figure level
    from matplotlib.patches import Patch
    mem_legend_elements = [
        Patch(facecolor="white", edgecolor="black", label="HBM3E (8 TB/s)"),
        Patch(facecolor="white", edgecolor="black", label="GDDR6 (512 GB/s)"),
        Patch(facecolor="white", edgecolor="black", label="DDR5 (64 GB/s)"),
    ]
    fig3.legend(handles=mem_legend_elements, loc="lower center", ncol=3,
                fontsize=9, frameon=True, fancybox=True,
                bbox_to_anchor=(0.5, -0.02))

    fig3.suptitle("Attention Breakdown by Memory Type (L=4096)", fontsize=13, y=1.01)
    fig3.tight_layout(rect=[0, 0.04, 1, 1])
    fig3.savefig(PLOT_DIR / "figure_memory_breakdown.png", dpi=200, bbox_inches="tight")
    print(f"Saved {PLOT_DIR / 'figure_memory_breakdown.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    if not (args.run or args.plot or args.all):
        args.all = True

    if args.run or args.all:
        run_experiments(args)

    rows = collect_results()
    write_summary_csv(DATA_DIR / "summary_memory_comparison.csv", rows)
    if args.plot or args.all:
        if not rows:
            raise SystemExit("No memory comparison data found.")
        plot_results(rows)


if __name__ == "__main__":
    main()
