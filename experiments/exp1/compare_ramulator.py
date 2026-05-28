#!/usr/bin/env python3
"""Compare ideal vs Ramulator results for exp1."""

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
IDEAL_DIR = EXP_DIR / "data"
RAMULATOR_DIR = EXP_DIR / "data_ramulator"
PLOT_DIR = EXP_DIR / "plots"


def collect_from(data_dir: Path, label: str) -> list[dict]:
    rows = []
    for csv_file in sorted(data_dir.glob("result_b*_l*_absorb_*.csv")):
        parts = csv_file.stem.split("_")
        try:
            batch = int(next(p[1:] for p in parts if p.startswith("b")))
            seq_len = int(next(p[1:] for p in parts if p.startswith("l")))
            absorb = parts[-1]
        except (StopIteration, ValueError):
            continue
        rows.append({
            "mode": label,
            "absorb": absorb,
            "batch_size": batch,
            "seq_len": seq_len,
            **attention_breakdown_from_csv(csv_file),
        })
    return rows


def plot_comparison(ideal_rows: list[dict], ramul_rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    key_fn = lambda r: (r["absorb"], r["batch_size"], r["seq_len"])
    ideal_map = {key_fn(r): r for r in ideal_rows}
    ramul_map = {key_fn(r): r for r in ramul_rows}

    configs = sorted(set(ideal_map.keys()) & set(ramul_map.keys()))
    if not configs:
        print("No matching configs found.")
        return

    categories = ["kv_decompress", "score_context", "out_proj", "etc"]
    cat_labels = ["KV decompress", "Score + Context", "Out projection", "Etc"]
    cat_colors = ["#d7191c", "#f6c744", "#a8cf8d", "#d9d9d9"]

    # --- Figure 1: Total latency comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, absorb in enumerate(["on", "off"]):
        ax = axes[idx]
        cfgs = [(b, s) for a, b, s in configs if a == absorb]
        cfgs.sort()
        labels = [f"B{b}\nL{s}" for b, s in cfgs]
        x = np.arange(len(cfgs))
        w = 0.35

        ideal_vals = [ideal_map[(absorb, b, s)]["total"] / 1e6 for b, s in cfgs]
        ramul_vals = [ramul_map[(absorb, b, s)]["total"] / 1e6 for b, s in cfgs]

        ax.bar(x - w / 2, ideal_vals, w, label="Ideal", color="#2196F3", edgecolor="black", lw=0.5)
        ax.bar(x + w / 2, ramul_vals, w, label="Ramulator", color="#FF5722", edgecolor="black", lw=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("Attention block latency (ms)")
        ax.set_title(f"Absorb {'ON' if absorb == 'on' else 'OFF'}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Show ratio
        for i, (iv, rv) in enumerate(zip(ideal_vals, ramul_vals)):
            if iv > 0:
                ratio = rv / iv
                ax.text(i, max(iv, rv) * 1.02, f"{ratio:.2f}x", ha="center", fontsize=7)

    fig.suptitle("Figure 6: Attention Latency — Ideal vs Ramulator", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figure6_ideal_vs_ramulator.png", dpi=200)
    print(f"Saved {PLOT_DIR / 'figure6_ideal_vs_ramulator.png'}")

    # --- Figure 2: Breakdown comparison per config ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for idx, absorb in enumerate(["on", "off"]):
        for col, mode_label, mode_map in [(0, "Ideal", ideal_map), (1, "Ramulator", ramul_map)]:
            ax = axes[idx][col]
            cfgs = [(b, s) for a, b, s in configs if a == absorb]
            cfgs.sort()
            labels = [f"B{b}/L{s}" for b, s in cfgs]
            x = np.arange(len(cfgs))
            w = 0.6

            for ci, (color, cat) in enumerate(zip(cat_colors, categories)):
                vals = [mode_map.get((absorb, b, s), {}).get(cat, 0) / 1e6 for b, s in cfgs]
                bottoms = [
                    sum(mode_map.get((absorb, b, s), {}).get(c, 0) / 1e6 for c in categories[:ci])
                    for b, s in cfgs
                ]
                ax.bar(x, vals, w, bottom=bottoms, color=color, edgecolor="black", lw=0.3, label=cat_labels[ci] if idx == 0 else "")

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=7, rotation=45)
            ax.set_ylabel("Latency (ms)")
            ax.set_title(f"Absorb {'ON' if absorb == 'on' else 'OFF'} — {mode_label}")
            ax.grid(axis="y", alpha=0.3)

    fig.legend(
        [plt.Rectangle((0, 0), 1, 1, color=c, ec="black", lw=0.5) for c in cat_colors],
        cat_labels, loc="upper center", ncol=4, frameon=False,
    )
    fig.suptitle("Attention Breakdown: Ideal vs Ramulator", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOT_DIR / "figure6_breakdown_comparison.png", dpi=200)
    print(f"Saved {PLOT_DIR / 'figure6_breakdown_comparison.png'}")

    # --- Figure 3: Overhead ratio heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for idx, absorb in enumerate(["on", "off"]):
        ax = axes[idx]
        batches = sorted(set(b for a, b, s in configs if a == absorb))
        seq_lens = sorted(set(s for a, b, s in configs if a == absorb))

        ratio_matrix = np.zeros((len(batches), len(seq_lens)))
        for bi, batch in enumerate(batches):
            for si, seq_len in enumerate(seq_lens):
                key = (absorb, batch, seq_len)
                if key in ideal_map and key in ramul_map:
                    iv = ideal_map[key]["total"]
                    rv = ramul_map[key]["total"]
                    ratio_matrix[bi, si] = rv / iv if iv > 0 else 0

        im = ax.imshow(ratio_matrix, cmap="YlOrRd", aspect="auto", vmin=1.0)
        ax.set_xticks(range(len(seq_lens)))
        ax.set_xticklabels(seq_lens)
        ax.set_yticks(range(len(batches)))
        ax.set_yticklabels(batches)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Batch per GPU")
        ax.set_title(f"Absorb {'ON' if absorb == 'on' else 'OFF'}")

        for bi in range(len(batches)):
            for si in range(len(seq_lens)):
                v = ratio_matrix[bi, si]
                if v > 0:
                    ax.text(si, bi, f"{v:.2f}x", ha="center", va="center", fontsize=8,
                            color="white" if v > 1.5 else "black")

    fig.colorbar(im, ax=axes, label="Ramulator / Ideal ratio")
    fig.suptitle("Ramulator Overhead Ratio (latency multiplier)", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figure6_overhead_heatmap.png", dpi=200)
    print(f"Saved {PLOT_DIR / 'figure6_overhead_heatmap.png'}")


def print_comparison_table(ideal_rows, ramul_rows):
    key_fn = lambda r: (r["absorb"], r["batch_size"], r["seq_len"])
    ideal_map = {key_fn(r): r for r in ideal_rows}
    ramul_map = {key_fn(r): r for r in ramul_rows}
    configs = sorted(set(ideal_map.keys()) & set(ramul_map.keys()))

    print(f"\n{'='*100}")
    print("IDEAL vs RAMULATOR COMPARISON")
    print(f"{'='*100}")
    print(f"{'Absorb':<8} {'B/GPU':<8} {'SeqLen':<8} {'Ideal (ms)':<14} {'Ramulator (ms)':<16} {'Ratio':<8} {'Overhead':<10}")
    print("-" * 100)

    for absorb, batch, seq_len in configs:
        iv = ideal_map[(absorb, batch, seq_len)]["total"] / 1e6
        rv = ramul_map[(absorb, batch, seq_len)]["total"] / 1e6
        ratio = rv / iv if iv > 0 else 0
        overhead = f"+{(ratio - 1) * 100:.1f}%"
        print(f"{absorb:<8} {batch:<8} {seq_len:<8} {iv:<14.2f} {rv:<16.2f} {ratio:<8.3f} {overhead:<10}")

    print(f"{'='*100}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--table", action="store_true", default=True)
    args = parser.parse_args()

    ideal_rows = collect_from(IDEAL_DIR, "ideal")
    ramul_rows = collect_from(RAMULATOR_DIR, "ramulator")

    if not ramul_rows:
        print("No Ramulator results found. Run the Ramulator experiment first.")
        return

    if args.table:
        print_comparison_table(ideal_rows, ramul_rows)

    if args.plot:
        plot_comparison(ideal_rows, ramul_rows)

    # Write combined summary
    all_rows = ideal_rows + ramul_rows
    write_summary_csv(EXP_DIR / "data" / "summary_comparison.csv", all_rows)
    print(f"\nSaved summary to {EXP_DIR / 'data' / 'summary_comparison.csv'}")


if __name__ == "__main__":
    main()
