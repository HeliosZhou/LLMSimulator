#!/usr/bin/env python3
"""Figure 6: MLA attention latency breakdown with/without reordering."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.sim_utils import (  # noqa: E402
    SimPoint,
    add_common_args,
    attention_breakdown_from_csv,
    run_simulation,
    write_summary_csv,
)


EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
PLOT_DIR = EXP_DIR / "plots"

BATCH_SIZES = [32, 64, 128, 256]
SEQ_LENGTHS = [2048, 4096, 8192]


def collect_results() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for csv_file in sorted(DATA_DIR.glob("result_b*_l*_absorb_*.csv")):
        parts = csv_file.stem.split("_")
        try:
            batch = int(next(p[1:] for p in parts if p.startswith("b")))
            seq_len = int(next(p[1:] for p in parts if p.startswith("l")))
            absorb = parts[-1]
        except (StopIteration, ValueError):
            continue
        rows.append(
            {
                "absorb": absorb,
                "batch_size": batch,
                "seq_len": seq_len,
                **attention_breakdown_from_csv(csv_file),
            }
        )
    return rows


def plot_results(rows: list[dict[str, float | int | str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    data = {(r["absorb"], r["seq_len"], r["batch_size"]): r for r in rows}
    categories = ["kv_decompress", "score_context", "out_proj", "etc"]
    cat_labels = ["KV decompress", "Score + Context", "Out projection", "Etc"]
    cat_colors = ["#d95f02", "#1b9e77", "#7570b3", "#8c8c8c"]

    x = np.arange(len(SEQ_LENGTHS))
    width = 0.8 / len(BATCH_SIZES)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    for i, seq_len in enumerate(SEQ_LENGTHS):
        for j, batch in enumerate(BATCH_SIZES):
            on_key = ("on", seq_len, batch)
            off_key = ("off", seq_len, batch)
            if on_key in data and off_key in data and data[on_key]["total"]:
                ratio = float(data[off_key]["total"]) / float(data[on_key]["total"])
                ax.bar(
                    x[i] + (j - len(BATCH_SIZES) / 2 + 0.5) * width,
                    ratio,
                    width * 0.85,
                    color=plt.cm.Blues(0.35 + j * 0.14),
                    edgecolor="black",
                    linewidth=0.4,
                    label=f"B={batch}" if i == 0 else "",
                )
    ax.set_yscale("log")
    ax.set_ylabel("Normalized latency\n(w/o reordering / w/ reordering)")
    ax.set_xlabel("Sequence length")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in SEQ_LENGTHS])
    ax.set_title("(a) Attention block speedup")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    for panel, absorb, title in [(1, "off", "(b) w/o reordering"), (2, "on", "(c) w/ reordering")]:
        ax = axes[panel]
        for i, seq_len in enumerate(SEQ_LENGTHS):
            for j, batch in enumerate(BATCH_SIZES):
                key = (absorb, seq_len, batch)
                if key not in data:
                    continue
                bottom = 0.0
                bar_x = x[i] + (j - len(BATCH_SIZES) / 2 + 0.5) * width
                for color, cat in zip(cat_colors, categories):
                    val = float(data[key][cat]) / 1e6
                    ax.bar(bar_x, val, width * 0.85, bottom=bottom, color=color, edgecolor="black", linewidth=0.3)
                    bottom += val
        ax.set_ylabel("Attention block latency (ms)")
        ax.set_xlabel("Sequence length")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in SEQ_LENGTHS])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    patches = [plt.Rectangle((0, 0), 1, 1, color=c, ec="black", lw=0.3) for c in cat_colors]
    fig.legend(patches, cat_labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(PLOT_DIR / "figure6_attention_breakdown.png", dpi=200)
    print(f"Saved {PLOT_DIR / 'figure6_attention_breakdown.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    if not (args.run or args.plot or args.all):
        args.all = True

    batches = [32] if args.quick else BATCH_SIZES
    seq_lens = [2048] if args.quick else SEQ_LENGTHS

    if args.run or args.all:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for absorb_label, use_absorb in [("on", True), ("off", False)]:
            for seq_len in seq_lens:
                for batch in batches:
                    name = f"result_b{batch}_l{seq_len}_absorb_{absorb_label}.csv"
                    point = SimPoint(batch_size=batch, seq_len=seq_len, use_absorb=use_absorb)
                    print(f"[Figure 6] absorb={absorb_label} B={batch} L={seq_len}")
                    if run_simulation(point, DATA_DIR, name, timeout=args.timeout, skip_existing=not args.overwrite) is None:
                        print(f"  skipped: {name}")

    rows = collect_results()
    write_summary_csv(DATA_DIR / "summary_attention_breakdown.csv", rows)
    if args.plot or args.all:
        if not rows:
            raise SystemExit("No Figure 6 data found.")
        plot_results(rows)


if __name__ == "__main__":
    main()
