#!/usr/bin/env python3
"""Analytical Figure 3 roofline reconstruction."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
PLOT_DIR = EXP_DIR / "plots"

PEAK_TFLOPS = 989.5
MEM_BW_GBPS = 4800.0
RIDGE = PEAK_TFLOPS * 1000.0 / MEM_BW_GBPS

POINTS = [
    {"name": "MHA core attention", "ari": 1.0, "kind": "attention"},
    {"name": "GQA core attention", "ari": 5.0, "kind": "attention"},
    {"name": "MLA core attention", "ari": 200.0, "kind": "attention"},
    {"name": "FFN B=64", "ari": 64.0, "kind": "ffn"},
    {"name": "FFN B=1K", "ari": 512.0, "kind": "ffn"},
    {"name": "MoE B=64", "ari": 16.0, "kind": "moe"},
    {"name": "MoE B=1K", "ari": 64.0, "kind": "moe"},
]


def performance(ari: float) -> float:
    return min(PEAK_TFLOPS, ari * MEM_BW_GBPS / 1000.0)


def write_summary(rows: list[dict[str, float | str]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / "summary_roofline.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "kind", "ari", "performance_tflops"])
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, float | str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    x = np.logspace(-1, 4, 300)
    y = np.minimum(PEAK_TFLOPS, x * MEM_BW_GBPS / 1000.0)

    colors = {"attention": "#1b9e77", "ffn": "#d95f02", "moe": "#7570b3"}
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, color="black", linewidth=1.8, label="Roofline")
    ax.axvline(RIDGE, color="gray", linestyle="--", linewidth=1.0, label=f"Ridge point = {RIDGE:.1f} Op/B")
    ax.axhline(PEAK_TFLOPS, color="gray", linestyle=":", linewidth=1.0)

    for row in rows:
        ari = float(row["ari"])
        perf = float(row["performance_tflops"])
        ax.scatter(ari, perf, s=70, color=colors[str(row["kind"])], edgecolor="black", zorder=3)
        ax.annotate(str(row["name"]), (ari, perf), fontsize=8, xytext=(5, 4), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOPs/Byte)")
    ax.set_ylabel("Performance (TFLOPS)")
    ax.set_title("Figure 3 style roofline analysis")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = PLOT_DIR / "figure3_roofline.png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if not (args.run or args.plot or args.all):
        args.all = True

    rows = [{**p, "performance_tflops": performance(float(p["ari"]))} for p in POINTS]
    write_summary(rows)
    if args.plot or args.run or args.all:
        plot(rows)


if __name__ == "__main__":
    main()
