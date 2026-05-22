#!/usr/bin/env python3
"""Analytical Figure 4 memory-footprint reconstruction."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
PLOT_DIR = EXP_DIR / "plots"

TOKENS = 8_000_000
BYTES_PER_PARAM = 2

MODELS = [
    {"model": "GPT-3", "total_params_b": 175.0, "activated_params_b": 175.0, "kv_per_token_bytes": 4.5 * 1024 * 1024},
    {"model": "Llama4-Maverick", "total_params_b": 400.0, "activated_params_b": 17.0, "kv_per_token_bytes": 192.0 * 1024},
    {"model": "DeepSeek-R1", "total_params_b": 671.0, "activated_params_b": 37.0, "kv_per_token_bytes": 68.6 * 1024},
]


def build_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for item in MODELS:
        total_weight_gb = float(item["total_params_b"]) * 1e9 * BYTES_PER_PARAM / 1e9
        activated_weight_gb = float(item["activated_params_b"]) * 1e9 * BYTES_PER_PARAM / 1e9
        kv_total_gb = float(item["kv_per_token_bytes"]) * TOKENS / 1e9
        rows.append(
            {
                "model": str(item["model"]),
                "total_params_b": float(item["total_params_b"]),
                "activated_params_b": float(item["activated_params_b"]),
                "activated_weight_gb": activated_weight_gb,
                "total_weight_gb": total_weight_gb,
                "kv_per_token_kb": float(item["kv_per_token_bytes"]) / 1024.0,
                "kv_for_8m_tokens_gb": kv_total_gb,
                "total_memory_gb": total_weight_gb + kv_total_gb,
            }
        )
    return rows


def write_summary(rows: list[dict[str, float | str]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / "summary_memory_footprint.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, float | str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    labels = [str(r["model"]) for r in rows]
    x = np.arange(len(labels))
    width = 0.36

    activated = [float(r["activated_weight_gb"]) for r in rows]
    total_weight = [float(r["total_weight_gb"]) for r in rows]
    kv = [float(r["kv_for_8m_tokens_gb"]) for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, activated, width, label="Activated parameters", color="#1b9e77", edgecolor="black", linewidth=0.4)
    ax.bar(x + width / 2, total_weight, width, label="Total weights", color="#7570b3", edgecolor="black", linewidth=0.4)
    ax.bar(x + width / 2, kv, width, bottom=total_weight, label="KV cache for 8M tokens", color="#d95f02", edgecolor="black", linewidth=0.4)

    for i, row in enumerate(rows):
        ax.annotate(
            f"{float(row['kv_per_token_kb']):.1f}KB/token",
            (x[i] + width / 2, total_weight[i] + kv[i]),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

    ax.set_yscale("log")
    ax.set_ylabel("Memory footprint (GB, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Figure 4 style memory footprint comparison")
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = PLOT_DIR / "figure4_memory_footprint.png"
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

    rows = build_rows()
    write_summary(rows)
    if args.plot or args.run or args.all:
        plot(rows)


if __name__ == "__main__":
    main()
