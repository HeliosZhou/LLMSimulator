#!/usr/bin/env python3
"""Plot HBM4 baseline TPOT breakdown: FFN vs Attention (stacked bar chart)."""

import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data" / "sim_work"
PLOT_DIR = Path(__file__).resolve().parent / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024]
MODELS = ["deepseekV3", "grok1", "llama3_405B"]
MODEL_LABELS = {"deepseekV3": "DeepSeek-V3", "grok1": "Grok 1", "llama3_405B": "Llama 3-405B"}

# Colors
FFN_COLOR = "#538233"
ATTN_COLOR = "#E7A77C"


def load_data():
    data = {}
    for model in MODELS:
        data[model] = {"ffn": [], "attn": [], "tpot": []}
        for bs in BATCH_SIZES:
            model_dir = DATA_DIR / f"{model}_bs{bs}"
            found = False
            for csv_file in model_dir.glob("*.csv"):
                if "synthesis" in csv_file.name and csv_file.stat().st_size > 0:
                    with open(csv_file) as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            attn = float(row.get("attn_dram", 0))
                            fc = float(row.get("fc_dram", 0))
                            moe = float(row.get("moe_dram", 0))
                            total = attn + fc + moe
                            total_ns = float(row.get("time", 0))
                            tpot_ms = total_ns / 2 / 1e6
                            ffn_pct = (fc + moe) / total if total > 0 else 0
                            attn_pct = attn / total if total > 0 else 0
                            data[model]["ffn"].append(tpot_ms * ffn_pct)
                            data[model]["attn"].append(tpot_ms * attn_pct)
                            data[model]["tpot"].append(tpot_ms)
                    found = True
                    break
            if not found:
                data[model]["ffn"].append(0)
                data[model]["attn"].append(0)
                data[model]["tpot"].append(0)
    return data


def plot(data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        x = np.arange(len(BATCH_SIZES))
        width = 0.65

        ffn_vals = np.array(data[model]["ffn"])
        attn_vals = np.array(data[model]["attn"])

        ax.bar(x, ffn_vals, width, label="FFN", color=FFN_COLOR,
               edgecolor="white", linewidth=0.5)
        ax.bar(x, attn_vals, width, bottom=ffn_vals, label="Attention",
               color=ATTN_COLOR, edgecolor="white", linewidth=0.5)

        ax.set_xlabel("Batch Size", fontsize=10)
        if idx == 0:
            ax.set_ylabel("TPOT (ms)", fontsize=10)
        ax.set_title(MODEL_LABELS[model], fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in BATCH_SIZES], rotation=45, fontsize=8)
        ax.set_yscale("log")
        ax.set_ylim(0.5, 50)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(fontsize=9, loc="upper left")

    fig.suptitle("HBM4 Baseline TPOT seq_len = 8K",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = PLOT_DIR / "hbm4_tpot_breakdown.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    data = load_data()
    plot(data)
