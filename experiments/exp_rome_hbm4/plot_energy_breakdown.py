#!/usr/bin/env python3
"""Plot HBM4 baseline energy breakdown: CAS vs ACT vs Interposer (Figure 14 style)."""

import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data" / "sim_work"
PLOT_DIR = Path(__file__).resolve().parent / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Energy model parameters (Folded Banks Table 6)
E_DATA_MOVE_pJ_per_bit = 2.55
I_ACT_mA = 26
VDD = 1.2
T_RC_ns = 45
E_ACT_nJ = I_ACT_mA * 1e-3 * VDD * T_RC_ns  # = 1.404 nJ per activation

BS = 256
MODELS = ["deepseekV3", "grok1", "llama3_405B"]
MODEL_LABELS = {"deepseekV3": "DeepSeek-V3", "grok1": "Grok 1", "llama3_405B": "Llama 3-405B"}

# Colors
CAS_COLOR = "#538233"
ACT_COLOR = "#E7A77C"
INTERPOSER_COLOR = "#8B7355"


def compute_energy():
    results = {}
    for model in MODELS:
        model_dir = DATA_DIR / f"{model}_bs{BS}"
        for csv_file in model_dir.glob("*.csv"):
            if "synthesis" in csv_file.name and csv_file.stat().st_size > 0:
                with open(csv_file) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        read_count = float(row.get("read_count", 0))
                        write_count = float(row.get("write_count", 0))
                        act_count = float(row.get("act_count", 0))

                        total_bits = (read_count + write_count) * 256

                        # Energy in J (per decode iteration)
                        E_CAS_J = total_bits * E_DATA_MOVE_pJ_per_bit * 1e-12
                        E_ACT_J = act_count * E_ACT_nJ * 1e-9
                        E_TOTAL_J = E_CAS_J + E_ACT_J

                        results[model] = {
                            "CAS": E_CAS_J,
                            "ACT": E_ACT_J,
                            "Total": E_TOTAL_J,
                        }
                break
    return results


def plot(results):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(MODELS))
    width = 0.25

    cas = [results[m]["CAS"] for m in MODELS]
    act = [results[m]["ACT"] for m in MODELS]

    ax.bar(x - width / 2, cas, width, label="CAS (Data Movement)", color=CAS_COLOR, edgecolor="white")
    ax.bar(x + width / 2, act, width, label="ACT (Activation)", color=ACT_COLOR, edgecolor="white")

    ax.set_ylabel("Energy (J)", fontsize=11)
    ax.set_title(f"HBM4 Baseline Energy per Decode Iteration\n(batch = {BS}, seq_len = 8K)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    for i, model in enumerate(MODELS):
        t = results[model]["Total"]
        ax.text(i, max(cas[i], act[i]) * 1.15,
                f"{t:.2f}J", ha="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    out_path = PLOT_DIR / "hbm4_energy_breakdown.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def print_summary(results):
    print(f"\nEnergy summary (batch={BS}, per iteration):")
    for model in MODELS:
        r = results[model]
        print(f"  {MODEL_LABELS[model]:<15} CAS={r['CAS']:.2f}J  ACT={r['ACT']:.2f}J  "
              f"Total={r['Total']:.2f}J")


if __name__ == "__main__":
    results = compute_energy()
    plot(results)
    print_summary(results)
