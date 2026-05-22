#!/usr/bin/env python3
"""Figure 12/13 style skewed expert-routing experiments."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.sim_utils import SimPoint, add_common_args, run_simulation, summarize_csv, write_summary_csv  # noqa: E402


EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
PLOT_DIR = EXP_DIR / "plots"

SKEWS = [0.0, 0.2, 0.4, 0.6, 0.8]
BATCHES = [32, 64, 128, 256, 512, 1024]
SYSTEMS = {
    "32gpu_x8": {"label": "32 GPU x8", "num_node": 4, "num_device": 8},
    "256gpu": {"label": "256 GPU", "num_node": 32, "num_device": 8},
}


def result_name(system: str, skew: float, batch: int) -> str:
    return f"result_system_{system}_s{int(skew * 10)}_b{batch}.csv"


def load_imbalance(num_expert: int, top_k: int, num_acc: int, skew: float, grouped: bool) -> float:
    weights = [1.0 / ((i + 1) ** skew) for i in range(num_expert)]
    total = sum(weights)
    probs = [w / total for w in weights]
    tokens_per_expert = [p * top_k for p in probs]
    if grouped:
        experts_per_acc = num_expert // num_acc
        loads = [sum(tokens_per_expert[i * experts_per_acc : (i + 1) * experts_per_acc]) for i in range(num_acc)]
    else:
        loads = tokens_per_expert[:num_acc]
    ideal = sum(loads) / len(loads)
    return max(loads) / ideal if ideal else math.nan


def collect() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for path in sorted(DATA_DIR.glob("result_system_*_s*_b*.csv")):
        stem = path.stem
        try:
            rest_stem = stem.replace("result_system_", "", 1)
            system, rest = rest_stem.rsplit("_s", 1)
            skew = int(rest.split("_b", 1)[0]) / 10.0
            batch = int(rest.split("_b", 1)[1])
        except (IndexError, ValueError):
            continue
        grouped = system == "32gpu_x8"
        rows.append(
            {
                "system": system,
                "system_label": SYSTEMS.get(system, {}).get("label", system),
                "skewness": skew,
                "batch_size": batch,
                "estimated_acc_load_imbalance": load_imbalance(256, 8, 32 if grouped else 256, skew, grouped),
                **summarize_csv(path),
            }
        )
    return rows


def plot(rows: list[dict[str, float | int | str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {0.0: "#1b9e77", 0.2: "#d95f02", 0.4: "#7570b3", 0.6: "#e7298a", 0.8: "#666666"}
    ax = axes[0]
    for skew in SKEWS:
        subset = sorted([r for r in rows if r["system"] == "32gpu_x8" and r["skewness"] == skew], key=lambda r: int(r["batch_size"]))
        if subset:
            ax.plot([int(r["batch_size"]) for r in subset], [float(r["throughput_tps"]) for r in subset], "o-", color=colors[skew], label=f"s={skew}")
    ax.set_title("Figure 12 style: 32 GPU skew sweep")
    ax.set_xlabel("Batch per system")
    ax.set_ylabel("System throughput (tokens/s)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    for system, meta in SYSTEMS.items():
        xs = []
        ys = []
        labels = []
        for skew in SKEWS:
            subset = [r for r in rows if r["system"] == system and r["skewness"] == skew]
            if not subset:
                continue
            best = max(subset, key=lambda r: float(r["throughput_tps"]))
            xs.append(skew)
            ys.append(float(best["throughput_tps"]))
            labels.append(float(best["estimated_acc_load_imbalance"]))
        ax.plot(xs, ys, "o-", label=meta["label"])
        for x, y, imb in zip(xs, ys, labels):
            ax.annotate(f"{imb:.2f}x", (x, y), fontsize=7, xytext=(2, 2), textcoords="offset points")
    ax.set_title("Figure 13 style: scale vs skew")
    ax.set_xlabel("Skewness")
    ax.set_ylabel("Best observed throughput (tokens/s)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = PLOT_DIR / "figure12_13_skew.png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    if not (args.run or args.plot or args.all):
        args.all = True

    skews = [0.0, 0.8] if args.quick else SKEWS
    batches = [32, 64] if args.quick else BATCHES
    systems = ["32gpu_x8"] if args.quick else list(SYSTEMS)

    if args.run or args.all:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for system in systems:
            meta = SYSTEMS[system]
            for skew in skews:
                for batch in batches:
                    point = SimPoint(
                        batch_size=batch,
                        seq_len=2048,
                        output_len=2,
                        num_node=meta["num_node"],
                        num_device=meta["num_device"],
                        infiniband_gen=7200,
                        skewness=skew,
                        none_expert_tp=1,
                        expert_tp=1,
                        use_absorb=True,
                        compressed_kv=True,
                        precision_byte=2,
                    )
                    print(f"[Figure 12/13] system={system} skew={skew} B={batch}")
                    run_simulation(point, DATA_DIR, result_name(system, skew, batch), args.timeout, not args.overwrite)

    rows = collect()
    write_summary_csv(DATA_DIR / "summary_skew.csv", rows)
    if args.plot or args.all:
        if not rows:
            raise SystemExit("No skew data found.")
        plot(rows)


if __name__ == "__main__":
    main()
