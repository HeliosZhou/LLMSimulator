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
FIG12_TOTAL_BATCHES = [1152, 2304, 4608, 6912, 9216, 11520, 13824, 16128, 18432, 20736]
FIG13_BATCH_PER_GPU = [24, 48, 96, 144, 192, 240, 288, 336, 384, 408]
SYSTEMS = {
    "fig12_32gpu": {"label": "32 GPU", "num_node": 4, "num_device": 8, "deployment_count": 1, "batch_mode": "total"},
    "fig13_32gpu_x8": {"label": "32 GPU x8", "num_node": 4, "num_device": 8, "deployment_count": 8, "batch_mode": "per_gpu"},
    "fig13_256gpu": {"label": "256 GPU", "num_node": 32, "num_device": 8, "deployment_count": 1, "batch_mode": "per_gpu"},
}
LEGACY_SYSTEM_ALIASES = {
    "32gpu": "fig12_32gpu",
    "32gpu_x8": "fig13_32gpu_x8",
    "256gpu": "fig13_256gpu",
}


def result_name(system: str, skew: float, batch_value: int) -> str:
    suffix = "tb" if SYSTEMS[system]["batch_mode"] == "total" else "bpg"
    return f"result_system_{system}_s{int(skew * 10)}_{suffix}{batch_value}.csv"


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
    for path in sorted(DATA_DIR.glob("result_system_*_s*_*b*.csv")):
        stem = path.stem
        try:
            rest_stem = stem.replace("result_system_", "", 1)
            system, rest = rest_stem.rsplit("_s", 1)
            system = LEGACY_SYSTEM_ALIASES.get(system, system)
            if system not in SYSTEMS:
                continue
            if "_bpg" in rest:
                skew = int(rest.split("_bpg", 1)[0]) / 10.0
                batch_per_gpu = int(rest.split("_bpg", 1)[1])
                total_batch = batch_per_gpu * int(SYSTEMS[system]["num_node"]) * int(SYSTEMS[system]["num_device"])
            elif "_tb" in rest:
                skew = int(rest.split("_tb", 1)[0]) / 10.0
                total_batch = int(rest.split("_tb", 1)[1])
                batch_per_gpu = total_batch // (int(SYSTEMS[system]["num_node"]) * int(SYSTEMS[system]["num_device"]))
            else:
                skew = int(rest.split("_b", 1)[0]) / 10.0
                total_batch = int(rest.split("_b", 1)[1])
                batch_per_gpu = total_batch // (int(SYSTEMS[system]["num_node"]) * int(SYSTEMS[system]["num_device"]))
        except (IndexError, ValueError):
            continue
        grouped = system != "fig13_256gpu"
        summary = summarize_csv(path)
        deployment_count = int(SYSTEMS.get(system, {}).get("deployment_count", 1))
        summary["paper_system_throughput_tps"] = summary["throughput_tps"] * deployment_count
        rows.append(
            {
                "system": system,
                "system_label": SYSTEMS.get(system, {}).get("label", system),
                "skewness": skew,
                "batch_per_gpu": batch_per_gpu,
                "total_batch": total_batch,
                "deployment_count": deployment_count,
                "estimated_acc_load_imbalance": load_imbalance(256, 8, 32 if grouped else 256, skew, grouped),
                **summary,
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
        subset = sorted([r for r in rows if r["system"] == "fig12_32gpu" and r["skewness"] == skew], key=lambda r: int(r["total_batch"]))
        if subset:
            ax.plot([int(r["total_batch"]) for r in subset], [float(r["paper_system_throughput_tps"]) for r in subset], "o-", color=colors[skew], label=f"s={skew}")
    ax.set_title("Figure 12 style: 32 GPU skew sweep")
    ax.set_xlabel("Total batch")
    ax.set_ylabel("System throughput (tokens/s)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    for system in ["fig13_32gpu_x8", "fig13_256gpu"]:
        meta = SYSTEMS[system]
        xs = []
        ys = []
        labels = []
        for skew in SKEWS:
            subset = [r for r in rows if r["system"] == system and r["skewness"] == skew]
            if not subset:
                continue
            best = max(subset, key=lambda r: float(r["paper_system_throughput_tps"]))
            xs.append(skew)
            ys.append(float(best["paper_system_throughput_tps"]))
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
    systems = ["fig12_32gpu"] if args.quick else list(SYSTEMS)

    if args.run or args.all:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for system in systems:
            meta = SYSTEMS[system]
            batches = [1152] if args.quick and meta["batch_mode"] == "total" else [24] if args.quick else (FIG12_TOTAL_BATCHES if meta["batch_mode"] == "total" else FIG13_BATCH_PER_GPU)
            for skew in skews:
                for batch in batches:
                    batch_size = batch
                    if meta["batch_mode"] == "per_gpu":
                        batch_size = batch * int(meta["num_node"]) * int(meta["num_device"])
                    point = SimPoint(
                        batch_size=batch_size,
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
                    label = "total_B" if meta["batch_mode"] == "total" else "B/GPU"
                    print(f"[Figure 12/13] system={system} skew={skew} {label}={batch} sim_B={batch_size}")
                    run_simulation(point, DATA_DIR, result_name(system, skew, batch), args.timeout, not args.overwrite)

    rows = collect()
    write_summary_csv(DATA_DIR / "summary_skew.csv", rows)
    if args.plot or args.all:
        if not rows:
            raise SystemExit("No skew data found.")
        plot(rows)


if __name__ == "__main__":
    main()
