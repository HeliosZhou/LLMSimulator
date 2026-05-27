#!/usr/bin/env python3
"""Figure 10/11 style interconnect sensitivity for DeepSeek-R1 decode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.sim_utils import SimPoint, add_common_args, run_simulation, summarize_csv, write_summary_csv  # noqa: E402


EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
PLOT_DIR = EXP_DIR / "plots"

SEQ_LENS = [2048, 8192]
FIG11_SEQ_LENS = [2048, 16384]
FIG10_TOTAL_BATCHES = list(range(1152, 20737, 1152))
FIG11_BATCH_PER_GPU = list(range(12, 421, 12))
SYSTEMS = {
    "fig10_32gpu_xdr": {
        "label": "32 GPU, 100GB/s",
        "num_node": 4,
        "num_device": 8,
        "ib": 800,
        "deployment_count": 1,
        "batch_mode": "total",
    },
    "fig11_32gpu_x8_900": {
        "label": "32 GPU x8, 900GB/s",
        "num_node": 4,
        "num_device": 8,
        "ib": 7200,
        "deployment_count": 8,
        "batch_mode": "per_gpu",
    },
    "fig11_256gpu_100": {
        "label": "256 GPU, 100GB/s",
        "num_node": 32,
        "num_device": 8,
        "ib": 800,
        "deployment_count": 1,
        "batch_mode": "per_gpu",
    },
    "fig11_256gpu_300": {
        "label": "256 GPU, 300GB/s",
        "num_node": 32,
        "num_device": 8,
        "ib": 2400,
        "deployment_count": 1,
        "batch_mode": "per_gpu",
    },
    "fig11_256gpu_900": {
        "label": "256 GPU, 900GB/s",
        "num_node": 32,
        "num_device": 8,
        "ib": 7200,
        "deployment_count": 1,
        "batch_mode": "per_gpu",
    },
}
LEGACY_SYSTEM_ALIASES = {
    "32gpu_xdr": "fig10_32gpu_xdr",
    "32gpu_nvlink": "fig11_32gpu_x8_900",
    "256gpu_100": "fig11_256gpu_100",
    "256gpu_300": "fig11_256gpu_300",
    "256gpu_900": "fig11_256gpu_900",
}


def result_name(system: str, seq_len: int, batch_value: int) -> str:
    meta = SYSTEMS[system]
    suffix = "tb" if meta["batch_mode"] == "total" else "bpg"
    return f"result_system_{system}_l{seq_len}_{suffix}{batch_value}.csv"


def collect() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for path in sorted(DATA_DIR.glob("result_system_*_l*_*b*.csv")):
        stem = path.stem
        try:
            system = stem.split("_l")[0].replace("result_system_", "")
            system = LEGACY_SYSTEM_ALIASES.get(system, system)
            if system not in SYSTEMS:
                continue
            rest = stem.split("_l", 1)[1]
            if "_bpg" in rest:
                seq_len = int(rest.split("_bpg", 1)[0])
                batch_per_gpu = int(rest.split("_bpg", 1)[1])
                total_batch = batch_per_gpu * int(SYSTEMS[system]["num_node"]) * int(SYSTEMS[system]["num_device"])
            elif "_tb" in rest:
                seq_len = int(rest.split("_tb", 1)[0])
                total_batch = int(rest.split("_tb", 1)[1])
                batch_per_gpu = total_batch // (int(SYSTEMS[system]["num_node"]) * int(SYSTEMS[system]["num_device"]))
            else:
                seq_len = int(rest.split("_b", 1)[0])
                total_batch = int(rest.split("_b", 1)[1])
                batch_per_gpu = total_batch // (int(SYSTEMS[system]["num_node"]) * int(SYSTEMS[system]["num_device"]))
        except (IndexError, ValueError):
            continue
        summary = summarize_csv(path)
        deployment_count = int(SYSTEMS.get(system, {}).get("deployment_count", 1))
        summary["paper_system_throughput_tps"] = summary["throughput_tps"] * deployment_count
        rows.append(
            {
                "system": system,
                "system_label": SYSTEMS.get(system, {}).get("label", system),
                "seq_len": seq_len,
                "batch_per_gpu": batch_per_gpu,
                "total_batch": total_batch,
                "deployment_count": deployment_count,
                **summary,
            }
        )
    return rows


def plot(rows: list[dict[str, float | int | str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

    ax = axes[0]
    for seq_len, color in zip(SEQ_LENS, colors):
        subset = sorted([r for r in rows if r["system"] == "fig10_32gpu_xdr" and r["seq_len"] == seq_len], key=lambda r: int(r["total_batch"]))
        if subset:
            ax.plot([int(r["total_batch"]) for r in subset], [float(r["paper_system_throughput_tps"]) for r in subset], "o-", color=color, label=f"L={seq_len}")
    ax.set_title("Figure 10: 32 GPU XDR")
    ax.set_xlabel("Total batch")
    ax.set_ylabel("System throughput (tokens/s)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    fig11_systems = ["fig11_32gpu_x8_900", "fig11_256gpu_900", "fig11_256gpu_300", "fig11_256gpu_100"]
    for color, system in zip(colors, fig11_systems):
        meta = SYSTEMS[system]
        for seq_len, marker in zip(FIG11_SEQ_LENS, ["o", "s"]):
            subset = sorted([r for r in rows if r["system"] == system and r["seq_len"] == seq_len], key=lambda r: int(r["batch_per_gpu"]))
            if not subset:
                continue
            x = [int(r["batch_per_gpu"]) for r in subset]
            y = [float(r["paper_system_throughput_tps"]) for r in subset]
            ax.plot(x, y, marker + "-", color=color, label=f"{meta['label']}, L={seq_len}")
    ax.set_title("Figure 11: deployment granularity")
    ax.set_xlabel("Batch per GPU")
    ax.set_ylabel("System throughput (tokens/s)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)

    fig.tight_layout()
    out = PLOT_DIR / "figure10_11_interconnect.png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    if not (args.run or args.plot or args.all):
        args.all = True

    systems = ["fig10_32gpu_xdr", "fig11_32gpu_x8_900"] if args.quick else list(SYSTEMS)

    if args.run or args.all:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for system in systems:
            meta = SYSTEMS[system]
            seq_lens = [2048] if args.quick else (SEQ_LENS if meta["batch_mode"] == "total" else FIG11_SEQ_LENS)
            batches = [1152] if args.quick and meta["batch_mode"] == "total" else [12] if args.quick else (FIG10_TOTAL_BATCHES if meta["batch_mode"] == "total" else FIG11_BATCH_PER_GPU)
            for seq_len in seq_lens:
                for batch in batches:
                    batch_size = batch
                    if meta["batch_mode"] == "per_gpu":
                        batch_size = batch * int(meta["num_node"]) * int(meta["num_device"])
                    point = SimPoint(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        output_len=2,
                        num_node=meta["num_node"],
                        num_device=meta["num_device"],
                        infiniband_gen=meta["ib"],
                        none_expert_tp=1,
                        expert_tp=1,
                        use_absorb=True,
                        compressed_kv=True,
                        precision_byte=2,
                    )
                    label = "total_B" if meta["batch_mode"] == "total" else "B/GPU"
                    print(f"[Figure 10/11] system={system} L={seq_len} {label}={batch} sim_B={batch_size}")
                    run_simulation(point, DATA_DIR, result_name(system, seq_len, batch), args.timeout, not args.overwrite)

    rows = collect()
    write_summary_csv(DATA_DIR / "summary_interconnect.csv", rows)
    if args.plot or args.all:
        if not rows:
            raise SystemExit("No interconnect data found.")
        plot(rows)


if __name__ == "__main__":
    main()
