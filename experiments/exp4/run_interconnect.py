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
BATCHES = [32, 64, 128, 256, 512, 1024]
SYSTEMS = {
    "32gpu_xdr": {"label": "32 GPU, 100GB/s", "num_node": 4, "num_device": 8, "ib": 800},
    "32gpu_nvlink": {"label": "32 GPU, 900GB/s", "num_node": 4, "num_device": 8, "ib": 7200},
    "256gpu_100": {"label": "256 GPU, 100GB/s", "num_node": 32, "num_device": 8, "ib": 800},
    "256gpu_300": {"label": "256 GPU, 300GB/s", "num_node": 32, "num_device": 8, "ib": 2400},
    "256gpu_900": {"label": "256 GPU, 900GB/s", "num_node": 32, "num_device": 8, "ib": 7200},
}


def result_name(system: str, seq_len: int, batch: int) -> str:
    return f"result_system_{system}_l{seq_len}_b{batch}.csv"


def collect() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for path in sorted(DATA_DIR.glob("result_system_*_l*_b*.csv")):
        stem = path.stem
        try:
            system = stem.split("_l")[0].replace("result_system_", "")
            rest = stem.split("_l", 1)[1]
            seq_len = int(rest.split("_b", 1)[0])
            batch = int(rest.split("_b", 1)[1])
        except (IndexError, ValueError):
            continue
        rows.append({"system": system, "system_label": SYSTEMS.get(system, {}).get("label", system), "seq_len": seq_len, "batch_size": batch, **summarize_csv(path)})
    return rows


def plot(rows: list[dict[str, float | int | str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(SEQ_LENS), figsize=(14, 5), sharey=True)
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#666666"]

    for ax, seq_len in zip(axes, SEQ_LENS):
        for color, (system, meta) in zip(colors, SYSTEMS.items()):
            subset = sorted([r for r in rows if r["system"] == system and r["seq_len"] == seq_len], key=lambda r: int(r["batch_size"]))
            if not subset:
                continue
            x = [int(r["batch_size"]) for r in subset]
            y = [float(r["throughput_tps"]) for r in subset]
            ax.plot(x, y, "o-", color=color, label=meta["label"])
        ax.set_title(f"L={seq_len}")
        ax.set_xlabel("Batch per system")
        ax.set_ylabel("System throughput (tokens/s)")
        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)

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

    systems = ["32gpu_xdr", "32gpu_nvlink"] if args.quick else list(SYSTEMS)
    seq_lens = [2048] if args.quick else SEQ_LENS
    batches = [32, 64] if args.quick else BATCHES

    if args.run or args.all:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for system in systems:
            meta = SYSTEMS[system]
            for seq_len in seq_lens:
                for batch in batches:
                    point = SimPoint(
                        batch_size=batch,
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
                    print(f"[Figure 10/11] system={system} L={seq_len} B={batch}")
                    run_simulation(point, DATA_DIR, result_name(system, seq_len, batch), args.timeout, not args.overwrite)

    rows = collect()
    write_summary_csv(DATA_DIR / "summary_interconnect.csv", rows)
    if args.plot or args.all:
        if not rows:
            raise SystemExit("No interconnect data found.")
        plot(rows)


if __name__ == "__main__":
    main()
