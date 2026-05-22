#!/usr/bin/env python3
"""Figure 14 style GPU vs GPU+PIM MoE execution comparison."""

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
BATCHES = [32, 64, 128, 256, 512]
MODES = {
    "gpu": {"label": "GPU-only", "processor": "GPU", "low_moe": False},
    "gpu_pim": {"label": "GPU+PIM MoE", "processor": "GPU+PIM", "low_moe": True},
}


def result_name(mode: str, seq_len: int, batch: int) -> str:
    return f"result_mode_{mode}_l{seq_len}_b{batch}.csv"


def collect() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for path in sorted(DATA_DIR.glob("result_mode_*_l*_b*.csv")):
        stem = path.stem
        try:
            mode = stem.split("_l")[0].replace("result_mode_", "")
            rest = stem.split("_l", 1)[1]
            seq_len = int(rest.split("_b", 1)[0])
            batch = int(rest.split("_b", 1)[1])
        except (IndexError, ValueError):
            continue
        rows.append({"mode": mode, "mode_label": MODES.get(mode, {}).get("label", mode), "seq_len": seq_len, "batch_size": batch, **summarize_csv(path)})
    by_key = {(r["seq_len"], r["batch_size"], r["mode"]): r for r in rows}
    for r in rows:
        base = by_key.get((r["seq_len"], r["batch_size"], "gpu"))
        if base and float(base["throughput_tps"]):
            r["normalized_throughput"] = float(r["throughput_tps"]) / float(base["throughput_tps"])
        else:
            r["normalized_throughput"] = 0.0
    return rows


def plot(rows: list[dict[str, float | int | str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(SEQ_LENS), figsize=(12, 5), sharey=True)
    for ax, seq_len in zip(axes, SEQ_LENS):
        subset = sorted([r for r in rows if r["mode"] == "gpu_pim" and r["seq_len"] == seq_len], key=lambda r: int(r["batch_size"]))
        if subset:
            ax.plot([int(r["batch_size"]) for r in subset], [float(r["normalized_throughput"]) for r in subset], "o-", color="#1b9e77")
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"L={seq_len}")
        ax.set_xlabel("Batch per system")
        ax.set_ylabel("Normalized throughput vs GPU-only")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = PLOT_DIR / "figure14_pim.png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    if not (args.run or args.plot or args.all):
        args.all = True

    seq_lens = [2048] if args.quick else SEQ_LENS
    batches = [32, 64] if args.quick else BATCHES

    if args.run or args.all:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for mode, meta in MODES.items():
            for seq_len in seq_lens:
                for batch in batches:
                    point = SimPoint(
                        batch_size=batch,
                        seq_len=seq_len,
                        output_len=2,
                        num_node=4,
                        num_device=8,
                        processor_type=meta["processor"],
                        use_low_unit_moe_only=meta["low_moe"],
                        none_expert_tp=1,
                        expert_tp=1,
                        use_absorb=True,
                        compressed_kv=True,
                        precision_byte=2,
                        pim_x=4,
                        pim_op_b=8,
                    )
                    print(f"[Figure 14] mode={mode} L={seq_len} B={batch}")
                    run_simulation(point, DATA_DIR, result_name(mode, seq_len, batch), args.timeout, not args.overwrite)

    rows = collect()
    write_summary_csv(DATA_DIR / "summary_pim.csv", rows)
    if args.plot or args.all:
        if not rows:
            raise SystemExit("No PIM data found.")
        plot(rows)


if __name__ == "__main__":
    main()
