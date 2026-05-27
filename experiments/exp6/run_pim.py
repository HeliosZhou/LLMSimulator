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

SEQ_LENS = [1024, 4096, 16384]
BATCH_PER_GPU = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]
NUM_NODE = 4
NUM_DEVICE = 8
MODES = {
    "gpu": {"label": "GPU-only", "processor": "GPU", "low_moe": False},
    "gpu_pim": {"label": "GPU+PIM MoE", "processor": "GPU+PIM", "low_moe": True},
}


def result_name(mode: str, seq_len: int, batch_per_gpu: int) -> str:
    return f"result_mode_{mode}_l{seq_len}_bpg{batch_per_gpu}.csv"


def collect() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    paths = sorted(DATA_DIR.glob("result_mode_*_l*_bpg*.csv"))
    paths.extend(p for p in sorted(DATA_DIR.glob("result_mode_*_l*_b*.csv")) if "_bpg" not in p.stem)
    for path in paths:
        stem = path.stem
        try:
            mode = stem.split("_l")[0].replace("result_mode_", "")
            rest = stem.split("_l", 1)[1]
            marker = "_bpg" if "_bpg" in rest else "_b"
            seq_len = int(rest.split(marker, 1)[0])
            batch = int(rest.split(marker, 1)[1])
        except (IndexError, ValueError):
            continue
        rows.append({"mode": mode, "mode_label": MODES.get(mode, {}).get("label", mode), "seq_len": seq_len, "batch_per_gpu": batch, "total_batch": batch * NUM_NODE * NUM_DEVICE, **summarize_csv(path)})
    by_key = {(r["seq_len"], r["batch_per_gpu"], r["mode"]): r for r in rows}
    for r in rows:
        base = by_key.get((r["seq_len"], r["batch_per_gpu"], "gpu"))
        if base and float(base["throughput_tps"]):
            r["normalized_throughput"] = float(r["throughput_tps"]) / float(base["throughput_tps"])
        else:
            r["normalized_throughput"] = 0.0
    return rows


def plot(rows: list[dict[str, float | int | str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    available_seq_lens = sorted({int(r["seq_len"]) for r in rows if r["mode"] == "gpu_pim"})
    available_batches = sorted({int(r["batch_per_gpu"]) for r in rows if r["mode"] == "gpu_pim"})
    seq_lens = [v for v in SEQ_LENS if v in available_seq_lens] or available_seq_lens
    batches = [v for v in BATCH_PER_GPU if v in available_batches] or available_batches
    heat = np.full((len(seq_lens), len(batches)), np.nan)
    data = {(int(r["seq_len"]), int(r["batch_per_gpu"])): float(r["normalized_throughput"]) for r in rows if r["mode"] == "gpu_pim"}
    for i, seq_len in enumerate(seq_lens):
        for j, batch in enumerate(batches):
            heat[i, j] = data.get((seq_len, batch), np.nan)
    if np.isnan(heat).all():
        raise SystemExit("No GPU+PIM rows found for Figure 14 heatmap.")

    fig, ax = plt.subplots(figsize=(11, 3.8))
    im = ax.imshow(heat, aspect="auto", cmap="Greys", vmin=np.nanmin(heat), vmax=np.nanmax(heat))
    for i in range(len(seq_lens)):
        for j in range(len(batches)):
            if np.isfinite(heat[i, j]):
                ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black" if heat[i, j] < np.nanmean(heat) else "white")
    ax.set_xticks(range(len(batches)))
    ax.set_xticklabels([str(v) for v in batches], rotation=0, fontsize=8)
    ax.set_yticks(range(len(seq_lens)))
    ax.set_yticklabels([str(v) for v in seq_lens])
    ax.set_xlabel("Batch per GPU")
    ax.set_ylabel("Sequence length")
    ax.set_title("Figure 14 style: normalized GPU+PIM throughput")
    fig.colorbar(im, ax=ax, label="Normalized throughput")
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

    seq_lens = [1024] if args.quick else SEQ_LENS
    batches = [8, 32] if args.quick else BATCH_PER_GPU

    if args.run or args.all:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for mode, meta in MODES.items():
            for seq_len in seq_lens:
                for batch in batches:
                    point = SimPoint(
                        batch_size=batch * NUM_NODE * NUM_DEVICE,
                        seq_len=seq_len,
                        output_len=2,
                        num_node=NUM_NODE,
                        num_device=NUM_DEVICE,
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
                    print(f"[Figure 14] mode={mode} L={seq_len} B/GPU={batch} total_B={point.batch_size}")
                    run_simulation(point, DATA_DIR, result_name(mode, seq_len, batch), args.timeout, not args.overwrite)

    rows = collect()
    write_summary_csv(DATA_DIR / "summary_pim.csv", rows)
    if args.plot or args.all:
        if not rows:
            raise SystemExit("No PIM data found.")
        plot(rows)


if __name__ == "__main__":
    main()
