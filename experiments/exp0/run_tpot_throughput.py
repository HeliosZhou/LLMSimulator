#!/usr/bin/env python3
"""Figure 2 style TPOT and per-GPU throughput sweep."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.sim_utils import DEFAULT_NUM_DEVICE, DEFAULT_NUM_GPUS, DEFAULT_NUM_NODE, SimPoint, add_common_args, run_simulation, summarize_csv, system_batch, write_summary_csv  # noqa: E402


EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
PLOT_DIR = EXP_DIR / "plots"

MODELS = {
    "gpt3_175B": {"label": "GPT-3", "ne_tp": 8, "expert_tp": 1, "absorb": False, "compressed_kv": False},
    "llama4_maverick": {"label": "Llama4-Maverick", "ne_tp": 8, "expert_tp": 1, "absorb": False, "compressed_kv": False},
    "deepseekV3": {"label": "DeepSeek-R1", "ne_tp": 1, "expert_tp": 1, "absorb": True, "compressed_kv": True},
}
SEQ_LENS = [2048, 8192]
BATCH_PER_GPU = {
    "gpt3_175B": [1, 4, 16, 64, 256],
    "llama4_maverick": [32, 64, 128, 256],
    "deepseekV3": [32, 64, 128, 256],
}


def result_name(model: str, seq_len: int, batch_per_gpu: int) -> str:
    return f"result_model_{model}_l{seq_len}_bpg{batch_per_gpu}.csv"


def collect() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for path in sorted(DATA_DIR.glob("result_model_*_l*_bpg*.csv")):
        try:
            stem = path.stem.replace("result_model_", "", 1)
            model, rest = stem.rsplit("_l", 1)
            seq_len = int(rest.split("_bpg", 1)[0])
            batch_per_gpu = int(rest.split("_bpg", 1)[1])
        except (IndexError, ValueError):
            continue
        summary = summarize_csv(path)
        summary["per_gpu_throughput_tps"] = summary["throughput_tps"] / DEFAULT_NUM_GPUS
        rows.append(
            {
                "model": model,
                "model_label": MODELS.get(model, {}).get("label", model),
                "seq_len": seq_len,
                "batch_per_gpu": batch_per_gpu,
                "total_batch": batch_per_gpu * DEFAULT_NUM_GPUS,
                **summary,
            }
        )
    return rows


def plot(rows: list[dict[str, float | int | str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(MODELS), figsize=(18, 8), sharex=False)
    colors = {2048: "#1b9e77", 8192: "#d95f02"}

    for col, (model, meta) in enumerate(MODELS.items()):
        for seq_len in SEQ_LENS:
            subset = sorted(
                [r for r in rows if r["model"] == model and r["seq_len"] == seq_len],
                key=lambda r: int(r["batch_per_gpu"]),
            )
            if not subset:
                continue
            x = [int(r["batch_per_gpu"]) for r in subset]
            tpot = [float(r["latency_ns"]) / 1e6 for r in subset]
            per_gpu = [float(r["per_gpu_throughput_tps"]) for r in subset]
            axes[0][col].plot(x, tpot, "o-", color=colors[seq_len], label=f"L={seq_len}")
            axes[1][col].plot(x, per_gpu, "o-", color=colors[seq_len], label=f"L={seq_len}")

        axes[0][col].set_title(meta["label"])
        axes[0][col].set_ylabel("TPOT (ms)")
        axes[1][col].set_ylabel("Per-GPU throughput (tokens/s)")
        axes[1][col].set_xlabel("Batch per GPU")
        for row in axes:
            row[col].grid(True, alpha=0.25)
            handles, _ = row[col].get_legend_handles_labels()
            if handles:
                row[col].legend(fontsize=8)

    fig.tight_layout()
    out = PLOT_DIR / "figure2_tpot_throughput.png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    if not (args.run or args.plot or args.all):
        args.all = True

    models = list(MODELS)
    seq_lens = [2048] if args.quick else SEQ_LENS
    quick_batches = {model: values[:1] for model, values in BATCH_PER_GPU.items()}

    if args.run or args.all:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for model in models:
            meta = MODELS[model]
            for seq_len in seq_lens:
                batches = quick_batches[model] if args.quick else BATCH_PER_GPU[model]
                for batch_per_gpu in batches:
                    point = SimPoint(
                        model=model,
                        batch_size=system_batch(batch_per_gpu),
                        seq_len=seq_len,
                        output_len=2,
                        num_node=DEFAULT_NUM_NODE,
                        num_device=DEFAULT_NUM_DEVICE,
                        none_expert_tp=meta["ne_tp"],
                        expert_tp=meta["expert_tp"],
                        use_absorb=meta["absorb"],
                        compressed_kv=meta["compressed_kv"],
                        precision_byte=2,
                    )
                    print(f"[Figure 2] model={model} L={seq_len} B/GPU={batch_per_gpu} total_B={point.batch_size}")
                    run_simulation(point, DATA_DIR, result_name(model, seq_len, batch_per_gpu), args.timeout, not args.overwrite)

    rows = collect()
    write_summary_csv(DATA_DIR / "summary_tpot_throughput.csv", rows)
    if args.plot or args.all:
        if not rows:
            raise SystemExit("No Figure 2 data found.")
        plot(rows)


if __name__ == "__main__":
    main()
