#!/usr/bin/env python3
"""Figure 9 style throughput-latency sweeps for GPT-3, Llama4-Maverick, DeepSeek-R1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.sim_utils import SimPoint, add_common_args, run_simulation, summarize_csv, write_summary_csv  # noqa: E402


EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
PLOT_DIR = EXP_DIR / "plots"

MODELS = {
    "gpt3_175B": {"label": "GPT-3", "ne_tp": 8, "expert_tp": 1, "absorb": False, "compressed_kv": False},
    "llama4_maverick": {"label": "Llama4-Maverick", "ne_tp": 8, "expert_tp": 1, "absorb": False, "compressed_kv": False},
    "deepseekV3": {"label": "DeepSeek-R1", "ne_tp": 1, "expert_tp": 1, "absorb": True, "compressed_kv": True},
}
SEQ_LENS = [2048, 8192]
TOTAL_BATCHES = {
    "gpt3_175B": list(range(96, 1729, 96)),
    "llama4_maverick": list(range(1152, 20737, 1152)),
    "deepseekV3": list(range(1152, 20737, 1152)),
}


def result_name(model: str, seq_len: int, total_batch: int) -> str:
    return f"result_model_{model}_l{seq_len}_tb{total_batch}.csv"


def collect() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    paths = sorted(DATA_DIR.glob("result_model_*_l*_tb*.csv"))
    paths.extend(p for p in sorted(DATA_DIR.glob("result_model_*_l*_b*.csv")) if "_tb" not in p.stem and "_bpg" not in p.stem)
    for path in paths:
        stem = path.stem
        try:
            rest_stem = stem.replace("result_model_", "", 1)
            model, rest = rest_stem.rsplit("_l", 1)
            marker = "_tb" if "_tb" in rest else "_b"
            seq_len = int(rest.split(marker, 1)[0])
            batch = int(rest.split(marker, 1)[1])
        except (IndexError, ValueError):
            continue
        rows.append({"model": model, "model_label": MODELS.get(model, {}).get("label", model), "seq_len": seq_len, "total_batch": batch, **summarize_csv(path)})
    return rows


def plot(rows: list[dict[str, float | int | str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(18, 5), sharey=False)
    if len(MODELS) == 1:
        axes = [axes]
    colors = {2048: "#1b9e77", 8192: "#d95f02"}

    for ax, (model, meta) in zip(axes, MODELS.items()):
        for seq_len in SEQ_LENS:
            subset = sorted(
                [r for r in rows if r["model"] == model and r["seq_len"] == seq_len],
                key=lambda r: int(r["total_batch"]),
            )
            if not subset:
                continue
            x = [float(r["latency_ns"]) / 1e6 for r in subset]
            y = [float(r["throughput_tps"]) for r in subset]
            labels = [str(int(r["total_batch"])) for r in subset]
            ax.plot(x, y, "o-", color=colors[seq_len], label=f"L={seq_len}")
            for lx, ly, label in zip(x, y, labels):
                ax.annotate(label, (lx, ly), fontsize=7, xytext=(2, 2), textcoords="offset points")
        ax.set_title(meta["label"])
        ax.set_xlabel("TPOT (ms)")
        ax.set_ylabel("System throughput (tokens/s)")
        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)

    fig.tight_layout()
    out = PLOT_DIR / "figure9_throughput_latency.png"
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

    if args.run or args.all:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for model in models:
            meta = MODELS[model]
            for seq_len in seq_lens:
                batches = TOTAL_BATCHES[model][:1] if args.quick else TOTAL_BATCHES[model]
                for batch in batches:
                    point = SimPoint(
                        model=model,
                        batch_size=batch,
                        seq_len=seq_len,
                        output_len=2,
                        num_node=4,
                        num_device=8,
                        none_expert_tp=meta["ne_tp"],
                        expert_tp=meta["expert_tp"],
                        use_absorb=meta["absorb"],
                        compressed_kv=meta["compressed_kv"],
                        precision_byte=2,
                    )
                    print(f"[Figure 9] model={model} L={seq_len} total_B={batch}")
                    run_simulation(point, DATA_DIR, result_name(model, seq_len, batch), args.timeout, not args.overwrite)

    rows = collect()
    write_summary_csv(DATA_DIR / "summary_throughput_latency.csv", rows)
    if args.plot or args.all:
        if not rows:
            raise SystemExit("No Figure 9 data found.")
        plot(rows)


if __name__ == "__main__":
    main()
