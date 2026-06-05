#!/usr/bin/env python3
"""Analyze HBM3E Ramulator hierarchy experiment results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
PLOT_DIR = EXP_DIR / "plots"
REPORT_PATH = EXP_DIR / "HBM3E_ANALYSIS_REPORT.md"

BATCH_PER_GPU = [32, 64, 128, 256]
SEQ_LENGTHS = [2048, 4096, 8192]
REORDERING_MODES = ["on", "off"]
RAMULATOR_MODES = ["on", "off"]

COUNT_FIELDS = [
    "act_count",
    "read_count",
    "write_count",
    "all_act_count",
    "all_read_count",
    "all_write_count",
    "ref_count",
]
TIME_FIELDS = ["memory_duration", "background_time"]
ENERGY_FIELDS = [
    "act_energy",
    "read_energy",
    "write_energy",
    "ref_energy",
    "background_energy",
    "total_energy",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def to_float(row: dict[str, str], field: str, default: float = 0.0) -> float:
    try:
        return float(row.get(field, default) or default)
    except (TypeError, ValueError):
        return default


def average_rows(rows: Iterable[dict[str, str]], row_type: str = "t2t") -> dict[str, float]:
    selected = [row for row in rows if row.get("type") == row_type]
    if not selected:
        selected = list(rows)
    out: dict[str, float] = defaultdict(float)
    if not selected:
        return {}
    for row in selected:
        for key, value in row.items():
            try:
                out[key] += float(value)
            except (TypeError, ValueError):
                pass
    for key in list(out):
        out[key] /= len(selected)
    return dict(out)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_result_name(path: Path) -> dict[str, int | str] | None:
    parts = path.stem.split("_")
    try:
        mem_type = parts[1]
        batch = int(next(part[1:] for part in parts if part.startswith("b")))
        seq_len = int(next(part[1:] for part in parts if part.startswith("l")))
        reorder = parts[parts.index("reorder") + 1]
        ramulator = parts[parts.index("ramul") + 1]
    except (IndexError, StopIteration, ValueError):
        return None
    if mem_type != "hbm3e":
        return None
    return {
        "memory_type": mem_type,
        "batch_size": batch,
        "seq_len": seq_len,
        "reorder": reorder,
        "ramulator": ramulator,
    }


def attention_breakdown(avg: dict[str, float]) -> dict[str, float]:
    is_absorb = (avg.get("tr_k_up_proj", 0.0) + avg.get("v_up_proj", 0.0)) > 0
    kv_decompress = (
        avg.get("tr_k_up_proj", 0.0) + avg.get("v_up_proj", 0.0)
        if is_absorb
        else avg.get("kv_up_proj", 0.0)
    )
    score_context = avg.get("atten_sum", 0.0) + avg.get("atten_gen", 0.0)
    out_proj = avg.get("o_proj", 0.0)
    etc = (
        avg.get("qkvgen", 0.0)
        + avg.get("q_down_proj", 0.0)
        + avg.get("kv_down_proj", 0.0)
        + avg.get("kr_proj", 0.0)
        + avg.get("q_up_proj", 0.0)
        + avg.get("qr_proj", 0.0)
        + avg.get("rope", 0.0)
        + avg.get("layernorm", 0.0)
        + avg.get("residual", 0.0)
    )
    return {
        "kv_decompress_ns": kv_decompress,
        "score_context_ns": score_context,
        "out_proj_ns": out_proj,
        "etc_ns": etc,
        "attention_total_ns": kv_decompress + score_context + out_proj + etc,
        "is_absorb": float(is_absorb),
    }


def summarize_file(path: Path) -> dict[str, Any] | None:
    meta = parse_result_name(path)
    if meta is None:
        return None
    avg = average_rows(read_csv_rows(path), "t2t")
    if not avg:
        return None

    latency_ns = avg.get("latency", avg.get("time", 0.0))
    row: dict[str, Any] = {
        **meta,
        "latency_ns": latency_ns,
        "latency_ms": latency_ns / 1e6,
        "throughput_tokens_per_s": (
            avg.get("batchsize", 0.0) / (latency_ns * 1e-9)
            if latency_ns > 0 and avg.get("batchsize", 0.0) > 0
            else 0.0
        ),
        "sim_batchsize": avg.get("batchsize", 0.0),
        "seqlen": avg.get("seqlen", 0.0),
        "oom": avg.get("OOM", 0.0),
        "memory_capacity_bytes": avg.get("memory_capacity", 0.0),
        "activation_size_bytes": avg.get("activation_size", 0.0),
        "weight_size_bytes": avg.get("weight_size", 0.0),
        "kv_cache_size_bytes": avg.get("kv_cache_size", 0.0),
        "total_memory_used_bytes": avg.get("total_memory_used", 0.0),
        "memory_utilization_pct": avg.get("memory_utilization", 0.0),
    }

    row.update(attention_breakdown(avg))
    for field in COUNT_FIELDS:
        row[field] = avg.get(field, 0.0)
    row["memory_duration_ns"] = avg.get("memory_duration", 0.0)
    row["background_time_ns"] = avg.get("background_time", 0.0)
    row["memory_duration_ms"] = row["memory_duration_ns"] / 1e6
    row["background_time_ms"] = row["background_time_ns"] / 1e6
    for field in ENERGY_FIELDS:
        row[f"{field}_nJ"] = avg.get(field, 0.0)
    return row


def collect_results() -> list[dict[str, Any]]:
    rows = []
    for csv_file in sorted(DATA_DIR.glob("result_hbm3e_b*_l*_reorder_*_ramul_*.csv")):
        row = summarize_file(csv_file)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: (r["reorder"], r["ramulator"], r["seq_len"], r["batch_size"]))
    return rows


def result_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    return {
        (str(r["reorder"]), str(r["ramulator"]), int(r["seq_len"]), int(r["batch_size"])): r
        for r in rows
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    data = result_index(rows)
    print("\nHBM3E Ramulator hierarchy summary")
    print("=" * 112)
    print(
        f"{'Reorder':<8} {'Seq':<6} {'Batch/GPU':<10} "
        f"{'Ideal ms':>10} {'Ramulator ms':>13} {'Ratio':>8} "
        f"{'ACT':>12} {'READ':>12} {'WRITE':>12} {'REF':>10} "
        f"{'mem_dur ms':>11} {'bg_time ms':>11}"
    )
    print("-" * 112)

    for reorder in REORDERING_MODES:
        for seq_len in SEQ_LENGTHS:
            for batch in BATCH_PER_GPU:
                off = data.get((reorder, "off", seq_len, batch))
                on = data.get((reorder, "on", seq_len, batch))
                if not off and not on:
                    continue
                ideal_ms = off.get("latency_ms", 0.0) if off else 0.0
                ramul_ms = on.get("latency_ms", 0.0) if on else 0.0
                ratio = ramul_ms / ideal_ms if ideal_ms > 0 and ramul_ms > 0 else 0.0
                counts = on or off or {}
                print(
                    f"{reorder:<8} {seq_len:<6} {batch:<10} "
                    f"{ideal_ms:>10.2f} {ramul_ms:>13.2f} {ratio:>8.2f} "
                    f"{counts.get('act_count', 0.0):>12.0f} "
                    f"{counts.get('read_count', 0.0):>12.0f} "
                    f"{counts.get('write_count', 0.0):>12.0f} "
                    f"{counts.get('ref_count', 0.0):>10.0f} "
                    f"{counts.get('memory_duration_ms', 0.0):>11.2f} "
                    f"{counts.get('background_time_ms', 0.0):>11.2f}"
                )
            print()


def generate_plots(rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    data = result_index(rows)

    for reorder in REORDERING_MODES:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        vals = []
        for seq_len in SEQ_LENGTHS:
            for batch in BATCH_PER_GPU:
                for ramul in RAMULATOR_MODES:
                    row = data.get((reorder, ramul, seq_len, batch))
                    if row:
                        vals.append(row["latency_ms"])
        positive_vals = [v for v in vals if v > 0]
        y_min = min(positive_vals) * 0.8 if positive_vals else 1
        y_max = max(positive_vals) * 1.3 if positive_vals else 10

        for idx, seq_len in enumerate(SEQ_LENGTHS):
            ax = axes[idx]
            x = np.arange(len(BATCH_PER_GPU))
            width = 0.35
            ideal_vals = []
            ramul_vals = []
            for batch in BATCH_PER_GPU:
                ideal_vals.append(data.get((reorder, "off", seq_len, batch), {}).get("latency_ms", 0.0))
                ramul_vals.append(data.get((reorder, "on", seq_len, batch), {}).get("latency_ms", 0.0))

            ax.bar(x - width / 2, ideal_vals, width, label="Ramulator off", color="#4C78A8")
            ax.bar(x + width / 2, ramul_vals, width, label="Ramulator on", color="#F58518")
            ax.set_title(f"Seq len = {seq_len}")
            ax.set_xlabel("Batch per GPU")
            if idx == 0:
                ax.set_ylabel("Latency (ms)")
                ax.legend()
            ax.set_xticks(x)
            ax.set_xticklabels(BATCH_PER_GPU)
            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
            ax.grid(axis="y", alpha=0.3)
            for i, (ideal, ramul) in enumerate(zip(ideal_vals, ramul_vals)):
                if ideal > 0 and ramul > 0:
                    ax.text(i, max(ideal, ramul) * 1.03, f"{ramul / ideal:.2f}x", ha="center", fontsize=8)

        fig.suptitle(f"HBM3E Ramulator hierarchy impact, reordering {reorder.upper()}")
        fig.tight_layout()
        out = PLOT_DIR / f"hbm3e_ramulator_reorder_{reorder}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved {out}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    count_specs = [
        ("act_count", "ACT"),
        ("read_count", "READ"),
        ("write_count", "WRITE"),
        ("ref_count", "REF"),
    ]
    seq_len = 4096
    for ax, (field, title) in zip(axes.flat, count_specs):
        x = np.arange(len(BATCH_PER_GPU))
        width = 0.35
        for offset, reorder in [(-width / 2, "on"), (width / 2, "off")]:
            vals = [
                data.get((reorder, "on", seq_len, batch), {}).get(field, 0.0)
                for batch in BATCH_PER_GPU
            ]
            ax.bar(x + offset, vals, width, label=f"reorder {reorder}")
        ax.set_title(f"{title} count, Ramulator on, L={seq_len}")
        ax.set_xlabel("Batch per GPU")
        ax.set_ylabel("Average command count")
        ax.set_xticks(x)
        ax.set_xticklabels(BATCH_PER_GPU)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out = PLOT_DIR / "hbm3e_ramulator_command_counts.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


def generate_report(rows: list[dict[str, Any]]) -> None:
    data = result_index(rows)
    lines = [
        "# HBM3E Ramulator Hierarchy Report",
        "",
        "This directory now contains only the HBM3E experiment matrix:",
        "",
        "- Reordering: on/off",
        "- Sequence length: 2048/4096/8192",
        "- Batch per GPU: 32/64/128/256",
        "- Ramulator hierarchy simulation: on/off",
        "",
        "The raw simulator CSV already includes ACT/READ/WRITE/REF command counts.",
        "`memory_duration` is memory service time. `background_time` is the DRAM background/standby energy time base accumulated from execution durations; it is intentionally kept separate from `memory_duration`.",
        "",
        "## Latency And Command Counts",
        "",
        "| Reorder | Seq | Batch/GPU | Off ms | On ms | On/Off | ACT | READ | WRITE | REF | memory_duration ms | background_time ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for reorder in REORDERING_MODES:
        for seq_len in SEQ_LENGTHS:
            for batch in BATCH_PER_GPU:
                off = data.get((reorder, "off", seq_len, batch))
                on = data.get((reorder, "on", seq_len, batch))
                if not off and not on:
                    continue
                off_ms = off.get("latency_ms", 0.0) if off else 0.0
                on_ms = on.get("latency_ms", 0.0) if on else 0.0
                ratio = on_ms / off_ms if off_ms > 0 and on_ms > 0 else 0.0
                counts = on or off or {}
                lines.append(
                    f"| {reorder} | {seq_len} | {batch} | {off_ms:.2f} | {on_ms:.2f} | {ratio:.2f} | "
                    f"{counts.get('act_count', 0.0):.0f} | {counts.get('read_count', 0.0):.0f} | "
                    f"{counts.get('write_count', 0.0):.0f} | {counts.get('ref_count', 0.0):.0f} | "
                    f"{counts.get('memory_duration_ms', 0.0):.2f} | {counts.get('background_time_ms', 0.0):.2f} |"
                )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- Raw CSV: `data/result_hbm3e_b{B}_l{L}_reorder_{on|off}_ramul_{on|off}.csv`",
            "- Summary CSV: `data/summary_hbm3e.csv`",
            "- Plots: `plots/hbm3e_ramulator_*.png`",
            "- Per-run configs: `configs/result_hbm3e_*.yaml`",
            "- Per-run logs: `logs/result_hbm3e_*.log`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Saved {REPORT_PATH}")


def expected_names() -> set[str]:
    names = set()
    for reorder in REORDERING_MODES:
        for seq_len in SEQ_LENGTHS:
            for batch in BATCH_PER_GPU:
                for ramul in RAMULATOR_MODES:
                    names.add(f"result_hbm3e_b{batch}_l{seq_len}_reorder_{reorder}_ramul_{ramul}.csv")
    return names


def print_missing() -> None:
    existing = {path.name for path in DATA_DIR.glob("result_hbm3e_b*_l*_reorder_*_ramul_*.csv")}
    missing = sorted(expected_names() - existing)
    extra = sorted(existing - expected_names())
    if missing:
        print("Missing expected HBM3E results:")
        for name in missing:
            print(f"  {name}")
    if extra:
        print("Extra HBM3E results outside the current matrix:")
        for name in extra:
            print(f"  {name}")
    if not missing and not extra:
        print("HBM3E result matrix is complete and contains no extra HBM3E CSVs.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--table", action="store_true", help="Print a compact table")
    parser.add_argument("--report", action="store_true", help="Generate markdown report")
    parser.add_argument("--missing", action="store_true", help="Show missing/extra result files")
    parser.add_argument("--all", action="store_true", help="Generate summary, table, plots, and report")
    args = parser.parse_args()

    if not (args.plot or args.table or args.report or args.missing or args.all):
        args.all = True

    rows = collect_results()
    print(f"Found {len(rows)} HBM3E result entries")
    write_summary_csv(DATA_DIR / "summary_hbm3e.csv", rows)
    print(f"Saved {DATA_DIR / 'summary_hbm3e.csv'}")

    if args.missing or args.all:
        print_missing()
    if args.table or args.all:
        print_table(rows)
    if args.plot or args.all:
        generate_plots(rows)
    if args.report or args.all:
        generate_report(rows)


if __name__ == "__main__":
    main()
