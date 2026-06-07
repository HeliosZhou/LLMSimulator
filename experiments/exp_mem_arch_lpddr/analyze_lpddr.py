#!/usr/bin/env python3
"""Analyze the 24-run Ramulator-on LPDDR5-parameter energy experiment."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
PLOT_DIR = EXP_DIR / "plots"
REPORT_PATH = EXP_DIR / "LPDDR5_DRAMPOWER_ANALYSIS.md"

BATCH_PER_GPU = [32, 64, 128, 256]
SEQ_LENGTHS = [2048, 4096, 8192]
REORDERING_MODES = ["on", "off"]

COUNT_FIELDS = [
    "act_count",
    "read_count",
    "write_count",
    "all_act_count",
    "all_read_count",
    "all_write_count",
    "ref_count",
]
DRAMPOWER_ENERGY_FIELDS = [
    "drampower_act_energy",
    "drampower_read_energy",
    "drampower_write_energy",
    "drampower_all_act_energy",
    "drampower_all_read_energy",
    "drampower_all_write_energy",
    "drampower_ref_energy",
    "drampower_background_energy",
    "drampower_total_energy",
]
DRAM_COMPONENTS = [
    ("ACT", "drampower_act_energy_nJ"),
    ("READ", "drampower_read_energy_nJ"),
    ("WRITE", "drampower_write_energy_nJ"),
    ("REF", "drampower_ref_energy_nJ"),
    ("BG", "drampower_background_energy_nJ"),
]
TOTAL_COMPONENTS = DRAM_COMPONENTS + [("MAC", "mac_energy_nJ")]
PPT_DRAM_HEATMAP_SIZE = (9.15, 5.15)
PPT_TOTAL_HEATMAP_SIZE = (10.7, 5.15)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


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
        batch = int(next(part[1:] for part in parts if part.startswith("b") and part[1:].isdigit()))
        seq_len = int(next(part[1:] for part in parts if part.startswith("l") and part[1:].isdigit()))
        reorder = parts[parts.index("reorder") + 1]
        ramulator = parts[parts.index("ramul") + 1]
    except (IndexError, StopIteration, ValueError):
        return None
    if ramulator != "on":
        return None
    return {
        "memory_type": "hbm3e_lpddr",
        "batch_size": batch,
        "seq_len": seq_len,
        "reorder": reorder,
        "ramulator": ramulator,
        "drampower": "lpddr5",
    }


def summarize_file(path: Path) -> dict[str, Any] | None:
    rows = read_csv_rows(path)
    meta = parse_result_name(path)
    if meta is None:
        return None
    avg = average_rows(rows, "t2t")
    if not avg:
        return None
    model_name = next((row.get("dram_energy_model", "") for row in rows if row.get("dram_energy_model")), "")
    latency_ns = avg.get("latency", avg.get("time", 0.0))
    sim_batchsize = avg.get("batchsize", 0.0)

    row: dict[str, Any] = {
        **meta,
        "source_file": path.name,
        "dram_energy_model": model_name or "fgdram+drampower_lpddr5_params",
        "latency_ns": latency_ns,
        "latency_ms": latency_ns / 1e6,
        "throughput_tokens_per_s": sim_batchsize / (latency_ns * 1e-9) if latency_ns > 0 else 0.0,
        "sim_batchsize": sim_batchsize,
        "seqlen": avg.get("seqlen", 0.0),
        "oom": avg.get("OOM", 0.0),
        "memory_capacity_bytes": avg.get("memory_capacity", 0.0),
        "activation_size_bytes": avg.get("activation_size", 0.0),
        "weight_size_bytes": avg.get("weight_size", 0.0),
        "kv_cache_size_bytes": avg.get("kv_cache_size", 0.0),
        "total_memory_used_bytes": avg.get("total_memory_used", 0.0),
        "memory_utilization_pct": avg.get("memory_utilization", 0.0),
        "memory_duration_ns": avg.get("memory_duration", 0.0),
        "background_time_ns": avg.get("background_time", 0.0),
        "memory_duration_ms": avg.get("memory_duration", 0.0) / 1e6,
        "background_time_ms": avg.get("background_time", 0.0) / 1e6,
        "mac_energy_nJ": avg.get("mac_energy", 0.0),
    }
    for field in COUNT_FIELDS:
        row[field] = avg.get(field, 0.0)
    for field in DRAMPOWER_ENERGY_FIELDS:
        row[f"{field}_nJ"] = avg.get(field, 0.0)
    row["drampower_total_energy_J"] = row["drampower_total_energy_nJ"] / 1e9
    row["mac_energy_J"] = row["mac_energy_nJ"] / 1e9
    row["total_plus_mac_energy_J"] = row["drampower_total_energy_J"] + row["mac_energy_J"]
    return row


def collect_results() -> list[dict[str, Any]]:
    rows = []
    for csv_file in sorted(DATA_DIR.glob("result_hbm3e_lpddr_b*_l*_reorder_*_ramul_on.csv")):
        row = summarize_file(csv_file)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: (r["reorder"], r["seq_len"], r["batch_size"]))
    return rows


def result_index(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], dict[str, Any]]:
    return {
        (str(r["reorder"]), int(r["seq_len"]), int(r["batch_size"])): r
        for r in rows
    }


def expected_names() -> set[str]:
    names = set()
    for reorder in REORDERING_MODES:
        for seq_len in SEQ_LENGTHS:
            for batch in BATCH_PER_GPU:
                names.add(f"result_hbm3e_lpddr_b{batch}_l{seq_len}_reorder_{reorder}_ramul_on.csv")
    return names


def print_missing() -> bool:
    existing = {path.name for path in DATA_DIR.glob("result_hbm3e_lpddr_b*_l*_reorder_*_ramul_on.csv")}
    missing = sorted(expected_names() - existing)
    extra = sorted(existing - expected_names())
    if missing:
        print("Missing expected LPDDR5-parameter results:")
        for name in missing:
            print(f"  {name}")
    if extra:
        print("Extra LPDDR5-parameter result files outside the matrix:")
        for name in extra:
            print(f"  {name}")
    if not missing and not extra:
        print("LPDDR5-parameter 24-run result matrix is complete.")
    return not missing and not extra


def print_table(rows: list[dict[str, Any]]) -> None:
    data = result_index(rows)
    print("\nHBM3E Ramulator-on with LPDDR5 DRAMPower parameters")
    print("=" * 108)
    print(
        f"{'Reorder':<8} {'Seq':<6} {'Batch/GPU':<10} {'Latency ms':>11} "
        f"{'DRAM J':>10} {'MAC J':>10} {'ACT':>12} {'READ':>12} {'WRITE':>12} {'REF':>10}"
    )
    print("-" * 108)
    for reorder in REORDERING_MODES:
        for seq_len in SEQ_LENGTHS:
            for batch in BATCH_PER_GPU:
                row = data.get((reorder, seq_len, batch))
                if not row:
                    continue
                print(
                    f"{reorder:<8} {seq_len:<6} {batch:<10} "
                    f"{row.get('latency_ms', 0.0):>11.2f} "
                    f"{row.get('drampower_total_energy_J', 0.0):>10.3f} "
                    f"{row.get('mac_energy_J', 0.0):>10.3f} "
                    f"{row.get('act_count', 0.0):>12.0f} "
                    f"{row.get('read_count', 0.0):>12.0f} "
                    f"{row.get('write_count', 0.0):>12.0f} "
                    f"{row.get('ref_count', 0.0):>10.0f}"
                )
            print()


def _format_energy_cell_value(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _component_energy_j(row: dict[str, Any], field: str) -> float:
    return float(row.get(field, 0.0)) / 1e9


def _component_share(row: dict[str, Any], field: str, denominator_fields: list[str]) -> float:
    denominator = sum(float(row.get(denominator_field, 0.0)) for denominator_field in denominator_fields)
    if denominator <= 0:
        return 0.0
    return float(row.get(field, 0.0)) / denominator * 100.0


def plot_component_heatmaps(
    rows: list[dict[str, Any]],
    output: Path,
    *,
    components: list[tuple[str, str]],
    title: str,
    note: str,
    figsize: tuple[float, float],
    kind: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm, Normalize

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    data = result_index(rows)

    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.linewidth": 0.75,
        }
    )

    fig, axes = plt.subplots(
        2,
        len(components),
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    denominator_fields = [field for _, field in components]

    for col_idx, (label, field) in enumerate(components):
        if kind == "energy":
            values_for_component = [_component_energy_j(row, field) for row in rows]
            positive_values = [value for value in values_for_component if value > 0]
            vmin = max(min(positive_values), 1e-3) if positive_values else 1e-3
            vmax = max(positive_values) if positive_values else 1.0
            if vmax <= vmin:
                vmax = vmin * 1.01
            norm = LogNorm(vmin=vmin, vmax=vmax)
            cmap = "YlOrRd"
        elif kind == "share":
            norm = Normalize(vmin=0, vmax=100)
            cmap = "YlGnBu"
        else:
            raise ValueError(f"Unknown heatmap kind: {kind}")

        for row_idx, reorder in enumerate(REORDERING_MODES):
            ax = axes[row_idx][col_idx]
            if kind == "energy":
                matrix = np.array(
                    [
                        [_component_energy_j(data[(reorder, seq_len, batch)], field) for seq_len in SEQ_LENGTHS]
                        for batch in BATCH_PER_GPU
                    ]
                )
            else:
                matrix = np.array(
                    [
                        [
                            _component_share(data[(reorder, seq_len, batch)], field, denominator_fields)
                            for seq_len in SEQ_LENGTHS
                        ]
                        for batch in BATCH_PER_GPU
                    ]
                )
            ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
            for i, batch in enumerate(BATCH_PER_GPU):
                for j, seq_len in enumerate(SEQ_LENGTHS):
                    value = matrix[i, j]
                    if kind == "energy":
                        norm_value = norm(max(value, norm.vmin))
                        cell_text = _format_energy_cell_value(value)
                    else:
                        norm_value = norm(value)
                        cell_text = f"{value:.1f}%"
                    text_color = "white" if norm_value > 0.58 else "#111111"
                    ax.text(j, i, cell_text, ha="center", va="center", fontsize=6.7, color=text_color)

            if row_idx == 0:
                ax.set_title(label, fontsize=9.2)
            if col_idx == 0:
                ax.set_ylabel(f"reorder {reorder}\nB/GPU", fontsize=7.8)
            ax.set_xticks(range(len(SEQ_LENGTHS)))
            ax.set_xticklabels([str(seq) for seq in SEQ_LENGTHS], fontsize=6.9)
            ax.set_yticks(range(len(BATCH_PER_GPU)))
            ax.set_yticklabels([str(batch) for batch in BATCH_PER_GPU], fontsize=6.9)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks(np.arange(-0.5, len(SEQ_LENGTHS), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(BATCH_PER_GPU), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.7)
            ax.tick_params(which="minor", bottom=False, left=False)

    for ax in axes[-1]:
        ax.set_xlabel("Seq", fontsize=7.8)

    fig.suptitle(title, fontsize=10.2, y=0.985)
    fig.text(0.976, 0.965, note, ha="right", va="top", fontsize=7.2, color="#444444")
    fig.subplots_adjust(left=0.058, right=0.992, top=0.875, bottom=0.105, hspace=0.24, wspace=0.065)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Saved {output}")


def generate_plots(rows: list[dict[str, Any]]) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_component_heatmaps(
        rows,
        PLOT_DIR / "lpddr5_dram_only_command_energy_heatmaps.png",
        components=DRAM_COMPONENTS,
        title="LPDDR5-param DRAM command energy (J/step)",
        note="cell = J/step",
        figsize=PPT_DRAM_HEATMAP_SIZE,
        kind="energy",
    )
    plot_component_heatmaps(
        rows,
        PLOT_DIR / "lpddr5_dram_only_command_share_heatmaps.png",
        components=DRAM_COMPONENTS,
        title="LPDDR5-param DRAM command energy share",
        note="cell = share of DRAM J/step",
        figsize=PPT_DRAM_HEATMAP_SIZE,
        kind="share",
    )
    plot_component_heatmaps(
        rows,
        PLOT_DIR / "lpddr5_dram_command_energy_heatmaps.png",
        components=TOTAL_COMPONENTS,
        title="LPDDR5-param component energy including MAC (J/step)",
        note="cell = J/step",
        figsize=PPT_TOTAL_HEATMAP_SIZE,
        kind="energy",
    )
    plot_component_heatmaps(
        rows,
        PLOT_DIR / "lpddr5_dram_command_share_heatmaps.png",
        components=TOTAL_COMPONENTS,
        title="LPDDR5-param energy share including MAC",
        note="cell = share of Total+MAC J/step",
        figsize=PPT_TOTAL_HEATMAP_SIZE,
        kind="share",
    )


def generate_legacy_plots(rows: list[dict[str, Any]]) -> None:
    """Keep the previous plot shapes available for manual comparison."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    data = result_index(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, reorder in zip(axes, REORDERING_MODES):
        x = np.arange(len(BATCH_PER_GPU))
        width = 0.24
        for idx, seq_len in enumerate(SEQ_LENGTHS):
            vals = [
                data.get((reorder, seq_len, batch), {}).get("drampower_total_energy_J", 0.0)
                for batch in BATCH_PER_GPU
            ]
            ax.bar(x + (idx - 1) * width, vals, width, label=f"Seq {seq_len}")
        ax.set_title(f"reorder {reorder}")
        ax.set_xlabel("Batch per GPU")
        ax.set_xticks(x)
        ax.set_xticklabels(BATCH_PER_GPU)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("DRAMPower energy with LPDDR5 params (J/step)")
    fig.tight_layout()
    out = PLOT_DIR / "lpddr5_drampower_energy_by_batch.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True, sharey=True)
    components = TOTAL_COMPONENTS
    for ax, (label, field) in zip(axes.flat, components):
        matrix = np.array(
            [
                [
                    (data.get((reorder, seq_len, batch), {}).get(field, 0.0) / 1e9)
                    for seq_len in SEQ_LENGTHS
                    for reorder in ["on", "off"]
                ]
                for batch in BATCH_PER_GPU
            ]
        )
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_title(label)
        ax.set_yticks(range(len(BATCH_PER_GPU)))
        ax.set_yticklabels(BATCH_PER_GPU)
        ax.set_xticks(range(len(SEQ_LENGTHS) * 2))
        ax.set_xticklabels([f"{seq}\n{reorder}" for seq in SEQ_LENGTHS for reorder in ["on", "off"]], fontsize=7)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0][0].set_ylabel("Batch/GPU")
    axes[1][0].set_ylabel("Batch/GPU")
    fig.suptitle("Component energy with LPDDR5 DRAMPower params (J/step)")
    fig.tight_layout()
    out = PLOT_DIR / "lpddr5_component_energy_heatmaps.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def generate_report(rows: list[dict[str, Any]]) -> None:
    data = result_index(rows)
    lines = [
        "# LPDDR5 DRAMPower Parameter Experiment",
        "",
        "This experiment keeps the HBM3E/B200 system target and Ramulator-on command collection from `exp_mem_arch`, but switches DRAMPower energy accounting to the LPDDR5 parameter path.",
        "",
        "- Reordering: on/off",
        "- Sequence length: 2048/4096/8192",
        "- Batch per GPU: 32/64/128/256",
        "- Ramulator hierarchy simulation: on only",
        "- DRAMPower energy model: `fgdram+drampower_lpddr5_params`",
        "",
        "The latency, memory capacity, and command-count path still use the HBM3E system target. Only the DRAMPower current/timing parameters are switched to LPDDR5 for energy accounting.",
        "",
        "## Results",
        "",
        "| Reorder | Seq | Batch/GPU | Latency ms | DRAM J/step | MAC J/step | ACT | READ | WRITE | REF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for reorder in REORDERING_MODES:
        for seq_len in SEQ_LENGTHS:
            for batch in BATCH_PER_GPU:
                row = data.get((reorder, seq_len, batch))
                if not row:
                    continue
                lines.append(
                    f"| {reorder} | {seq_len} | {batch} | "
                    f"{row.get('latency_ms', 0.0):.2f} | "
                    f"{row.get('drampower_total_energy_J', 0.0):.4f} | "
                    f"{row.get('mac_energy_J', 0.0):.4f} | "
                    f"{row.get('act_count', 0.0):.0f} | "
                    f"{row.get('read_count', 0.0):.0f} | "
                    f"{row.get('write_count', 0.0):.0f} | "
                    f"{row.get('ref_count', 0.0):.0f} |"
                )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- Raw CSV: `data/result_hbm3e_lpddr_b{B}_l{L}_reorder_{on|off}_ramul_on.csv`",
            "- Summary CSV: `data/summary_lpddr.csv`",
            "- DRAM-only absolute heatmap: `plots/lpddr5_dram_only_command_energy_heatmaps.png`",
            "- DRAM-only relative heatmap: `plots/lpddr5_dram_only_command_share_heatmaps.png`",
            "- Total+MAC absolute heatmap: `plots/lpddr5_dram_command_energy_heatmaps.png`",
            "- Total+MAC relative heatmap: `plots/lpddr5_dram_command_share_heatmaps.png`",
            "- Per-run configs: `configs/result_hbm3e_lpddr_*.yaml`",
            "- Per-run logs: `logs/result_hbm3e_lpddr_*.log`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Saved {REPORT_PATH}")


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
    print(f"Found {len(rows)} LPDDR5-parameter result entries")
    write_summary_csv(DATA_DIR / "summary_lpddr.csv", rows)
    print(f"Saved {DATA_DIR / 'summary_lpddr.csv'}")

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
