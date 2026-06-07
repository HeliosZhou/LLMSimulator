#!/usr/bin/env python3
"""Plot ACT/READ/WRITE/REF/BG/MAC energy heatmaps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
PLOT_DIR = EXP_DIR / "plots"
BATCH_PER_GPU = [32, 64, 128, 256]
SEQ_LENGTHS = [2048, 4096, 8192]
ENERGY_CSV_PATH = EXP_DIR / "data" / "energy_breakdown_ramulator_on_drampower_ref.csv"
REORDER_MODES = ["on", "off"]
DRAM_COMPONENT_LABELS = ["ACT", "READ", "WRITE", "REF", "BG"]
DRAM_ENERGY_KEYS = ["act_J_step", "read_J_step", "write_J_step", "ref_J_step", "background_J_step"]
DRAM_SHARE_KEYS = ["act_pct_dram", "read_pct_dram", "write_pct_dram", "ref_pct_dram", "background_pct_dram"]
TOTAL_COMPONENT_LABELS = DRAM_COMPONENT_LABELS + ["MAC"]
TOTAL_ENERGY_KEYS = DRAM_ENERGY_KEYS + ["mac_J_step"]
TOTAL_SHARE_KEYS = [
    "act_pct_total_plus_mac",
    "read_pct_total_plus_mac",
    "write_pct_total_plus_mac",
    "ref_pct_total_plus_mac",
    "background_pct_total_plus_mac",
    "mac_pct_total_plus_mac",
]
PPT_DRAM_HEATMAP_SIZE = (9.15, 5.15)
PPT_TOTAL_HEATMAP_SIZE = (10.7, 5.15)


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _ensure_share_fields(row: dict[str, float | int | str]) -> None:
    denominators = [
        (sum(float(row[key]) for key in DRAM_ENERGY_KEYS), DRAM_ENERGY_KEYS, DRAM_SHARE_KEYS),
        (sum(float(row[key]) for key in TOTAL_ENERGY_KEYS), TOTAL_ENERGY_KEYS, TOTAL_SHARE_KEYS),
    ]
    for denominator, energy_keys, share_keys in denominators:
        for energy_key, share_key in zip(energy_keys, share_keys):
            if share_key not in row:
                row[share_key] = 0.0 if denominator <= 0 else float(row[energy_key]) / denominator * 100.0


def read_energy_csv(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    numeric_keys = _dedupe(TOTAL_ENERGY_KEYS + DRAM_SHARE_KEYS + TOTAL_SHARE_KEYS)
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                out: dict[str, float | int | str] = {
                    "reorder": row["reorder"],
                    "seq_len": int(row["seq_len"]),
                    "batch_per_gpu": int(row["batch_per_gpu"]),
                }
                for key in numeric_keys:
                    if key in row and row[key] != "":
                        out[key] = float(row[key])
                for key in TOTAL_ENERGY_KEYS:
                    if key not in out:
                        raise KeyError(key)
                _ensure_share_fields(out)
                rows.append(out)
            except (KeyError, ValueError):
                continue
    return rows


def _format_cell_value(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _format_pct_cell_value(value: float) -> str:
    return f"{value:.1f}%"


def plot_heatmaps(
    rows: list[dict[str, float | int | str]],
    output: Path,
    *,
    component_labels: list[str],
    value_keys: list[str],
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

    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.linewidth": 0.75,
        }
    )

    data = {
        (str(row["reorder"]), int(row["batch_per_gpu"]), int(row["seq_len"])): row
        for row in rows
    }
    fig, axes = plt.subplots(
        2,
        len(value_keys),
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    for col_idx, (label, key) in enumerate(zip(component_labels, value_keys)):
        values_for_component = [float(row[key]) for row in rows]
        if kind == "energy":
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

        for row_idx, reorder in enumerate(REORDER_MODES):
            ax = axes[row_idx][col_idx]
            matrix = np.array(
                [
                    [float(data[(reorder, batch, seq_len)][key]) for seq_len in SEQ_LENGTHS]
                    for batch in BATCH_PER_GPU
                ]
            )
            ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
            for i, batch in enumerate(BATCH_PER_GPU):
                for j, seq_len in enumerate(SEQ_LENGTHS):
                    value = matrix[i, j]
                    if kind == "energy":
                        norm_value = norm(max(value, norm.vmin))
                        cell_text = _format_cell_value(value)
                    else:
                        norm_value = norm(value)
                        cell_text = _format_pct_cell_value(value)
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

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--energy-csv",
        type=Path,
        default=ENERGY_CSV_PATH,
        help="CSV with absolute component energy fields",
    )
    parser.add_argument(
        "--dram-heatmap-output",
        type=Path,
        default=PLOT_DIR / "figure_dram_only_command_energy_heatmaps.png",
        help="Output path for annotated DRAM-only raw command energy heatmaps",
    )
    parser.add_argument(
        "--dram-share-heatmap-output",
        type=Path,
        default=PLOT_DIR / "figure_dram_only_command_share_heatmaps.png",
        help="Output path for annotated DRAM-only relative command share heatmaps",
    )
    parser.add_argument(
        "--heatmap-output",
        type=Path,
        default=PLOT_DIR / "figure_dram_command_energy_heatmaps.png",
        help="Output path for annotated raw component energy heatmaps including MAC",
    )
    parser.add_argument(
        "--share-heatmap-output",
        type=Path,
        default=PLOT_DIR / "figure_dram_command_share_heatmaps.png",
        help="Output path for annotated relative component share heatmaps including MAC",
    )
    args = parser.parse_args()

    energy_rows = read_energy_csv(args.energy_csv)
    if not energy_rows:
        raise SystemExit(f"No energy rows found in {args.energy_csv}")
    plot_heatmaps(
        energy_rows,
        args.dram_heatmap_output,
        component_labels=DRAM_COMPONENT_LABELS,
        value_keys=DRAM_ENERGY_KEYS,
        title="DRAM command energy (J/step)",
        note="cell = J/step",
        figsize=PPT_DRAM_HEATMAP_SIZE,
        kind="energy",
    )
    plot_heatmaps(
        energy_rows,
        args.dram_share_heatmap_output,
        component_labels=DRAM_COMPONENT_LABELS,
        value_keys=DRAM_SHARE_KEYS,
        title="DRAM command energy share",
        note="cell = share of DRAM J/step",
        figsize=PPT_DRAM_HEATMAP_SIZE,
        kind="share",
    )
    plot_heatmaps(
        energy_rows,
        args.heatmap_output,
        component_labels=TOTAL_COMPONENT_LABELS,
        value_keys=TOTAL_ENERGY_KEYS,
        title="Component energy including MAC (J/step)",
        note="cell = J/step",
        figsize=PPT_TOTAL_HEATMAP_SIZE,
        kind="energy",
    )
    plot_heatmaps(
        energy_rows,
        args.share_heatmap_output,
        component_labels=TOTAL_COMPONENT_LABELS,
        value_keys=TOTAL_SHARE_KEYS,
        title="Energy share including MAC",
        note="cell = share of Total+MAC J/step",
        figsize=PPT_TOTAL_HEATMAP_SIZE,
        kind="share",
    )


if __name__ == "__main__":
    main()
