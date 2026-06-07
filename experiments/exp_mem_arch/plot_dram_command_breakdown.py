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
TOTAL_COMPONENT_LABELS = ["ACT", "READ", "WRITE", "REF", "BG", "MAC"]
TOTAL_ENERGY_KEYS = ["act_J_step", "read_J_step", "write_J_step", "ref_J_step", "background_J_step", "mac_J_step"]
TOTAL_SHARE_KEYS = [
    "act_pct_total_plus_mac",
    "read_pct_total_plus_mac",
    "write_pct_total_plus_mac",
    "ref_pct_total_plus_mac",
    "background_pct_total_plus_mac",
    "mac_pct_total_plus_mac",
]
PPT_HEATMAP_SIZE = (10.7, 5.15)


def read_energy_csv(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                out: dict[str, float | int | str] = {
                    "reorder": row["reorder"],
                    "seq_len": int(row["seq_len"]),
                    "batch_per_gpu": int(row["batch_per_gpu"]),
                }
                for key in TOTAL_ENERGY_KEYS + TOTAL_SHARE_KEYS:
                    out[key] = float(row[key])
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


def plot_energy_heatmaps(rows: list[dict[str, float | int | str]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm

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
        len(TOTAL_ENERGY_KEYS),
        figsize=PPT_HEATMAP_SIZE,
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    for col_idx, (label, key) in enumerate(zip(TOTAL_COMPONENT_LABELS, TOTAL_ENERGY_KEYS)):
        values_for_component = [float(row[key]) for row in rows]
        norm = LogNorm(vmin=max(min(values_for_component), 1e-3), vmax=max(values_for_component))
        for row_idx, reorder in enumerate(["on", "off"]):
            ax = axes[row_idx][col_idx]
            matrix = np.array(
                [
                    [float(data[(reorder, batch, seq_len)][key]) for seq_len in SEQ_LENGTHS]
                    for batch in BATCH_PER_GPU
                ]
            )
            im = ax.imshow(matrix, cmap="YlOrRd", norm=norm, aspect="auto")
            for i, batch in enumerate(BATCH_PER_GPU):
                for j, seq_len in enumerate(SEQ_LENGTHS):
                    value = matrix[i, j]
                    text_color = "white" if norm(value) > 0.58 else "#111111"
                    ax.text(j, i, _format_cell_value(value), ha="center", va="center", fontsize=6.7, color=text_color)

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

    fig.suptitle("Component energy including MAC (J/step)", fontsize=10.2, y=0.985)
    fig.text(0.976, 0.965, "cell = J/step", ha="right", va="top", fontsize=7.2, color="#444444")
    fig.subplots_adjust(left=0.058, right=0.992, top=0.875, bottom=0.105, hspace=0.24, wspace=0.065)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    print(f"Saved {output}")


def plot_share_heatmaps(rows: list[dict[str, float | int | str]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

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
        len(TOTAL_SHARE_KEYS),
        figsize=PPT_HEATMAP_SIZE,
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    for col_idx, (label, key) in enumerate(zip(TOTAL_COMPONENT_LABELS, TOTAL_SHARE_KEYS)):
        for row_idx, reorder in enumerate(["on", "off"]):
            ax = axes[row_idx][col_idx]
            matrix = np.array(
                [
                    [float(data[(reorder, batch, seq_len)][key]) for seq_len in SEQ_LENGTHS]
                    for batch in BATCH_PER_GPU
                ]
            )
            ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
            for i, batch in enumerate(BATCH_PER_GPU):
                for j, seq_len in enumerate(SEQ_LENGTHS):
                    value = matrix[i, j]
                    text_color = "white" if value > 55 else "#111111"
                    ax.text(j, i, f"{value:.1f}%", ha="center", va="center", fontsize=6.7, color=text_color)

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

    fig.suptitle("Energy share including MAC", fontsize=10.2, y=0.985)
    fig.text(0.976, 0.965, "cell = share of Total+MAC J/step", ha="right", va="top", fontsize=7.2, color="#444444")
    fig.subplots_adjust(left=0.058, right=0.992, top=0.875, bottom=0.105, hspace=0.24, wspace=0.065)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
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
        "--heatmap-output",
        type=Path,
        default=PLOT_DIR / "figure_dram_command_energy_heatmaps.png",
        help="Output path for annotated raw component energy heatmaps",
    )
    parser.add_argument(
        "--share-heatmap-output",
        type=Path,
        default=PLOT_DIR / "figure_dram_command_share_heatmaps.png",
        help="Output path for annotated relative component share heatmaps",
    )
    args = parser.parse_args()

    energy_rows = read_energy_csv(args.energy_csv)
    if not energy_rows:
        raise SystemExit(f"No energy rows found in {args.energy_csv}")
    plot_energy_heatmaps(energy_rows, args.heatmap_output)
    plot_share_heatmaps(energy_rows, args.share_heatmap_output)


if __name__ == "__main__":
    main()
