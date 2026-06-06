#!/usr/bin/env python3
"""Plot DRAM J/step against memory footprint for reorder on/off."""

from __future__ import annotations

import argparse
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
REPORT_PATH = EXP_DIR / "POWER_RESULTS_ANALYSIS.zh-CN.md"
PLOT_DIR = EXP_DIR / "plots"
SEQ_LENGTHS = [2048, 4096, 8192]
BATCH_PER_GPU = [32, 64, 128, 256]

MEMORY_TOTAL_GB = {
    ("on", 32, 2048): 73.33,
    ("on", 32, 4096): 77.58,
    ("on", 32, 8192): 86.08,
    ("on", 64, 2048): 77.60,
    ("on", 64, 4096): 86.10,
    ("on", 64, 8192): 103.10,
    ("on", 128, 2048): 86.15,
    ("on", 128, 4096): 103.15,
    ("on", 128, 8192): 137.15,
    ("on", 256, 2048): 103.24,
    ("on", 256, 4096): 137.24,
    ("on", 256, 8192): 205.24,
    ("off", 32, 2048): 77.32,
    ("off", 32, 4096): 85.57,
    ("off", 32, 8192): 102.07,
    ("off", 64, 2048): 85.59,
    ("off", 64, 4096): 102.09,
    ("off", 64, 8192): 135.09,
    ("off", 128, 2048): 102.13,
    ("off", 128, 4096): 135.13,
    ("off", 128, 8192): 201.13,
    ("off", 256, 2048): 135.21,
    ("off", 256, 4096): 201.21,
    ("off", 256, 8192): 333.21,
}

OOM_CASES = {
    ("on", 256, 8192),
    ("off", 128, 8192),
    ("off", 256, 4096),
    ("off", 256, 8192),
}


def _to_float(value: str) -> float:
    return float(value.replace("%", "").replace("x", "").strip())


def parse_power_table(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cols = [col.strip() for col in line.strip().strip("|").split("|")]
        if len(cols) != 13 or cols[0] not in {"on", "off"}:
            continue
        try:
            row = {
                "reorder": cols[0],
                "seq_len": int(cols[1]),
                "batch_per_gpu": int(cols[2]),
                "latency_ms": _to_float(cols[3]),
                "memory_duration_ms": _to_float(cols[4]),
                "dram_j_step": _to_float(cols[5]),
                "dram_j_token": _to_float(cols[6]),
                "total_mac_j_token": _to_float(cols[7]),
                "act_pct": _to_float(cols[8]),
                "read_pct": _to_float(cols[9]),
                "write_pct": _to_float(cols[10]),
                "ref_pct": _to_float(cols[11]),
                "background_pct": _to_float(cols[12]),
            }
        except ValueError:
            continue
        rows.append(row)
    return rows


def plot(rows: list[dict[str, float | int | str]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        2048: "#d96b27",
        4096: "#2878b5",
        8192: "#0b6b3a",
    }
    markers = {
        32: "o",
        64: "s",
        128: "^",
        256: "D",
    }

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.linewidth": 0.8,
            "legend.handlelength": 1.0,
            "legend.handletextpad": 0.35,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4), sharey=True)
    step_values = [float(row["dram_j_step"]) for row in rows]
    token_values = [float(row["dram_j_token"]) for row in rows]
    token_axes = []

    for ax, reorder in zip(axes, ["on", "off"]):
        ax_token = ax.twinx()
        token_axes.append(ax_token)
        subset = [row for row in rows if row["reorder"] == reorder]
        for seq_len in SEQ_LENGTHS:
            seq_rows = sorted(
                [row for row in subset if row["seq_len"] == seq_len],
                key=lambda row: int(row["batch_per_gpu"]),
            )
            if not seq_rows:
                continue
            xs = [
                MEMORY_TOTAL_GB[(str(row["reorder"]), int(row["batch_per_gpu"]), int(row["seq_len"]))]
                for row in seq_rows
            ]
            step_ys = [float(row["dram_j_step"]) for row in seq_rows]
            token_ys = [float(row["dram_j_token"]) for row in seq_rows]
            ax.plot(
                xs,
                step_ys,
                color=colors[seq_len],
                linewidth=1.6,
                label=f"Seq {seq_len}",
                zorder=2,
            )
            ax_token.plot(
                xs,
                token_ys,
                color=colors[seq_len],
                linewidth=1.35,
                linestyle="--",
                alpha=0.80,
                zorder=1,
            )
            for row in seq_rows:
                batch = int(row["batch_per_gpu"])
                key = (str(row["reorder"]), batch, int(row["seq_len"]))
                x = MEMORY_TOTAL_GB[key]
                is_oom = key in OOM_CASES
                ax.scatter(
                    x,
                    float(row["dram_j_step"]),
                    marker=markers[batch],
                    s=36,
                    color=colors[seq_len],
                    edgecolor="black",
                    linewidth=1.2 if is_oom else 0.55,
                    alpha=0.45 if is_oom else 1.0,
                    zorder=3,
                )
                ax_token.scatter(
                    x,
                    float(row["dram_j_token"]),
                    marker=markers[batch],
                    s=32,
                    facecolors="none",
                    edgecolors=colors[int(row["seq_len"])],
                    linewidth=1.1 if is_oom else 0.75,
                    alpha=0.45 if is_oom else 0.85,
                    zorder=2,
                )
        ax.set_yscale("log")
        ax_token.set_yscale("log")
        ax.grid(True, which="major", axis="both", alpha=0.28, linewidth=0.6)
        ax.grid(True, which="minor", axis="y", alpha=0.12, linewidth=0.4)
        ax.set_xlabel("Total memory footprint (GB)", fontsize=8.5)
        ax.set_title(f"reorder {reorder}", fontsize=10.5)
        ax.spines["top"].set_visible(False)
        ax_token.spines["top"].set_visible(False)
        ax.set_ylim(min(step_values) * 0.75, max(step_values) * 1.35)
        ax_token.set_ylim(min(token_values) * 0.75, max(token_values) * 1.35)
        ax.set_xlim(65, 350)

    axes[0].set_ylabel("DRAM energy (J/step)", fontsize=8.5)
    token_axes[-1].set_ylabel("DRAM energy (J/token)", fontsize=8.5)
    token_axes[0].set_yticklabels([])

    seq_handles = [
        plt.Line2D([0], [0], color=colors[seq_len], linewidth=1.8, label=f"Seq {seq_len}")
        for seq_len in SEQ_LENGTHS
    ]
    batch_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=markers[batch],
            linestyle="None",
            color="#555555",
            markerfacecolor="#dddddd",
            markeredgecolor="black",
            markersize=6,
            label=f"B/GPU {batch}",
        )
        for batch in BATCH_PER_GPU
    ]
    oom_handle = plt.Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        color="#555555",
        markerfacecolor="#dddddd",
        markeredgecolor="black",
        alpha=0.45,
        markersize=6,
        label="OOM",
    )
    metric_handles = [
        plt.Line2D([0], [0], color="#222222", linewidth=1.8, linestyle="-", label="J/step"),
        plt.Line2D([0], [0], color="#222222", linewidth=1.5, linestyle="--", label="J/token"),
    ]
    fig.legend(
        handles=metric_handles + seq_handles + batch_handles + [oom_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        frameon=False,
        fontsize=7.6,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    print(f"Saved {output}")


def _plot_reorder_panel(
    ax,
    rows: list[dict[str, float | int | str]],
    reorder: str,
    colors: dict[int, str],
    markers: dict[int, str],
) -> tuple[list[float], list[float], list[float], object]:
    xs_all: list[float] = []
    step_values: list[float] = []
    token_values: list[float] = []
    ax_token = ax.twinx()
    subset = [row for row in rows if row["reorder"] == reorder]
    for seq_len in SEQ_LENGTHS:
        seq_rows = sorted(
            [row for row in subset if row["seq_len"] == seq_len],
            key=lambda row: int(row["batch_per_gpu"]),
        )
        if not seq_rows:
            continue
        xs = [
            MEMORY_TOTAL_GB[(str(row["reorder"]), int(row["batch_per_gpu"]), int(row["seq_len"]))]
            for row in seq_rows
        ]
        step_ys = [float(row["dram_j_step"]) for row in seq_rows]
        token_ys = [float(row["dram_j_token"]) for row in seq_rows]
        xs_all.extend(xs)
        step_values.extend(step_ys)
        token_values.extend(token_ys)
        ax.plot(xs, step_ys, color=colors[seq_len], linewidth=1.8, label=f"Seq {seq_len}", zorder=2)
        ax_token.plot(
            xs,
            token_ys,
            color=colors[seq_len],
            linewidth=1.5,
            linestyle="--",
            alpha=0.80,
            zorder=1,
        )
        for row in seq_rows:
            batch = int(row["batch_per_gpu"])
            key = (str(row["reorder"]), batch, int(row["seq_len"]))
            is_oom = key in OOM_CASES
            x = MEMORY_TOTAL_GB[key]
            ax.scatter(
                x,
                float(row["dram_j_step"]),
                marker=markers[batch],
                s=52,
                color=colors[seq_len],
                edgecolor="black",
                linewidth=1.2 if is_oom else 0.6,
                alpha=0.45 if is_oom else 1.0,
                zorder=3,
            )
            ax_token.scatter(
                x,
                float(row["dram_j_token"]),
                marker=markers[batch],
                s=46,
                facecolors="none",
                edgecolors=colors[int(row["seq_len"])],
                linewidth=1.2 if is_oom else 0.8,
                alpha=0.45 if is_oom else 0.85,
                zorder=2,
            )

    ax.set_yscale("log")
    ax_token.set_yscale("log")
    ax.grid(True, which="major", axis="both", alpha=0.28, linewidth=0.6)
    ax.grid(True, which="minor", axis="y", alpha=0.12, linewidth=0.4)
    ax.set_xlabel("Total memory footprint (GB)")
    ax.set_ylabel("DRAM energy (J/step)")
    ax_token.set_ylabel("DRAM energy (J/token)")
    ax.set_title(f"reorder {reorder}", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax_token.spines["top"].set_visible(False)
    return xs_all, step_values, token_values, ax_token


def plot_separate(rows: list[dict[str, float | int | str]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        2048: "#d96b27",
        4096: "#2878b5",
        8192: "#0b6b3a",
    }
    markers = {
        32: "o",
        64: "s",
        128: "^",
        256: "D",
    }
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.linewidth": 0.8,
            "legend.handlelength": 1.2,
            "legend.handletextpad": 0.4,
        }
    )

    for reorder in ["on", "off"]:
        fig, ax = plt.subplots(1, 1, figsize=(6.9, 5.1))
        xs, step_ys, token_ys, ax_token = _plot_reorder_panel(ax, rows, reorder, colors, markers)
        if xs and step_ys:
            x_margin = (max(xs) - min(xs)) * 0.10
            ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
            ax.set_ylim(min(step_ys) * 0.85, max(step_ys) * 1.18)
            ax_token.set_ylim(min(token_ys) * 0.85, max(token_ys) * 1.18)

        seq_handles = [
            plt.Line2D([0], [0], color=colors[seq_len], linewidth=1.8, label=f"Seq {seq_len}")
            for seq_len in SEQ_LENGTHS
        ]
        batch_handles = [
            plt.Line2D(
                [0],
                [0],
                marker=markers[batch],
                linestyle="None",
                color="#555555",
                markerfacecolor="#dddddd",
                markeredgecolor="black",
                markersize=6,
                label=f"B/GPU {batch}",
            )
            for batch in BATCH_PER_GPU
        ]
        oom_handle = plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="#555555",
            markerfacecolor="#dddddd",
            markeredgecolor="black",
            alpha=0.45,
            markersize=6,
            label="OOM",
        )
        metric_handles = [
            plt.Line2D([0], [0], color="#222222", linewidth=1.8, linestyle="-", label="J/step"),
            plt.Line2D([0], [0], color="#222222", linewidth=1.5, linestyle="--", label="J/token"),
        ]
        fig.legend(
            handles=metric_handles + seq_handles + batch_handles + [oom_handle],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=5,
            frameon=False,
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.16, 1, 1))
        output = output_dir / f"figure_dram_jstep_vs_total_memory_reorder_{reorder}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220)
        print(f"Saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=REPORT_PATH, help="Markdown report to parse")
    parser.add_argument(
        "--output",
        type=Path,
        default=PLOT_DIR / "figure_dram_jstep_vs_total_memory.png",
        help="Output image path",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Also save reorder-on and reorder-off plots with independent axis ranges",
    )
    args = parser.parse_args()

    rows = parse_power_table(args.input)
    if not rows:
        raise SystemExit(f"No 24-row power result table found in {args.input}")
    plot(rows, args.output)
    if args.separate:
        plot_separate(rows, args.output.parent)


if __name__ == "__main__":
    main()
