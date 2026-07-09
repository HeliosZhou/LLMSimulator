#!/usr/bin/env python3
"""Plot RowHammer/RowPress-oriented figures from rowrisk CSV outputs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DEFAULT_TCK_NS = 0.5
DEFAULT_REFRESH_INTERVAL_CYCLES = 7800


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def as_int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value == "":
        return 0
    return int(float(value))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_row_open_count_distribution(plt, aggressors: list[dict[str, str]], output: Path) -> None:
    opens = [as_int(row, "opens") for row in aggressors if as_int(row, "opens") > 0]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bins = range(1, max(opens) + 2 if opens else 2)
    ax.hist(opens, bins=bins, color="#2f5d8a", edgecolor="white", align="left", rwidth=0.9)
    ax.set_title("Row Reopen Count Distribution")
    ax.set_xlabel("Number of ACT-to-PRE Sessions per Physical Row")
    ax.set_ylabel("Row Count")
    ax.grid(axis="y", alpha=0.25)
    if opens:
        p90 = percentile(sorted(float(v) for v in opens), 0.9)
        ax.axvline(p90, color="#d95f02", linestyle="--", linewidth=1.5, label=f"p90={p90:.1f}")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_session_lifetime_distribution(
    plt,
    sessions: list[dict[str, str]],
    output: Path,
    tck_ns: float,
    refresh_interval_cycles: int,
) -> None:
    lifetimes = [as_int(row, "lifetime_cycles") for row in sessions if row.get("lifetime_cycles", "")]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(lifetimes, bins=50, color="#4c9a73", edgecolor="white")
    ax.set_title("Row Session Open-Time Distribution")
    ax.set_xlabel("Open Time (cycles)")
    ax.set_ylabel("Session Count")
    ax.grid(axis="y", alpha=0.25)
    if lifetimes:
        p90 = percentile(sorted(float(v) for v in lifetimes), 0.9)
        p99 = percentile(sorted(float(v) for v in lifetimes), 0.99)
        refresh_us = refresh_interval_cycles * tck_ns / 1000.0
        ax.axvline(p90, color="#7570b3", linestyle="--", linewidth=1.4, label=f"p90={p90:.0f}")
        ax.axvline(p99, color="#e7298a", linestyle="--", linewidth=1.4, label=f"p99={p99:.0f}")
        ax.axvline(
            refresh_interval_cycles,
            color="#d95f02",
            linestyle=":",
            linewidth=1.8,
            label=f"refresh={refresh_interval_cycles} cyc ({refresh_us:.2f} us)",
        )
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_long_session_command_relationship(
    plt,
    sessions: list[dict[str, str]],
    output: Path,
    tck_ns: float,
    refresh_interval_cycles: int,
) -> None:
    closed = [row for row in sessions if row.get("lifetime_cycles", "")]
    if not closed:
        return

    lifetimes = sorted(as_int(row, "lifetime_cycles") for row in closed)
    threshold = percentile([float(v) for v in lifetimes], 0.95)
    long_rows = [row for row in closed if as_int(row, "lifetime_cycles") >= threshold]

    xs = [as_int(row, "lifetime_cycles") for row in long_rows]
    ys = [as_int(row, "command_count") for row in long_rows]
    rdwr = [as_int(row, "rd_count") + as_int(row, "wr_count") for row in long_rows]

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    scatter = ax.scatter(xs, ys, c=rdwr, cmap="viridis", s=22, alpha=0.8, edgecolors="none")
    ax.set_title("Long-Lived Sessions: Open Time vs Total Commands")
    ax.set_xlabel("Open Time (cycles)")
    ax.set_ylabel("Commands in ACT-to-PRE Session")
    ax.grid(alpha=0.25)
    refresh_us = refresh_interval_cycles * tck_ns / 1000.0
    ax.axvline(
        refresh_interval_cycles,
        color="#d95f02",
        linestyle=":",
        linewidth=1.8,
        label=f"refresh={refresh_interval_cycles} cyc ({refresh_us:.2f} us)",
    )
    if xs:
        ax.axhline(sum(ys) / len(ys), color="#1b9e77", linestyle="--", linewidth=1.3, label="mean commands")
    ax.legend(frameon=False, loc="best")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("RD+WR Count")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot rowrisk figures from rowrisk CSV outputs.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing row_lifecycle.csv and aggressor_summary.csv")
    parser.add_argument("--output-dir", type=Path, help="Directory for generated PNG figures. Default: input dir")
    parser.add_argument("--tck-ns", type=float, default=DEFAULT_TCK_NS, help="Cycle time in ns. Default: 0.5")
    parser.add_argument(
        "--refresh-interval-cycles",
        type=int,
        default=DEFAULT_REFRESH_INTERVAL_CYCLES,
        help="Refresh interval in cycles for reference line. Default: 7800",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    lifecycle_path = input_dir / "row_lifecycle.csv"
    aggressor_path = input_dir / "aggressor_summary.csv"

    sessions = read_csv(lifecycle_path)
    aggressors = read_csv(aggressor_path)

    plt = setup_matplotlib()

    plot_row_open_count_distribution(
        plt,
        aggressors,
        output_dir / "figure_row_reopen_count_distribution.png",
    )
    plot_session_lifetime_distribution(
        plt,
        sessions,
        output_dir / "figure_rowsession_lifetime_distribution.png",
        args.tck_ns,
        args.refresh_interval_cycles,
    )
    plot_long_session_command_relationship(
        plt,
        sessions,
        output_dir / "figure_long_rowsession_lifetime_vs_command_count.png",
        args.tck_ns,
        args.refresh_interval_cycles,
    )


if __name__ == "__main__":
    main()
