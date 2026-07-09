from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
TRACE_PATH = BASE_DIR.parent / "cmd_trace" / "cmd_hbm3E.log.inst0.ch0"


def parse_rowsessions_from_trace(
    trace_path: Path,
) -> tuple[
    dict[tuple[int, int, int, int, int, int], list[int]],
    dict[tuple[int, int, int, int, int, int], list[int]],
    list[int],
]:
    open_banks: dict[
        tuple[int, int, int, int, int],
        tuple[tuple[int, int, int, int, int, int], int],
    ] = {}
    row_session_lifetimes: dict[tuple[int, int, int, int, int, int], list[int]] = defaultdict(list)
    row_act_cycles: dict[tuple[int, int, int, int, int, int], list[int]] = defaultdict(list)
    all_lifetimes: list[int] = []

    def close_bank(bank_key: tuple[int, int, int, int, int], close_cycle: int) -> None:
        existing = open_banks.pop(bank_key, None)
        if existing is None:
            return
        row_key, act_cycle = existing
        lifetime = close_cycle - act_cycle
        row_session_lifetimes[row_key].append(lifetime)
        all_lifetimes.append(lifetime)

    with trace_path.open(newline="") as f:
        reader = csv.reader(f, skipinitialspace=True)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue

            cycle = int(row[0])
            cmd = row[1].strip()
            ch = int(row[2])
            pch = int(row[3])
            rk = int(row[4])
            bg = int(row[5])
            ba = int(row[6])
            rr = int(row[7])

            bank_key = (ch, pch, rk, bg, ba)
            row_key = (ch, pch, rk, bg, ba, rr)

            if cmd == "ACT":
                close_bank(bank_key, cycle)
                open_banks[bank_key] = (row_key, cycle)
                row_act_cycles[row_key].append(cycle)
            elif cmd in {"PRE", "RDA", "WRA"}:
                close_bank(bank_key, cycle)
            elif cmd == "PREA":
                for existing_bank_key in list(open_banks):
                    if existing_bank_key[:3] == (ch, pch, rk):
                        close_bank(existing_bank_key, cycle)

    return row_session_lifetimes, row_act_cycles, all_lifetimes


def plot_row_act_pre_count_distribution(
    row_session_lifetimes: dict[tuple[int, int, int, int, int, int], list[int]],
    output_path: Path,
) -> None:
    counts = Counter(len(lifetimes) for lifetimes in row_session_lifetimes.values())
    xs = sorted(counts)
    ys = [counts[x] for x in xs]

    plt.figure(figsize=(8, 5))
    plt.bar(xs, ys, width=0.8, color="#2F6B7C", edgecolor="black", linewidth=0.4)
    plt.xlabel("Same-row ACT->PRE count")
    plt.ylabel("Number of rows")
    plt.title("Distribution of ACT->PRE counts per row")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_rowsession_open_lifetime_distribution(
    all_lifetimes: list[int],
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(all_lifetimes, bins=80, color="#C97C2C", edgecolor="black", linewidth=0.35)
    plt.xlabel("Row-session open lifetime (cycles)")
    plt.ylabel("Number of row sessions")
    plt.title("Distribution of row-session open lifetimes")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def detect_hot_row_band(
    row_session_lifetimes: dict[tuple[int, int, int, int, int, int], list[int]],
) -> tuple[int, int]:
    unique_counts = sorted({len(lifetimes) for lifetimes in row_session_lifetimes.values()})
    if not unique_counts:
        raise ValueError("No row-session counts found.")
    if len(unique_counts) == 1:
        return unique_counts[0], unique_counts[0]

    gaps = []
    for prev_count, next_count in zip(unique_counts, unique_counts[1:]):
        gaps.append((next_count - prev_count, prev_count, next_count))

    largest_gap, _, hot_start = max(gaps, key=lambda item: item[0])
    if largest_gap <= 1:
        hot_start = unique_counts[max(0, int(len(unique_counts) * 0.9))]

    return hot_start, unique_counts[-1]


def select_hot_rows(
    row_session_lifetimes: dict[tuple[int, int, int, int, int, int], list[int]],
    min_count: int,
    max_count: int,
) -> dict[tuple[int, int, int, int, int, int], list[int]]:
    return {
        row_key: lifetimes
        for row_key, lifetimes in row_session_lifetimes.items()
        if min_count <= len(lifetimes) <= max_count
    }


def plot_hot_row_lifetime_distribution(
    hot_rows: dict[tuple[int, int, int, int, int, int], list[int]],
    hot_min: int,
    hot_max: int,
    output_path: Path,
) -> None:
    lifetimes = [lifetime for values in hot_rows.values() for lifetime in values]

    plt.figure(figsize=(8, 5))
    plt.hist(lifetimes, bins=80, color="#B24C63", edgecolor="black", linewidth=0.35)
    plt.xlabel("Row-session open lifetime (cycles)")
    plt.ylabel("Number of hot-row sessions")
    plt.title(
        f"Open lifetime distribution for hot rows ({hot_min}-{hot_max} ACT->PRE events)"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_hot_row_average_inter_act_interval_distribution(
    hot_rows: dict[tuple[int, int, int, int, int, int], list[int]],
    row_act_cycles: dict[tuple[int, int, int, int, int, int], list[int]],
    hot_min: int,
    hot_max: int,
    output_path: Path,
) -> None:
    average_intervals: list[float] = []
    for row_key in hot_rows:
        act_cycles = row_act_cycles[row_key]
        intervals = [next_cycle - prev_cycle for prev_cycle, next_cycle in zip(act_cycles, act_cycles[1:])]
        if intervals:
            average_intervals.append(sum(intervals) / len(intervals))

    plt.figure(figsize=(8, 5))
    plt.hist(average_intervals, bins=80, color="#5C8D3A", edgecolor="black", linewidth=0.35)
    plt.xlabel("Average inter-ACT interval per hot row (cycles)")
    plt.ylabel("Number of hot rows")
    plt.title(
        f"Average inter-ACT interval per hot row ({hot_min}-{hot_max} ACT->PRE events)"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_summary(
    row_session_lifetimes: dict[tuple[int, int, int, int, int, int], list[int]],
    hot_rows: dict[tuple[int, int, int, int, int, int], list[int]],
    row_act_cycles: dict[tuple[int, int, int, int, int, int], list[int]],
    hot_min: int,
    hot_max: int,
    output_path: Path,
) -> None:
    hot_lifetimes = sorted(lifetime for values in hot_rows.values() for lifetime in values)
    hot_counts = sorted(len(values) for values in hot_rows.values())
    intervals: list[int] = []
    average_intervals: list[float] = []
    for row_key in hot_rows:
        act_cycles = row_act_cycles[row_key]
        row_intervals = [next_cycle - prev_cycle for prev_cycle, next_cycle in zip(act_cycles, act_cycles[1:])]
        intervals.extend(row_intervals)
        if row_intervals:
            average_intervals.append(sum(row_intervals) / len(row_intervals))
    intervals.sort()
    average_intervals.sort()

    def percentile(values: list[float], ratio: float) -> float | None:
        if not values:
            return None
        index = min(len(values) - 1, max(0, int((len(values) - 1) * ratio)))
        return values[index]

    with output_path.open("w") as f:
        f.write(f"trace_path={TRACE_PATH}\n")
        f.write(f"total_rows={len(row_session_lifetimes)}\n")
        f.write(f"hot_row_min={hot_min}\n")
        f.write(f"hot_row_max={hot_max}\n")
        f.write(f"selected_hot_rows={len(hot_rows)}\n")
        if hot_counts:
            f.write(f"selected_hot_row_count_min={hot_counts[0]}\n")
            f.write(f"selected_hot_row_count_max={hot_counts[-1]}\n")
        f.write(f"selected_hot_sessions={len(hot_lifetimes)}\n")
        if hot_lifetimes:
            f.write(f"hot_lifetime_p50={percentile(hot_lifetimes, 0.50)}\n")
            f.write(f"hot_lifetime_p90={percentile(hot_lifetimes, 0.90)}\n")
            f.write(f"hot_lifetime_p99={percentile(hot_lifetimes, 0.99)}\n")
            f.write(f"hot_lifetime_max={hot_lifetimes[-1]}\n")
        if intervals:
            f.write(f"hot_inter_act_interval_count={len(intervals)}\n")
            f.write(f"hot_inter_act_interval_p50={percentile(intervals, 0.50)}\n")
            f.write(f"hot_inter_act_interval_p90={percentile(intervals, 0.90)}\n")
            f.write(f"hot_inter_act_interval_p99={percentile(intervals, 0.99)}\n")
            f.write(f"hot_inter_act_interval_max={intervals[-1]}\n")
        if average_intervals:
            f.write(f"hot_avg_inter_act_per_row_p50={percentile(average_intervals, 0.50)}\n")
            f.write(f"hot_avg_inter_act_per_row_p90={percentile(average_intervals, 0.90)}\n")
            f.write(f"hot_avg_inter_act_per_row_p99={percentile(average_intervals, 0.99)}\n")
            f.write(f"hot_avg_inter_act_per_row_max={average_intervals[-1]}\n")


def main() -> None:
    row_session_lifetimes, row_act_cycles, all_lifetimes = parse_rowsessions_from_trace(TRACE_PATH)
    hot_min, hot_max = detect_hot_row_band(row_session_lifetimes)
    hot_rows = select_hot_rows(row_session_lifetimes, hot_min, hot_max)

    plot_row_act_pre_count_distribution(
        row_session_lifetimes,
        BASE_DIR / "figure_row_act_pre_count_distribution.png",
    )
    plot_rowsession_open_lifetime_distribution(
        all_lifetimes,
        BASE_DIR / "figure_rowsession_open_lifetime_distribution.png",
    )
    plot_hot_row_lifetime_distribution(
        hot_rows,
        hot_min,
        hot_max,
        BASE_DIR / "figure_hot_rows_lifetime_distribution.png",
    )
    plot_hot_row_average_inter_act_interval_distribution(
        hot_rows,
        row_act_cycles,
        hot_min,
        hot_max,
        BASE_DIR / "figure_hot_rows_average_inter_act_interval_distribution.png",
    )
    write_summary(
        row_session_lifetimes,
        hot_rows,
        row_act_cycles,
        hot_min,
        hot_max,
        BASE_DIR / "plot_summary.txt",
    )


if __name__ == "__main__":
    main()
