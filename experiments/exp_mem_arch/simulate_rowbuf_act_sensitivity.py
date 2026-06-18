#!/usr/bin/env python3
"""Post-process ACT sensitivity for reorder-on energy rows.

This is a post-process what-if analysis: it changes ACT count/energy to hit a
target RD+WR-per-ACT or READ-per-ACT ratio while keeping READ/WRITE traffic and
MAC energy unchanged.

The default row-episode timing model also rescales memory duration, REF, and
background energy. If accesses per ACT decrease, each open-row episode has fewer
column commands; elapsed memory time is therefore estimated from fixed column
traffic plus per-row open/close overhead rather than held constant or scaled
directly with ACT count.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXP_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = EXP_DIR / "data" / "energy_breakdown_ramulator_on_drampower_ref.csv"
DEFAULT_REPORT = EXP_DIR / "ROWBUF_ACT5_WHATIF.md"
DEFAULT_CSV = EXP_DIR / "data" / "rowbuf_act5_whatif.csv"
DEFAULT_PLOT_CSV = EXP_DIR / "data" / "rowbuf_act5_energy_breakdown_for_plot.csv"


@dataclass(frozen=True)
class TimingSpec:
    n_rcd: float = 28.0
    n_ras: float = 68.0
    n_rp: float = 28.0
    n_ccd: float = 4.0
    n_rtps: float = 8.0
    n_cwl: float = 8.0
    n_bl: float = 2.0
    n_wr: float = 32.0


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def load_rows(path: Path, reorder: str) -> list[dict[str, str]]:
    with path.open("r", newline="") as src:
        rows = [row for row in csv.DictReader(src) if row["reorder"] == reorder]
    rows.sort(key=lambda row: (int(row["seq_len"]), int(row["batch_per_gpu"])))
    return rows


def access_count(row: dict[str, str], metric: str) -> float:
    read, write = read_write_counts(row)
    if metric == "read":
        return read
    if metric == "rdwr":
        return read + write
    raise ValueError(f"Unsupported metric: {metric}")


def read_write_counts(row: dict[str, str]) -> tuple[float, float]:
    read = f(row, "read_count_avg") + f(row, "all_read_count_avg")
    write = f(row, "write_count_avg") + f(row, "all_write_count_avg")
    return read, write


def act_count(row: dict[str, str]) -> float:
    return f(row, "act_count_avg") + f(row, "all_act_count_avg")


def post_column_to_pre_cycles(read_count: float, write_count: float, spec: TimingSpec) -> float:
    total = read_count + write_count
    if total <= 0:
        return spec.n_rtps
    write_fraction = write_count / total
    read_post = spec.n_rtps
    write_post = spec.n_cwl + spec.n_bl + spec.n_wr
    return read_post * (1.0 - write_fraction) + write_post * write_fraction


def row_episode_cycles(accesses_per_act: float, read_count: float, write_count: float, spec: TimingSpec) -> float:
    k = max(accesses_per_act, 1.0)
    column_span = max(0.0, k - 1.0) * spec.n_ccd
    post_column = post_column_to_pre_cycles(read_count, write_count, spec)
    return max(spec.n_ras, spec.n_rcd + column_span + post_column) + spec.n_rp


def timing_scale(
    row: dict[str, str],
    old_act_count: float,
    new_act_count: float,
    timing_model: str,
    spec: TimingSpec,
) -> dict[str, float]:
    read, write = read_write_counts(row)
    total_accesses = read + write
    if timing_model == "event-only" or total_accesses <= 0 or old_act_count <= 0 or new_act_count <= 0:
        old_k = total_accesses / old_act_count if old_act_count else 0.0
        new_k = total_accesses / new_act_count if new_act_count else 0.0
        return {
            "old_total_cols_per_act": old_k,
            "new_total_cols_per_act": new_k,
            "old_episode_cycles": 0.0,
            "new_episode_cycles": 0.0,
            "duration_scale": 1.0,
        }

    if timing_model != "row-episode":
        raise ValueError(f"Unsupported timing model: {timing_model}")

    old_k = total_accesses / old_act_count
    new_k = total_accesses / new_act_count
    old_episode = row_episode_cycles(old_k, read, write, spec)
    new_episode = row_episode_cycles(new_k, read, write, spec)
    old_cycles_per_access = old_episode / max(old_k, 1e-9)
    new_cycles_per_access = new_episode / max(new_k, 1e-9)
    return {
        "old_total_cols_per_act": old_k,
        "new_total_cols_per_act": new_k,
        "old_episode_cycles": old_episode,
        "new_episode_cycles": new_episode,
        "duration_scale": new_cycles_per_access / old_cycles_per_access,
    }


def recompute_row(
    row: dict[str, str],
    target: float,
    metric: str,
    mode: str,
    uniform_scale: float | None,
    timing_model: str,
    timing_spec: TimingSpec,
) -> dict[str, Any]:
    old_act_count = act_count(row)
    accesses = access_count(row, metric)
    old_ratio = accesses / old_act_count
    act_scale = uniform_scale if mode == "weighted-average" else old_ratio / target
    new_act_count = old_act_count * act_scale
    timing = timing_scale(row, old_act_count, new_act_count, timing_model, timing_spec)

    old_act_j = f(row, "act_J_step") + f(row, "all_act_J_step")
    new_act_j = old_act_j * act_scale
    fixed_access_j = (
        f(row, "read_J_step")
        + f(row, "write_J_step")
        + f(row, "all_read_J_step")
        + f(row, "all_write_J_step")
    )
    old_ref_j = f(row, "ref_J_step")
    old_bg_j = f(row, "background_J_step")
    new_ref_j = old_ref_j * timing["duration_scale"]
    new_bg_j = old_bg_j * timing["duration_scale"]
    old_dram_j = f(row, "dram_total_J_step")
    new_dram_j = new_act_j + fixed_access_j + new_ref_j + new_bg_j
    mac_j = f(row, "mac_J_step")
    tokens = f(row, "tokens_per_step")
    old_total_plus_mac_j = f(row, "total_plus_mac_J_step")
    new_total_plus_mac_j = new_dram_j + mac_j

    return {
        "source_result": row["source_result"],
        "reorder": row["reorder"],
        "seq_len": int(row["seq_len"]),
        "batch_per_gpu": int(row["batch_per_gpu"]),
        "target_metric": metric,
        "target_accesses_per_act": target,
        "adjustment_mode": mode,
        "old_accesses_per_act": old_ratio,
        "new_accesses_per_act": accesses / new_act_count if new_act_count else 0.0,
        "old_total_cols_per_act": timing["old_total_cols_per_act"],
        "new_total_cols_per_act": timing["new_total_cols_per_act"],
        "act_count_old": old_act_count,
        "act_count_new": new_act_count,
        "act_count_scale": act_scale,
        "timing_model": timing_model,
        "old_episode_cycles": timing["old_episode_cycles"],
        "new_episode_cycles": timing["new_episode_cycles"],
        "duration_scale": timing["duration_scale"],
        "memory_duration_ms_old": f(row, "memory_duration_ms"),
        "memory_duration_ms_new": f(row, "memory_duration_ms") * timing["duration_scale"],
        "background_time_ms_old": f(row, "background_time_ms"),
        "background_time_ms_new": f(row, "background_time_ms") * timing["duration_scale"],
        "ref_count_old": f(row, "ref_count_avg"),
        "ref_count_new": f(row, "ref_count_avg") * timing["duration_scale"],
        "act_J_step_old": old_act_j,
        "act_J_step_new": new_act_j,
        "ref_J_step_old": old_ref_j,
        "ref_J_step_new": new_ref_j,
        "background_J_step_old": old_bg_j,
        "background_J_step_new": new_bg_j,
        "dram_total_J_step_old": old_dram_j,
        "dram_total_J_step_new": new_dram_j,
        "dram_total_J_step_delta": new_dram_j - old_dram_j,
        "dram_total_pct_delta": (new_dram_j / old_dram_j - 1.0) * 100.0,
        "mac_J_step": mac_j,
        "total_plus_mac_J_step_old": old_total_plus_mac_j,
        "total_plus_mac_J_step_new": new_total_plus_mac_j,
        "total_plus_mac_pct_delta": (new_total_plus_mac_j / old_total_plus_mac_j - 1.0) * 100.0,
        "dram_total_J_token_old": old_dram_j / tokens,
        "dram_total_J_token_new": new_dram_j / tokens,
    }


def recompute_energy_row(
    row: dict[str, str],
    target: float,
    metric: str,
    mode: str,
    uniform_scale: float | None,
    timing_model: str,
    timing_spec: TimingSpec,
) -> dict[str, Any]:
    """Return a row compatible with plot_dram_command_breakdown.py."""
    if row["reorder"] != "on":
        return dict(row)

    old_act_count = act_count(row)
    accesses = access_count(row, metric)
    old_ratio = accesses / old_act_count
    act_scale = uniform_scale if mode == "weighted-average" else old_ratio / target
    new_act_count = old_act_count * act_scale
    timing = timing_scale(row, old_act_count, new_act_count, timing_model, timing_spec)

    out: dict[str, Any] = dict(row)
    out["source_result"] = f"{row['source_result']}::rowbuf_act_target"
    out["act_count_avg"] = f(row, "act_count_avg") * act_scale
    out["all_act_count_avg"] = f(row, "all_act_count_avg") * act_scale
    out["act_J_step"] = f(row, "act_J_step") * act_scale
    out["all_act_J_step"] = f(row, "all_act_J_step") * act_scale
    out["memory_duration_ms"] = f(row, "memory_duration_ms") * timing["duration_scale"]
    out["background_time_ms"] = f(row, "background_time_ms") * timing["duration_scale"]
    out["ref_count_avg"] = f(row, "ref_count_avg") * timing["duration_scale"]
    out["ref_J_step"] = f(row, "ref_J_step") * timing["duration_scale"]
    out["background_J_step"] = f(row, "background_J_step") * timing["duration_scale"]
    out["dram_total_J_step"] = (
        out["act_J_step"]
        + f(row, "read_J_step")
        + f(row, "write_J_step")
        + out["all_act_J_step"]
        + f(row, "all_read_J_step")
        + f(row, "all_write_J_step")
        + out["ref_J_step"]
        + out["background_J_step"]
    )
    out["total_plus_mac_J_step"] = out["dram_total_J_step"] + f(row, "mac_J_step")
    tokens = f(row, "tokens_per_step")
    out["dram_total_J_token"] = out["dram_total_J_step"] / tokens
    out["total_plus_mac_J_token"] = out["total_plus_mac_J_step"] / tokens

    dram_components = [
        ("act_J_step", "act_pct_dram"),
        ("read_J_step", "read_pct_dram"),
        ("write_J_step", "write_pct_dram"),
        ("all_act_J_step", "all_act_pct_dram"),
        ("all_read_J_step", "all_read_pct_dram"),
        ("all_write_J_step", "all_write_pct_dram"),
        ("ref_J_step", "ref_pct_dram"),
        ("background_J_step", "background_pct_dram"),
    ]
    total_components = [
        ("act_J_step", "act_pct_total_plus_mac"),
        ("read_J_step", "read_pct_total_plus_mac"),
        ("write_J_step", "write_pct_total_plus_mac"),
        ("ref_J_step", "ref_pct_total_plus_mac"),
        ("background_J_step", "background_pct_total_plus_mac"),
        ("mac_J_step", "mac_pct_total_plus_mac"),
    ]
    for energy_key, pct_key in dram_components:
        out[pct_key] = 0.0 if out["dram_total_J_step"] <= 0 else float(out[energy_key]) / out["dram_total_J_step"] * 100.0
    for energy_key, pct_key in total_components:
        out[pct_key] = (
            0.0
            if out["total_plus_mac_J_step"] <= 0
            else float(out[energy_key]) / out["total_plus_mac_J_step"] * 100.0
        )
    return out


def compute(
    rows: list[dict[str, str]],
    target: float,
    metric: str,
    mode: str,
    timing_model: str,
    timing_spec: TimingSpec,
) -> list[dict[str, Any]]:
    uniform_scale = None
    if mode == "weighted-average":
        total_accesses = sum(access_count(row, metric) for row in rows)
        total_act = sum(act_count(row) for row in rows)
        uniform_scale = (total_accesses / total_act) / target
    return [recompute_row(row, target, metric, mode, uniform_scale, timing_model, timing_spec) for row in rows]


def compute_plot_rows(
    all_rows: list[dict[str, str]],
    target: float,
    metric: str,
    mode: str,
    target_reorder: str,
    timing_model: str,
    timing_spec: TimingSpec,
) -> list[dict[str, Any]]:
    target_rows = [row for row in all_rows if row["reorder"] == target_reorder]
    uniform_scale = None
    if mode == "weighted-average":
        total_accesses = sum(access_count(row, metric) for row in target_rows)
        total_act = sum(act_count(row) for row in target_rows)
        uniform_scale = (total_accesses / total_act) / target
    return [
        recompute_energy_row(row, target, metric, mode, uniform_scale, timing_model, timing_spec)
        for row in all_rows
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_all_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as src:
        rows = list(csv.DictReader(src))
    rows.sort(key=lambda row: (row["reorder"], int(row["seq_len"]), int(row["batch_per_gpu"])))
    return rows


def fmt(value: float) -> str:
    return f"{value:.3f}"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(EXP_DIR))
    except ValueError:
        return str(path)


def report_lines(
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, str]],
    *,
    input_path: Path,
    target: float,
    metric: str,
    mode: str,
    timing_model: str,
    timing_spec: TimingSpec,
) -> list[str]:
    total_old_dram = sum(row["dram_total_J_step_old"] for row in rows)
    total_new_dram = sum(row["dram_total_J_step_new"] for row in rows)
    total_old = sum(row["total_plus_mac_J_step_old"] for row in rows)
    total_new = sum(row["total_plus_mac_J_step_new"] for row in rows)
    baseline_weighted = sum(access_count(row, metric) for row in baseline_rows) / sum(act_count(row) for row in baseline_rows)
    weighted_duration_scale = (
        sum(row["memory_duration_ms_new"] for row in rows)
        / sum(row["memory_duration_ms_old"] for row in rows)
    )
    total_old_ref_bg = sum(row["ref_J_step_old"] + row["background_J_step_old"] for row in rows)
    total_new_ref_bg = sum(row["ref_J_step_new"] + row["background_J_step_new"] for row in rows)

    metric_label = "READ+WRITE" if metric == "rdwr" else "READ"
    method_line = (
        "- Method: increase ACT count/energy to hit the target; keep READ, WRITE, and MAC energy fixed; "
        "rescale memory duration, REF, and background energy using the row-episode timing model."
        if timing_model == "row-episode"
        else "- Method: change ACT count and ACT energy only; keep READ, WRITE, REF, background, MAC, latency, and memory duration unchanged."
    )
    lines = [
        "# Reorder-on ACT Sensitivity What-if",
        "",
        "## Setup",
        "",
        f"- Input: `{display_path(input_path)}`",
        "- Scope: `reorder=on`, `ramulator=on`, `type=t2t` rows already aggregated in the input CSV.",
        f"- Target: `{metric_label}/ACT = {target:g}`.",
        f"- Adjustment mode: `{mode}`.",
        f"- Timing model: `{timing_model}`.",
        method_line,
        "- Interpretation: post-process timing-aware approximation, not a full Ramulator re-simulation.",
        f"- Row episode cycles: `max(nRAS, nRCD + (k-1)*nCCD + post_col_to_pre) + nRP`, with nRCD={timing_spec.n_rcd:g}, nRAS={timing_spec.n_ras:g}, nRP={timing_spec.n_rp:g}, nCCD={timing_spec.n_ccd:g}.",
        "",
        "## Summary",
        "",
        f"- Baseline weighted `{metric_label}/ACT`: **{baseline_weighted:.3f}**.",
        f"- Weighted memory-duration scale: **{weighted_duration_scale:.3f}x**.",
        f"- REF+background: **{total_old_ref_bg:.3f} J/step** -> **{total_new_ref_bg:.3f} J/step**.",
        f"- DRAM total: **{total_old_dram:.3f} J/step** -> **{total_new_dram:.3f} J/step** (**{(total_new_dram / total_old_dram - 1.0) * 100.0:.1f}%**).",
        f"- Total+MAC: **{total_old:.3f} J/step** -> **{total_new:.3f} J/step** (**{(total_new / total_old - 1.0) * 100.0:.1f}%**).",
        "",
        "## Generated Figures",
        "",
        "- DRAM-only absolute heatmap: `plots/figure_rowbuf_act5_dram_only_command_energy_heatmaps.png`",
        "- DRAM-only relative heatmap: `plots/figure_rowbuf_act5_dram_only_command_share_heatmaps.png`",
        "- Total+MAC absolute heatmap: `plots/figure_rowbuf_act5_dram_command_energy_heatmaps.png`",
        "- Total+MAC relative heatmap: `plots/figure_rowbuf_act5_dram_command_share_heatmaps.png`",
        "",
        "## Per Configuration",
        "",
        "| Seq | Batch/GPU | Old accesses/ACT | New accesses/ACT | ACT scale | Time scale | Old DRAM J | New DRAM J | DRAM delta | Old Total+MAC J | New Total+MAC J | Total+MAC delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['seq_len']} | {row['batch_per_gpu']} | "
            f"{fmt(row['old_accesses_per_act'])} | {fmt(row['new_accesses_per_act'])} | "
            f"{fmt(row['act_count_scale'])} | "
            f"{fmt(row['duration_scale'])} | "
            f"{fmt(row['dram_total_J_step_old'])} | {fmt(row['dram_total_J_step_new'])} | "
            f"{row['dram_total_pct_delta']:.1f}% | "
            f"{fmt(row['total_plus_mac_J_step_old'])} | {fmt(row['total_plus_mac_J_step_new'])} | "
            f"{row['total_plus_mac_pct_delta']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Lowering accesses/ACT from the baseline requires more ACT commands for the same READ/WRITE traffic.",
            "- The row-episode model accounts for the fact that lower accesses/ACT also shortens each opened-row column-service episode; it does not simply multiply time by ACT scale.",
            "- REF and background energy are scaled by the estimated memory-duration change; READ, WRITE, and MAC energy are unchanged.",
            "- The model does not capture real queueing, bank-level overlap, row conflicts, refresh scheduling, or read/write turnaround. A full experiment would need Ramulator to emit or enforce a different access trace/address mapping and rerun timing.",
            "",
        ]
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--target", type=float, default=5.0)
    parser.add_argument("--metric", choices=["rdwr", "read"], default="rdwr")
    parser.add_argument("--mode", choices=["per-config", "weighted-average"], default="per-config")
    parser.add_argument("--reorder", choices=["on", "off"], default="on")
    parser.add_argument("--timing-model", choices=["row-episode", "event-only"], default="row-episode")
    parser.add_argument("--n-rcd", type=float, default=TimingSpec.n_rcd)
    parser.add_argument("--n-ras", type=float, default=TimingSpec.n_ras)
    parser.add_argument("--n-rp", type=float, default=TimingSpec.n_rp)
    parser.add_argument("--n-ccd", type=float, default=TimingSpec.n_ccd)
    parser.add_argument("--n-rtps", type=float, default=TimingSpec.n_rtps)
    parser.add_argument("--n-cwl", type=float, default=TimingSpec.n_cwl)
    parser.add_argument("--n-bl", type=float, default=TimingSpec.n_bl)
    parser.add_argument("--n-wr", type=float, default=TimingSpec.n_wr)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--plot-csv-output", type=Path, default=DEFAULT_PLOT_CSV)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    timing_spec = TimingSpec(
        n_rcd=args.n_rcd,
        n_ras=args.n_ras,
        n_rp=args.n_rp,
        n_ccd=args.n_ccd,
        n_rtps=args.n_rtps,
        n_cwl=args.n_cwl,
        n_bl=args.n_bl,
        n_wr=args.n_wr,
    )

    baseline_rows = load_rows(args.input, args.reorder)
    if not baseline_rows:
        raise SystemExit(f"No rows found for reorder={args.reorder} in {args.input}")
    rows = compute(baseline_rows, args.target, args.metric, args.mode, args.timing_model, timing_spec)
    plot_rows = compute_plot_rows(
        load_all_rows(args.input),
        args.target,
        args.metric,
        args.mode,
        args.reorder,
        args.timing_model,
        timing_spec,
    )
    write_csv(args.csv_output, rows)
    write_csv(args.plot_csv_output, plot_rows)
    args.report_output.write_text(
        "\n".join(
            report_lines(
                rows,
                baseline_rows,
                input_path=args.input,
                target=args.target,
                metric=args.metric,
                mode=args.mode,
                timing_model=args.timing_model,
                timing_spec=timing_spec,
            )
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.csv_output}")
    print(f"Wrote {args.plot_csv_output}")
    print(f"Wrote {args.report_output}")


if __name__ == "__main__":
    main()
