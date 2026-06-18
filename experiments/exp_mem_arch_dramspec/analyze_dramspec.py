#!/usr/bin/env python3
"""Analyze DRAMSpec-calibrated HBM3E-like experiment results."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

EXP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXP_DIR.parents[1]
DATA_DIR = EXP_DIR / "data"
SUMMARY_CSV = DATA_DIR / "summary_dramspec_hbm3e_like.csv"
REPORT_PATH = EXP_DIR / "DRAMSPEC_HBM3E_LIKE_ANALYSIS.md"
DRAMSPEC_SUMMARY = EXP_DIR / "generated" / "dramspec_hbm3e_like_summary.json"
BASELINE_SUMMARY = PROJECT_DIR / "experiments" / "exp_mem_arch" / "data" / "summary_hbm3e.csv"

def env_list(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    return value.split() if value else default


BATCH_PER_GPU = [int(value) for value in env_list("BATCH_SIZES", ["32", "64", "128", "256"])]
SEQ_LENGTHS = [int(value) for value in env_list("SEQ_LENGTHS", ["2048", "4096", "8192"])]
REORDERING_MODES = env_list("REORDERING_MODES", ["on"])
RAMULATOR_MODES = env_list("RAMULATOR_MODES", ["on", "off"])

COUNT_FIELDS = [
    "act_count",
    "read_count",
    "write_count",
    "all_act_count",
    "all_read_count",
    "all_write_count",
    "ref_count",
]
DRAMPOWER_FIELDS = [
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


def parse_result_name(path: Path) -> dict[str, int | str] | None:
    parts = path.stem.split("_")
    try:
        batch = int(next(part[1:] for part in parts if part.startswith("b")))
        seq_len = int(next(part[1:] for part in parts if part.startswith("l")))
        reorder = parts[parts.index("reorder") + 1]
        ramulator = parts[parts.index("ramul") + 1]
    except (IndexError, StopIteration, ValueError):
        return None
    return {
        "memory_type": "hbm3e_dramspec",
        "batch_size": batch,
        "seq_len": seq_len,
        "reorder": reorder,
        "ramulator": ramulator,
    }


def summarize_file(path: Path) -> dict[str, Any] | None:
    meta = parse_result_name(path)
    if meta is None:
        return None
    rows = read_csv_rows(path)
    avg = average_rows(rows, "t2t")
    if not avg:
        return None
    model_name = next((row.get("dram_energy_model", "") for row in rows if row.get("dram_energy_model")), "")
    latency_ns = avg.get("latency", avg.get("time", 0.0))
    out: dict[str, Any] = {
        **meta,
        "dram_energy_model": model_name,
        "latency_ns": latency_ns,
        "latency_ms": latency_ns / 1e6,
        "sim_batchsize": avg.get("batchsize", 0.0),
        "seqlen": avg.get("seqlen", 0.0),
        "memory_duration_ns": avg.get("memory_duration", 0.0),
        "background_time_ns": avg.get("background_time", 0.0),
        "mac_energy_nJ": avg.get("mac_energy", 0.0),
        "total_memory_used_bytes": avg.get("total_memory_used", 0.0),
        "memory_utilization_pct": avg.get("memory_utilization", 0.0),
    }
    for field in COUNT_FIELDS:
        out[field] = avg.get(field, 0.0)
    for field in DRAMPOWER_FIELDS:
        out[f"{field}_nJ"] = avg.get(field, 0.0)
    return out


def collect_results() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected = {
        (reorder, ramulator, seq, batch)
        for reorder in REORDERING_MODES
        for ramulator in RAMULATOR_MODES
        for seq in SEQ_LENGTHS
        for batch in BATCH_PER_GPU
    }
    for path in sorted(DATA_DIR.glob("result_hbm3e_dramspec_b*_l*_reorder_*_ramul_*.csv")):
        row = summarize_file(path)
        if row is not None:
            key = (
                str(row["reorder"]),
                str(row["ramulator"]),
                int(row["seq_len"]),
                int(row["batch_size"]),
            )
            if key not in expected:
                continue
            rows.append(row)
    rows.sort(key=lambda r: (r["reorder"], r["ramulator"], r["seq_len"], r["batch_size"]))
    return rows


def write_summary_csv(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_baseline_rows() -> dict[tuple[str, str, int, int], dict[str, Any]]:
    if not BASELINE_SUMMARY.exists():
        return {}
    out: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    with BASELINE_SUMMARY.open("r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("drampower") != "on":
                continue
            try:
                key = (
                    row["reorder"],
                    row["ramulator"],
                    int(row["seq_len"]),
                    int(row["batch_size"]),
                )
            except (KeyError, ValueError):
                continue
            out[key] = row
    return out


def ratio(value: float, base: float) -> float:
    return value / base if base else 0.0


def missing() -> list[str]:
    missing_files: list[str] = []
    for reorder in REORDERING_MODES:
        for seq in SEQ_LENGTHS:
            for batch in BATCH_PER_GPU:
                for ramul in RAMULATOR_MODES:
                    name = f"result_hbm3e_dramspec_b{batch}_l{seq}_reorder_{reorder}_ramul_{ramul}.csv"
                    if not (DATA_DIR / name).exists():
                        missing_files.append(name)
    return missing_files


def write_report(rows: list[dict[str, Any]]) -> None:
    summary = json.loads(DRAMSPEC_SUMMARY.read_text()) if DRAMSPEC_SUMMARY.exists() else {}
    timing_ns = summary.get("timing_ns_from_dramspec", {})
    current = summary.get("current_mA_from_dramspec", {})
    baseline_rows = load_baseline_rows()
    comparison_rows: list[tuple[dict[str, Any], dict[str, Any], float, float]] = []
    for row in rows:
        key = (
            str(row["reorder"]),
            str(row["ramulator"]),
            int(row["seq_len"]),
            int(row["batch_size"]),
        )
        base = baseline_rows.get(key)
        if not base:
            continue
        base_latency = float(base.get("latency_ms", 0.0) or 0.0)
        base_energy = float(base.get("drampower_total_energy_nJ", 0.0) or 0.0) / 1e9
        current_energy = float(row.get("drampower_total_energy_nJ", 0.0) or 0.0) / 1e9
        comparison_rows.append(
            (row, base, ratio(float(row["latency_ms"]), base_latency), ratio(current_energy, base_energy))
        )
    lines = [
        "# DRAMSpec-Calibrated HBM3E-Like Experiment",
        "",
        "This experiment keeps the B200/HBM3E system target but replaces the memory timing and DRAMPower-style current parameters with a DRAMSpec-calibrated HBM3E-like configuration.",
        "",
        "## Parameter Source",
        "",
        "- DRAMSpec technology input: `inputs/tech_hbm3e_calibrated_10nm.json`",
        "- DRAMSpec architecture input: `inputs/arch_hbm3e_like_b200_24gb_8gbps.json`",
        "- Generated Ramulator config: `generated/dram_config_HBM3E_DRAMSpec.yaml`",
        "- Generated DRAMPower-style config: `generated/dramspec_hbm3e_like_power.yaml`",
        f"- Included modes: reorder={','.join(REORDERING_MODES)}, ramulator={','.join(RAMULATOR_MODES)}.",
        "- Scope: calibrated HBM3E-like model, not vendor datasheet-level HBM3E.",
        "",
        "## DRAMSpec Output Snapshot",
        "",
        "| Parameter | Value |",
        "|---|---:|",
    ]
    for key in ["trcd", "tcl", "tras", "trp", "trc", "twr", "trfc", "trefI"]:
        if key in timing_ns:
            lines.append(f"| {key} ns | {timing_ns[key]:.4f} |")
    for key in ["IDD0", "IDD2n", "IDD3n", "IDD4R", "IDD4W", "IDD5B"]:
        if key in current:
            lines.append(f"| {key} mA | {current[key]:.4f} |")
    lines.extend(["", "## Results", ""])
    if not rows:
        lines.append("No completed runs found.")
    else:
        lines.extend(
            [
                "| Reorder | Ramulator | Seq | Batch/GPU | Latency ms | DRAMPower J/step | ACT | READ | WRITE | REF |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            total_j = row.get("drampower_total_energy_nJ", 0.0) / 1e9
            lines.append(
                f"| {row['reorder']} | {row['ramulator']} | {row['seq_len']} | {row['batch_size']} | "
                f"{row['latency_ms']:.4f} | {total_j:.6f} | "
                f"{row.get('act_count', 0.0):.0f} | {row.get('read_count', 0.0):.0f} | "
                f"{row.get('write_count', 0.0):.0f} | {row.get('ref_count', 0.0):.0f} |"
            )
    if comparison_rows:
        lines.extend(
            [
                "",
                "## Baseline Comparison",
                "",
                "Baseline is `experiments/exp_mem_arch/data/summary_hbm3e.csv` filtered to `drampower=on`, i.e. the previous HBM3 timing + HBM2-derived current adapter.",
                "",
                "| Reorder | Ramulator | Seq | Batch/GPU | Latency Ratio | DRAMPower Ratio |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        highlights = {
            ("on", "on", 2048, 32),
            ("on", "on", 8192, 256),
            ("off", "on", 2048, 32),
            ("off", "on", 8192, 256),
            ("on", "off", 2048, 32),
            ("off", "off", 8192, 256),
        }
        for row, _base, latency_ratio, energy_ratio in comparison_rows:
            key = (str(row["reorder"]), str(row["ramulator"]), int(row["seq_len"]), int(row["batch_size"]))
            if key not in highlights:
                continue
            lines.append(
                f"| {row['reorder']} | {row['ramulator']} | {row['seq_len']} | {row['batch_size']} | "
                f"{latency_ratio:.3f}x | {energy_ratio:.3f}x |"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The 10nm calibrated input lowers `IDD4R/IDD4W` substantially versus the earlier 29nm HBM-derived input; treat the result as HBM3E-like until calibrated against JEDEC/vendor current tables.",
            "- Current implementation overrides the Ramulator config through `system.dram_config_path` and the DRAMPower-style adapter through `system.optimization.dram_power_config_path`.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--missing", action="store_true")
    args = parser.parse_args()

    rows = collect_results()
    write_summary_csv(rows)
    write_report(rows)
    if args.missing:
        missing_files = missing()
        if missing_files:
            print("\n".join(missing_files))
        else:
            print("All expected runs are present.")
    else:
        print(f"Wrote {SUMMARY_CSV}")
        print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
