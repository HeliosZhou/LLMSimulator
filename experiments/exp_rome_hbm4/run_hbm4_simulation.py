#!/usr/bin/env python3
"""
HBM4 Baseline Simulation — RoMe-style experiment setup.

Reproduces the HBM4 baseline side of RoMe Figure 12:
  - 3 models: DeepSeek-V3, Grok 1, Llama 3-405B
  - Decode stage, sequence length = 8K
  - 8 accelerators with model-specific TP/EP/DP
  - Sweep batch sizes, record TPOT

Usage:
    cd ~/LLMSimulator
    cmake --build build -j
    python3 experiments/exp_rome_hbm4/run_hbm4_simulation.py
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
DATA_DIR = EXP_DIR / "data"
RUN_BIN = REPO_ROOT / "build" / "run"

SEQ_LEN = 8192

# RoMe Figure 12 setup: same parallel strategies
MODEL_OPTIONS = {
    "deepseekV3": {
        "precision_byte": 1,
        "compressed_kv": True,
        "use_absorb": True,
        "num_device": 8,
        "none_expert_tensor_degree": 1,  # TP=1 for attention
        "expert_tensor_degree": 1,       # EP for MoE
    },
    "grok1": {
        "precision_byte": 2,
        "compressed_kv": False,
        "use_absorb": False,
        "num_device": 8,
        "none_expert_tensor_degree": 8,  # TP=8 for attention
        "expert_tensor_degree": 1,       # EP for MoE
    },
    "llama3_405B": {
        "precision_byte": 1,
        "compressed_kv": False,
        "use_absorb": False,
        "num_device": 8,
        "none_expert_tensor_degree": 8,  # TP=8
        "expert_tensor_degree": 1,
    },
}

# Batch sizes to sweep (constrained by memory capacity)
BATCH_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024]


def seq_len_for_model(model_name: str) -> int:
    if model_name == "grok1":
        return 8191  # max_seq_len is exactly 8192
    return SEQ_LEN


def make_config(
    model_name: str,
    batch_size: int,
    output_dir: Path,
    gantt_dir: Path,
) -> dict:
    """Build LLMSimulator YAML config for HBM4 baseline."""
    opts = MODEL_OPTIONS[model_name]
    seq_len = seq_len_for_model(model_name)
    dp_degree = opts["num_device"] // opts["none_expert_tensor_degree"]

    return {
        "model": {"model_name": model_name},
        "system": {
            "gpu_gen": "B200",
            "nvlink_gen": 5,
            "infiniband_gen": 800,
            "num_node": 1,
            "num_device": opts["num_device"],
            "processor_type": "GPU",
            "memory_bandwidth": 16.0e12,  # 16 TB/s → triggers HBM4 config
            "memory_capacity": 256.0 * 1024 * 1024 * 1024,  # 256 GB
            "distribution": {
                "expert_tensor_degree": opts["expert_tensor_degree"],
                "none_expert_tensor_degree": opts["none_expert_tensor_degree"],
            },
            "optimization": {
                "parallel_execution": False,
                "hetero_subbatch": False,
                "disagg_system": False,
                "use_low_unit_moe_only": False,
                "use_ramulator": True,
                "use_drampower": False,
                "dram_power_model": "hbm3e_adapter",
                "compressed_kv": opts["compressed_kv"],
                "use_absorb": opts["use_absorb"],
                "use_flash_mla": True,
                "use_flash_attention": True,
                "reuse_kv_cache": False,
                "kv_cache_reuse_rate": 0.0,
                "prefill_mode": False,
                "decode_mode": True,
            },
        },
        "serving": {
            "max_batch_size": batch_size,
            "max_process_token": seq_len,
        },
        "simulation": {
            "data": "synthesis",
            "input_len": seq_len,
            "output_len": 2,
            "precision_byte": opts["precision_byte"],
            "skewness": 0.0,
            "iter": 1,
            "injection_rate": 0,
            "exit_out_of_memory": False,
            "mem_cap_limit": False,
        },
        "log": {
            "print_log": False,
            "export_gantt": False,
            "output_directory": str(output_dir),
            "gantt_directory": str(gantt_dir),
        },
    }


def run_one(model_name: str, batch_size: int, work_dir: Path) -> dict | None:
    """Run a single simulation and return results dict, or None on OOM."""
    sim_dir = work_dir / f"{model_name}_bs{batch_size}"
    sim_dir.mkdir(parents=True, exist_ok=True)
    gantt_dir = sim_dir / "gantt"
    gantt_dir.mkdir(exist_ok=True)

    config_path = sim_dir / "config.json"
    config = make_config(model_name, batch_size, sim_dir, gantt_dir)
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    env = os.environ.copy()
    env["LLMSIM_TRACE_PATH"] = str(DATA_DIR / f"{model_name}_bs{batch_size}_trace.csv")
    env["LLMSIM_TRACE_STAGE"] = "decode"

    try:
        result = subprocess.run(
            [str(RUN_BIN), str(config_path)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {model_name} bs={batch_size}")
        return None

    if result.returncode != 0:
        # Check if OOM
        if "out of memory" in result.stderr.lower() or "SIGFPE" in result.stderr:
            print(f"  [OOM] {model_name} bs={batch_size}")
            return None
        print(f"  [ERROR] {model_name} bs={batch_size}: {result.stderr[:200]}")
        return None

    # Parse stdout for the output CSV path and total cycles
    tpot = None
    total_cycles = None

    # The simulator prints the output CSV path and total cycles
    # Look for the CSV file in sim_dir
    for f in sim_dir.glob("*.csv"):
        if "synthesis" in f.name and f.stat().st_size > 0:
            try:
                with open(f) as csvf:
                    reader = csv.DictReader(csvf)
                    for row in reader:
                        if "time" in row:
                            total_cycles = float(row["time"])
                            # output_len in CSV may be 0; use config value
                            tpot = total_cycles / 2  # output_len=2 for decode
                            break
            except Exception:
                pass
            break

    print(f"  [OK] {model_name} bs={batch_size} tpot={tpot} total_cycles={total_cycles}")
    return {
        "model": model_name,
        "batch_size": batch_size,
        "tpot": tpot,
        "total_cycles": total_cycles,
    }


def main() -> None:
    if not RUN_BIN.exists():
        raise SystemExit(
            f"Missing simulator binary: {RUN_BIN}\n"
            "Build with: cmake --build build -j"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    # Use persistent work directory so we can read results after each run
    work_dir = DATA_DIR / "sim_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    for model_name in MODEL_OPTIONS:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        for bs in BATCH_SIZES:
            print(f"\n  Batch size: {bs}")
            result = run_one(model_name, bs, work_dir)
            if result is not None:
                all_results.append(result)

    # Write results CSV
    results_csv = DATA_DIR / "hbm4_baseline_results.csv"
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(results_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults written to {results_csv}")
    else:
        print("\nNo results collected.")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary: HBM4 Baseline TPOT")
    print(f"{'='*60}")
    for model_name in MODEL_OPTIONS:
        model_results = [r for r in all_results if r["model"] == model_name]
        if model_results:
            print(f"\n{model_name}:")
            for r in model_results:
                tpot_str = f"{r['tpot']:.2f}" if r['tpot'] else "N/A"
                print(f"  bs={r['batch_size']:>3d}  TPOT={tpot_str}")


if __name__ == "__main__":
    main()
