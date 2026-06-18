#!/usr/bin/env python3
"""Run LLMSimulator traces for RoMe Figure 1 and regenerate the plot."""

from __future__ import annotations

import os
import subprocess
import json
import tempfile
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
DATA_DIR = EXP_DIR / "data"
RUN_BIN = REPO_ROOT / "build" / "run"
REPRO_SCRIPT = EXP_DIR / "reproduce_figure1.py"

SEQ_LEN = 8192


MODEL_OPTIONS = {
    "deepseekV3": {
        "precision_byte": 1,
        "compressed_kv": True,
        "use_absorb": True,
        "num_device": 8,
        # RoMe uses TP=1 for DeepSeek-V3 decode attention and expert
        # parallelism for MoE layers across the 8 accelerators.
        "none_expert_tensor_degree": 1,
        "expert_tensor_degree": 1,
    },
    "grok1": {
        "precision_byte": 2,
        "compressed_kv": False,
        "use_absorb": False,
        "num_device": 8,
        # RoMe reports Grok 1 with TP=8 for the non-expert path; routed
        # experts are distributed across accelerators rather than tensor-split.
        "none_expert_tensor_degree": 8,
        "expert_tensor_degree": 1,
    },
    "llama3_405B": {
        "precision_byte": 1,
        "compressed_kv": False,
        "use_absorb": False,
        "num_device": 8,
        "none_expert_tensor_degree": 8,
        "expert_tensor_degree": 1,
    },
}


def seq_len_for_model(model_name: str) -> int:
    # The simulator asserts input_len < max_seq_len.  Grok 1's max_seq_len is
    # exactly 8192, so use the nearest valid 8K history length for trace export.
    if model_name == "grok1":
        return 8191
    return SEQ_LEN


def base_config(model_name: str, stage: str, output_dir: Path, gantt_dir: Path) -> dict:
    options = MODEL_OPTIONS[model_name]
    seq_len = seq_len_for_model(model_name)
    dp_degree = options["num_device"] // options["none_expert_tensor_degree"]
    return {
        "model": {"model_name": model_name},
        "system": {
            "gpu_gen": "B100",
            "nvlink_gen": 5,
            "infiniband_gen": 800,
            "num_node": 1,
            "num_device": options["num_device"],
            "processor_type": "GPU",
            "distribution": {
                "expert_tensor_degree": options["expert_tensor_degree"],
                "none_expert_tensor_degree": options["none_expert_tensor_degree"],
            },
            "optimization": {
                "parallel_execution": False,
                "hetero_subbatch": False,
                "disagg_system": False,
                "use_low_unit_moe_only": False,
                "use_ramulator": False,
                "use_drampower": False,
                "dram_power_model": "hbm3e_adapter",
                "compressed_kv": options["compressed_kv"],
                "use_absorb": options["use_absorb"],
                "use_flash_mla": True,
                "use_flash_attention": True,
                "reuse_kv_cache": False,
                "kv_cache_reuse_rate": 0.0,
                "prefill_mode": stage == "prefill",
                "decode_mode": stage == "decode",
            },
        },
        "serving": {
            "max_batch_size": max(1, dp_degree),
            "max_process_token": seq_len,
        },
        "simulation": {
            "data": "synthesis",
            "input_len": seq_len,
            "output_len": 2,
            "precision_byte": options["precision_byte"],
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


def run_trace(model_name: str, stage: str, work_dir: Path) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    config_path = work_dir / f"{model_name}_{stage}.yaml"
    sim_output_dir = work_dir / "sim_outputs"
    gantt_dir = work_dir / "gantt"
    sim_output_dir.mkdir(parents=True, exist_ok=True)
    gantt_dir.mkdir(parents=True, exist_ok=True)

    trace_path = DATA_DIR / f"{model_name}_{stage}_trace.csv"
    if trace_path.exists():
        trace_path.unlink()

    config_path.write_text(
        json.dumps(base_config(model_name, stage, sim_output_dir, gantt_dir), indent=2) + "\n"
    )

    env = os.environ.copy()
    env["LLMSIM_TRACE_PATH"] = str(trace_path)
    env["LLMSIM_TRACE_STAGE"] = stage

    print(f"[trace] {model_name} {stage} -> {trace_path}")
    subprocess.run([str(RUN_BIN), str(config_path)], cwd=REPO_ROOT, env=env, check=True)
    return trace_path


def main() -> None:
    if not RUN_BIN.exists():
        raise SystemExit(f"Missing simulator binary: {RUN_BIN}. Build with `cmake --build build -j` first.")

    with tempfile.TemporaryDirectory(prefix="rome_fig1_", dir=EXP_DIR) as tmp:
        work_dir = Path(tmp)
        traces: list[Path] = []
        for model_name in MODEL_OPTIONS:
            for stage in ("prefill", "decode"):
                traces.append(run_trace(model_name, stage, work_dir))

        subprocess.run(
            ["python3", str(REPRO_SCRIPT), "--from-trace", *[str(path) for path in traces]],
            cwd=REPO_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
