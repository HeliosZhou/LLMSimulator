#!/usr/bin/env python3
"""Shared helpers for LLMSimulator paper experiments."""

from __future__ import annotations

import csv
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "build"
DEFAULT_CONFIG = BUILD_DIR / "config.yaml"
DEFAULT_NUM_NODE = 4
DEFAULT_NUM_DEVICE = 8
DEFAULT_NUM_GPUS = DEFAULT_NUM_NODE * DEFAULT_NUM_DEVICE


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)


def boolify_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Normalize common on/off fields to bools before editing."""
    def convert(v: Any) -> Any:
        if isinstance(v, str):
            if v.lower() in {"on", "true", "yes"}:
                return True
            if v.lower() in {"off", "false", "no"}:
                return False
        if isinstance(v, dict):
            return {k: convert(val) for k, val in v.items()}
        if isinstance(v, list):
            return [convert(val) for val in v]
        return v

    return convert(cfg)


def num_gpus(num_node: int = DEFAULT_NUM_NODE, num_device: int = DEFAULT_NUM_DEVICE) -> int:
    return num_node * num_device


def system_batch(batch_per_gpu: int, num_node: int = DEFAULT_NUM_NODE, num_device: int = DEFAULT_NUM_DEVICE) -> int:
    return batch_per_gpu * num_gpus(num_node, num_device)


@dataclass(frozen=True)
class SimPoint:
    model: str = "deepseekV3"
    gpu_gen: str = "B200"
    processor_type: str = "GPU"
    num_node: int = 1
    num_device: int = 8
    nvlink_gen: int = 5
    infiniband_gen: int = 800
    expert_tp: int = 1
    none_expert_tp: int = 1
    batch_size: int = 32
    seq_len: int = 2048
    output_len: int = 2
    precision_byte: int = 1
    iterations: int = 3
    skewness: float = 0.0
    use_absorb: bool = True
    compressed_kv: bool = True
    decode_mode: bool = True
    prefill_mode: bool = False
    use_flash_mla: bool = True
    use_flash_attention: bool = True
    reuse_kv_cache: bool = True
    kv_cache_reuse_rate: float = 0.0
    parallel_execution: bool = False
    hetero_subbatch: bool = False
    disagg_system: bool = False
    use_low_unit_moe_only: bool = False
    use_ramulator: bool = False
    max_process_token: int = 0
    mem_cap_limit: bool = False
    exit_out_of_memory: bool = False
    logic_x: int | None = None
    logic_op_b: float | None = None
    pim_x: int | None = None
    pim_op_b: float | None = None


def build_config(point: SimPoint, output_dir: Path, base_config: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg = boolify_config(load_yaml(base_config))

    cfg["model"]["model_name"] = point.model
    cfg["system"]["gpu_gen"] = point.gpu_gen
    cfg["system"]["processor_type"] = point.processor_type
    cfg["system"]["num_node"] = point.num_node
    cfg["system"]["num_device"] = point.num_device
    cfg["system"]["nvlink_gen"] = point.nvlink_gen
    cfg["system"]["infiniband_gen"] = point.infiniband_gen
    for key in ["logic_x", "logic_op_b", "pim_x", "pim_op_b"]:
        value = getattr(point, key)
        if value is not None:
            cfg["system"][key] = value
    cfg["system"]["distribution"]["expert_tensor_degree"] = point.expert_tp
    cfg["system"]["distribution"]["none_expert_tensor_degree"] = point.none_expert_tp

    opt = cfg["system"]["optimization"]
    opt["use_absorb"] = point.use_absorb
    opt["compressed_kv"] = point.compressed_kv
    opt["use_flash_mla"] = point.use_flash_mla
    opt["use_flash_attention"] = point.use_flash_attention
    opt["reuse_kv_cache"] = point.reuse_kv_cache
    opt["kv_cache_reuse_rate"] = point.kv_cache_reuse_rate
    opt["parallel_execution"] = point.parallel_execution
    opt["hetero_subbatch"] = point.hetero_subbatch
    opt["disagg_system"] = point.disagg_system
    opt["use_low_unit_moe_only"] = point.use_low_unit_moe_only
    opt["use_ramulator"] = point.use_ramulator
    opt["prefill_mode"] = point.prefill_mode
    opt["decode_mode"] = point.decode_mode

    cfg["serving"]["max_batch_size"] = point.batch_size
    cfg["serving"]["max_process_token"] = point.max_process_token

    sim = cfg["simulation"]
    sim["data"] = "synthesis"
    sim["input_len"] = point.seq_len
    sim["output_len"] = point.output_len
    sim["precision_byte"] = point.precision_byte
    sim["skewness"] = point.skewness
    sim["iter"] = point.iterations
    sim["injection_rate"] = 0
    sim["exit_out_of_memory"] = point.exit_out_of_memory
    sim["mem_cap_limit"] = point.mem_cap_limit

    cfg["log"]["print_log"] = False
    cfg["log"]["export_gantt"] = False
    cfg["log"]["output_directory"] = str(output_dir)
    return cfg


def run_simulation(
    point: SimPoint,
    output_dir: Path,
    result_name: str,
    timeout: int = 900,
    skip_existing: bool = True,
) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / result_name
    if skip_existing and dest.exists():
        return dest

    cfg = build_config(point, output_dir)
    config_dir = output_dir / "configs"
    config_path = config_dir / f"{Path(result_name).stem}.yaml"
    save_yaml(config_path, cfg)

    before = {p.resolve() for p in output_dir.glob("*.csv")}
    result = subprocess.run(
        ["./run", str(config_path)],
        cwd=BUILD_DIR,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        print(result.stdout[-1000:])
        print(result.stderr[-1000:])
        return None

    candidates = [p for p in output_dir.glob("*.csv") if p.resolve() not in before]
    if not candidates:
        candidates = sorted(output_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            return None
        latest = candidates[-1]
    else:
        latest = max(candidates, key=lambda p: p.stat().st_mtime)

    if latest.resolve() != dest.resolve():
        if dest.exists():
            dest.unlink()
        shutil.move(str(latest), str(dest))
    return dest


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], field: str, default: float = 0.0) -> float:
    try:
        return float(row.get(field, default) or default)
    except (TypeError, ValueError):
        return default


def average_rows(rows: Iterable[dict[str, str]], row_type: str = "t2t") -> dict[str, float]:
    selected = [r for r in rows if r.get("type") == row_type]
    if not selected:
        selected = list(rows)
    out: dict[str, float] = defaultdict(float)
    if not selected:
        return out
    for row in selected:
        for key, val in row.items():
            try:
                out[key] += float(val)
            except (TypeError, ValueError):
                pass
    for key in list(out):
        out[key] /= len(selected)
    return out


def summarize_csv(path: Path) -> dict[str, float]:
    avg = average_rows(read_csv_rows(path), "t2t")
    latency_ns = avg.get("latency", avg.get("time", 0.0))
    batch = avg.get("batchsize", 0.0)
    throughput = 0.0
    if latency_ns > 0 and batch > 0:
        throughput = batch / (latency_ns * 1e-9)

    fc_attn = (
        avg.get("qkvgen", 0.0)
        + avg.get("q_down_proj", 0.0)
        + avg.get("kv_down_proj", 0.0)
        + avg.get("kr_proj", 0.0)
        + avg.get("q_up_proj", 0.0)
        + avg.get("qr_proj", 0.0)
        + avg.get("kv_up_proj", 0.0)
        + avg.get("tr_k_up_proj", 0.0)
        + avg.get("v_up_proj", 0.0)
        + avg.get("o_proj", 0.0)
    )
    core_attention = avg.get("atten_sum", 0.0) + avg.get("atten_gen", 0.0)
    moe_ffn = avg.get("ffn", 0.0) + avg.get("expert_ffn", 0.0)
    communication = avg.get("communication", 0.0)
    known = fc_attn + core_attention + moe_ffn + communication
    etc = max(0.0, latency_ns - known)

    return {
        "latency_ns": latency_ns,
        "throughput_tps": throughput,
        "batchsize": batch,
        "seqlen": avg.get("seqlen", 0.0),
        "fc_attn_ns": fc_attn,
        "core_attention_ns": core_attention,
        "moe_ffn_ns": moe_ffn,
        "communication_ns": communication,
        "etc_ns": etc,
        "oom": avg.get("OOM", 0.0),
    }


def attention_breakdown_from_csv(path: Path) -> dict[str, float]:
    avg = average_rows(read_csv_rows(path), "t2t")
    is_absorb = (avg.get("tr_k_up_proj", 0.0) + avg.get("v_up_proj", 0.0)) > 0
    kv_decompress = (
        avg.get("tr_k_up_proj", 0.0) + avg.get("v_up_proj", 0.0)
        if is_absorb
        else avg.get("kv_up_proj", 0.0)
    )
    score_context = avg.get("atten_sum", 0.0) + avg.get("atten_gen", 0.0)
    out_proj = avg.get("o_proj", 0.0)
    # Figure 6 is an attention-block layer breakdown. The CSV-level
    # `communication` stat is a global communication bucket collected from
    # Comm stamps and should not be folded into the attention-layer "Etc"
    # bucket here; doing so makes reordered MLA look dominated by gray bars.
    etc = (
        avg.get("qkvgen", 0.0)
        + avg.get("q_down_proj", 0.0)
        + avg.get("kv_down_proj", 0.0)
        + avg.get("kr_proj", 0.0)
        + avg.get("q_up_proj", 0.0)
        + avg.get("qr_proj", 0.0)
        + avg.get("rope", 0.0)
        + avg.get("layernorm", 0.0)
        + avg.get("residual", 0.0)
    )
    return {
        "kv_decompress": kv_decompress,
        "score_context": score_context,
        "out_proj": out_proj,
        "etc": etc,
        "total": kv_decompress + score_context + out_proj + etc,
        "is_absorb": float(is_absorb),
    }


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


def add_common_args(parser: Any) -> None:
    parser.add_argument("--run", action="store_true", help="Run simulations")
    parser.add_argument("--plot", action="store_true", help="Generate plots from data")
    parser.add_argument("--all", action="store_true", help="Run simulations and plot")
    parser.add_argument("--quick", action="store_true", help="Use a small smoke-test grid")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing CSV results")
    parser.add_argument("--timeout", type=int, default=900, help="Per-point simulator timeout in seconds")
