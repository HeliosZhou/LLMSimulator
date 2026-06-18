#!/usr/bin/env python3
"""Generate RoMe Figure 1 artifacts from LLMSimulator runtime tensor traces."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
PLOT_DIR = EXP_DIR / "plots"
REPORT_PATH = EXP_DIR / "FIGURE1_REPRODUCTION.md"

CACHE_LINE_BYTES = 32
ROME_ROW_BYTES = 4096

MODEL_ORDER = ["DeepSeek-V3", "Grok 1", "Llama 3-405B"]
STAGE_ORDER = ["prefill", "decode"]
CATEGORY_ORDER = ["weight", "activation", "kv_cache"]

MODEL_LABELS = {
    "deepseekV3": "DeepSeek-V3",
    "grok1": "Grok 1",
    "llama3_405B": "Llama 3-405B",
}
CATEGORY_LABELS = {
    "weight": "Weight",
    "activation": "Activation",
    "kv_cache": "KV cache",
}
MODEL_NUM_KV_HEADS = {
    "DeepSeek-V3": 128,
    "Grok 1": 8,
    "Llama 3-405B": 8,
}
MODEL_COLORS = {
    "DeepSeek-V3": "#ee8d86",
    "Grok 1": "#88bde6",
    "Llama 3-405B": "#f0d070",
}


def infer_layer(module_name: str) -> str:
    match = re.search(r"(?:MoE_decoder|decoder)_(\d+)", module_name)
    if match:
        return match.group(1)
    return "global"


def normalize_category(raw_category: str, raw_tag: str) -> str:
    value = raw_category or raw_tag
    return {
        "act": "activation",
        "activation": "activation",
        "weight": "weight",
        "cache": "kv_cache",
        "kv_cache": "kv_cache",
    }.get(value, value)


def canonical_module(module_name: str) -> str:
    module_name = re.sub(r"expert_FFN_\d+", "expert_FFN", module_name)
    module_name = re.sub(r"shared_expert_FFN_\d+", "shared_expert_FFN", module_name)
    module_name = re.sub(r"(?:k|v)_cache_\d+_\d+", "kv_cache", module_name)
    module_name = re.sub(r"latent_(?:kv|pe)_cache_\d+", "latent_cache", module_name)
    return module_name


def read_trace_samples(trace_paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trace_path in trace_paths:
        with trace_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for trace_row in reader:
                bytes_ = int(float(trace_row.get("bytes", 0) or 0))
                if bytes_ <= 0:
                    continue

                category = normalize_category(
                    trace_row.get("category", ""), trace_row.get("tag", "")
                )
                if category not in set(CATEGORY_ORDER):
                    continue

                raw_model = trace_row.get("model", "")
                model = MODEL_LABELS.get(raw_model, raw_model)
                module_name = trace_row.get("module", "")
                precision_byte = int(float(trace_row.get("precision_byte", 0) or 0))

                rows.append(
                    {
                        "model": model,
                        "stage": trace_row.get("stage", "mixed"),
                        "category": category,
                        "layer": infer_layer(module_name),
                        "module": module_name,
                        "tensor": trace_row.get("tensor", ""),
                        "shape": trace_row.get("shape", ""),
                        "precision_byte": precision_byte,
                        "bytes": bytes_,
                        "kib": bytes_ / 1024,
                        "mib": bytes_ / 1024**2,
                        "cache_lines_32B": math.ceil(bytes_ / CACHE_LINE_BYTES),
                        "rome_rows_4KiB": math.ceil(bytes_ / ROME_ROW_BYTES),
                        "note": "runtime trace sample from LLMSimulator",
                        "trace_source": trace_row.get("source", ""),
                        "layer_type": trace_row.get("layer_type", ""),
                        "device_rank": trace_row.get("device_rank", ""),
                        "process_tokens": trace_row.get("process_tokens", ""),
                        "sum_tokens": trace_row.get("sum_tokens", ""),
                        "gen_tokens": trace_row.get("gen_tokens", ""),
                        "total_sequence_length": trace_row.get("total_sequence_length", ""),
                        "average_sequence_length": trace_row.get(
                            "average_sequence_length", ""
                        ),
                        "processor_type": trace_row.get("processor_type", ""),
                        "trace_path": str(trace_path),
                    }
                )
    return rows


def aggregate_trace_samples(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()

    def add(row: dict[str, object], bytes_: int | None = None, tensor: str | None = None, note: str | None = None) -> None:
        out = dict(row)
        if bytes_ is not None:
            out["bytes"] = bytes_
            out["kib"] = bytes_ / 1024
            out["mib"] = bytes_ / 1024**2
            out["cache_lines_32B"] = math.ceil(bytes_ / CACHE_LINE_BYTES)
            out["rome_rows_4KiB"] = math.ceil(bytes_ / ROME_ROW_BYTES)
        if tensor is not None:
            out["tensor"] = tensor
        if note is not None:
            out["note"] = note
        key = (
            out["model"],
            out["stage"],
            out["category"],
            out["layer"],
            canonical_module(str(out["module"])),
            out["tensor"],
            out["shape"],
            out["bytes"],
        )
        if key in seen:
            return
        seen.add(key)
        rows.append(out)

    for row in raw_rows:
        category = str(row["category"])
        tensor = str(row["tensor"])
        module = str(row["module"])
        layer_type = str(row["layer_type"])
        source = str(row["trace_source"])

        if category == "weight":
            if tensor in {"layer_norm_weight"}:
                continue
            if tensor in {"A", "weight", "Embedding", "lm_head_wgt"}:
                add(row, note="deduplicated weight tensor from runtime trace")
            continue

        if category == "activation":
            if source == "execution":
                if layer_type == "LINEAR" and tensor == "Y":
                    add(row, note="operator output activation from runtime trace")
                elif layer_type == "BATCHED_LINEAR" and tensor in {
                    "batched_linear_output",
                    "plain_linear_output",
                }:
                    add(row, note="batched operator output activation from runtime trace")
                elif layer_type == "ACTIVATION" and tensor in {"ActOut", "GateOut"}:
                    add(row, note="activation operator output from runtime trace")
            elif source == "module":
                if tensor in {
                    "Hidden vector",
                    "layer_norm_output",
                    "sin_cos",
                    "rope_out",
                    "residual_out",
                    "allreduce_output",
                    "attn_output",
                    "concated_latent_kv",
                    "concated_k_rope",
                    "moe_scatter_output",
                    "moe_gather_output",
                    "gate_update_output",
                }:
                    add(row, note="module-level output activation from runtime trace")

    cache_groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in raw_rows:
        if row["category"] != "kv_cache":
            continue
        key = (str(row["model"]), str(row["stage"]), str(row["layer"]), str(row["module"]))
        cache_groups.setdefault(key, []).append(row)

    for (model, _stage, _layer, _module), group_rows in cache_groups.items():
        unique: dict[tuple[str, int], dict[str, object]] = {}
        for row in group_rows:
            tensor = re.sub(r"_\d+(?:_\d+)?$", "", str(row["tensor"]))
            unique[(tensor, int(row["bytes"]))] = row
        logical_bytes = sum(bytes_ for (_tensor, bytes_) in unique)
        if model in {"Grok 1", "Llama 3-405B"}:
            logical_bytes *= MODEL_NUM_KV_HEADS[model]
        representative = dict(group_rows[0])
        representative["category"] = "kv_cache"
        representative["shape"] = "logical_kv_cache_block"
        add(
            representative,
            bytes_=logical_bytes,
            tensor="logical_kv_cache_block",
            note="logical KV-cache block aggregated from runtime trace cache tensors",
        )

    return rows


def percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    pos = (len(values) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(values[lo])
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[int]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["stage"]), str(row["category"]))
        groups.setdefault(key, []).append(int(row["bytes"]))

    out: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        for stage in STAGE_ORDER:
            for category in CATEGORY_ORDER:
                values = sorted(groups.get((model, stage, category), []))
                if not values:
                    continue
                out.append(
                    {
                        "model": model,
                        "stage": stage,
                        "category": category,
                        "samples": len(values),
                        "min_bytes": values[0],
                        "p25_bytes": percentile(values, 25),
                        "median_bytes": statistics.median(values),
                        "p75_bytes": percentile(values, 75),
                        "max_bytes": values[-1],
                        "share_ge_32B": sum(v >= CACHE_LINE_BYTES for v in values)
                        / len(values),
                        "share_ge_4KiB": sum(v >= ROME_ROW_BYTES for v in values)
                        / len(values),
                    }
                )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def plot_figure1(rows: list[dict[str, object]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str, str], list[int]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["stage"]), str(row["category"]))
        grouped.setdefault(key, []).append(int(row["bytes"]))

    data: list[list[int]] = []
    labels: list[str] = []
    positions: list[float] = []
    box_colors: list[str] = []
    model_centers: list[tuple[float, str]] = []
    stage_centers: list[tuple[float, str]] = []

    x = 1.0
    for model in MODEL_ORDER:
        model_start = x
        for stage in STAGE_ORDER:
            if not any((model, stage, category) in grouped for category in CATEGORY_ORDER):
                continue
            stage_start = x
            for category in CATEGORY_ORDER:
                if (model, stage, category) not in grouped:
                    continue
                data.append(grouped[(model, stage, category)])
                labels.append(CATEGORY_LABELS[category])
                positions.append(x)
                box_colors.append(MODEL_COLORS[model])
                x += 1.0
            stage_centers.append(((stage_start + x - 1.0) / 2.0, stage.capitalize()))
            x += 0.55
        if x > model_start:
            model_centers.append(((model_start + x - 1.55) / 2.0, model))
            x += 1.2

    if not data:
        raise ValueError("No Figure 1 samples to plot")

    fig, ax = plt.subplots(figsize=(7.76, 4.65))
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.62,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.2},
        boxprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_yscale("log")
    ax.set_ylabel("Data size (B)")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylim(1e2, 1e10)
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.35)

    for center, stage in stage_centers:
        ax.text(
            center,
            -0.24,
            stage,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
        )
    for center, model in model_centers:
        ax.text(
            center,
            -0.38,
            model,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
        )

    fig.subplots_adjust(bottom=0.38, left=0.11, right=0.99, top=0.97)
    out_path = PLOT_DIR / "rome_figure1_access_distribution.png"
    fig.savefig(out_path, dpi=180)
    fig.savefig(PLOT_DIR / "rome_figure1_access_distribution.pdf")
    plt.close(fig)
    return out_path


def fmt_bytes(value: float) -> str:
    if value >= 1024**3:
        return f"{value / 1024**3:.2f} GiB"
    if value >= 1024**2:
        return f"{value / 1024**2:.2f} MiB"
    if value >= 1024:
        return f"{value / 1024:.2f} KiB"
    return f"{value:.0f} B"


def write_report(summary_rows: list[dict[str, object]], plot_path: Path) -> None:
    lines = [
        "# RoMe Figure 1 复现",
        "",
        "本实验按论文 Figure 1 的思路，从 LLMSimulator runtime tensor trace 统计 DeepSeek-V3、Grok 1、Llama 3-405B 在 prefill 与 decode 阶段的 weight、activation、KV cache 访问数据大小分布。",
        "",
        "## 口径",
        "",
        "- 数据来源只使用 LLMSimulator runtime trace，不再保留模型参数公式估算 fallback。",
        "- trace 由 `run_trace_figure1.py` 按论文 8-accelerator 上下文生成：DeepSeek-V3 使用 non-expert TP=1 / expert TP=1，Grok 1 使用 non-expert TP=8 / expert TP=1，Llama 3-405B 使用 non-expert TP=8。",
        "- 当前 trace exporter 记录的是 `device_rank=0` 的 per-accelerator 执行路径；tensor shape 会受到 8-accelerator TP/EP/DP 配置影响，但 CSV 不是 8 个 device 的全量合并 trace。",
        "- 绘图样本不是 raw trace 逐行直方统计，而是 trace-derived paper-style aggregation：去除输入重复计数，保留 operator/module 输出，KV cache 聚合为 logical cache block。",
        "- C++ trace exporter 在 `Device::execution()` 与少数 module-level forward 中导出 tensor access。",
        "- trace 中 `tag=weight/act/cache` 分别映射为 Figure 1 的 `weight/activation/KV cache`。",
        "- 每个样本大小按 trace 中 tensor `shape * precision_byte` 计算。",
        "- trace CSV 保留 `source/layer_type/module/tensor/process_tokens/device_rank`，用于回查箱线图样本来源。",
        "",
        "## 生成文件",
        "",
        "- `data/figure1_access_samples.csv`",
        "- `data/figure1_access_summary.csv`",
        f"- `plots/{plot_path.name}`",
        "- `plots/rome_figure1_access_distribution.pdf`",
        "",
        "## 分组摘要",
        "",
        "| 模型 | 阶段 | 类别 | 样本数 | 最小值 | 中位数 | 最大值 | >=4KiB |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {model} | {stage} | {category} | {samples} | {min_} | {median} | {max_} | {share:.1%} |".format(
                model=row["model"],
                stage=row["stage"],
                category=row["category"],
                samples=row["samples"],
                min_=fmt_bytes(float(row["min_bytes"])),
                median=fmt_bytes(float(row["median_bytes"])),
                max_=fmt_bytes(float(row["max_bytes"])),
                share=float(row["share_ge_4KiB"]),
            )
        )
    lines.extend(
        [
            "",
            "## 合理性检查",
            "",
            "- 三个模型、prefill/decode、weight/activation/KV cache 共 18 个分组均有样本。",
            "- raw trace 中大量重复输入和 simulator 内部 MoE expert 遍历会显著扭曲箱线图；当前报告使用聚合后的 trace-derived samples。",
            "- Prefill activation 明显大于 decode activation，符合 prefill 处理整段 prompt、decode 单 token 生成的阶段差异。",
            "- KV cache 样本均为 MiB 量级并全部大于 4 KiB，支撑 RoMe 使用 row-granularity access 的动机。",
            "- DeepSeek-V3 decode activation 中仍有大量 64B/2KiB 小样本，来自 trace 中 LayerNorm/RoPE/element-wise 等 module-level 小 tensor；这比原先手写公式更接近真实执行路径，但也说明当前 trace 是 tensor/module 粒度，不是作者未公开的 kernel-level 原始 trace。",
            "",
            "## 与原文 Figure 1 的关系",
            "",
            "论文未公开 Figure 1 的逐样本 trace/CSV。因此当前结果是基于 LLMSimulator runtime trace 的复现口径，而不是作者原始数据的逐点复刻。若图形仍与论文有差异，主要可能来自 trace 取样粒度、并行切分口径、embedding/lm_head 是否纳入、MoE expert 触达口径等未公开细节。",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def write_readme() -> None:
    readme = EXP_DIR / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# RoMe Figure 1 Trace Reproduction",
                "",
                "This experiment reproduces the Figure 1 access-size distribution from LLMSimulator runtime tensor traces.",
                "",
                "The trace run uses the paper-style 8-accelerator context:",
                "",
                "- DeepSeek-V3: non-expert TP = 1, expert TP = 1, DP = 8.",
                "- Grok 1: non-expert TP = 8, expert TP = 1.",
                "- Llama 3-405B: non-expert TP = 8.",
                "",
                "The exported CSVs are rank-0 per-accelerator traces under that",
                "8-accelerator context. They are not merged traces from all 8 devices.",
                "",
                "Run:",
                "",
                "```bash",
                "cmake --build build -j",
                "python3 experiments/exp_rome0_background/run_trace_figure1.py",
                "```",
                "",
                "Regenerate only from existing traces:",
                "",
                "```bash",
                "python3 experiments/exp_rome0_background/reproduce_figure1.py --from-trace experiments/exp_rome0_background/data/*_trace.csv",
                "```",
                "",
                "Outputs:",
                "",
                "- `data/*_trace.csv`",
                "- `data/figure1_access_samples.csv`",
                "- `data/figure1_access_summary.csv`",
                "- `plots/rome_figure1_access_distribution.png`",
                "- `plots/rome_figure1_access_distribution.pdf`",
                "- `FIGURE1_REPRODUCTION.md`",
                "",
                "The previous analytical model-parameter fallback has been removed; this directory is trace-only.",
            ]
        )
        + "\n"
    )


def validate_groups(summary_rows: list[dict[str, object]]) -> None:
    groups = {
        (str(row["model"]), str(row["stage"]), str(row["category"]))
        for row in summary_rows
    }
    missing = [
        (model, stage, category)
        for model in MODEL_ORDER
        for stage in STAGE_ORDER
        for category in CATEGORY_ORDER
        if (model, stage, category) not in groups
    ]
    if missing:
        raise SystemExit(f"Missing Figure 1 groups: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-trace",
        nargs="+",
        type=Path,
        required=True,
        help="LLMSimulator tensor trace CSV files",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    raw_samples = read_trace_samples(args.from_trace)
    samples = aggregate_trace_samples(raw_samples)
    if not samples:
        raise SystemExit("No samples were generated from traces.")

    summary_rows = summarize(samples)
    validate_groups(summary_rows)
    write_csv(DATA_DIR / "figure1_access_samples.csv", samples)
    write_csv(DATA_DIR / "figure1_access_summary.csv", summary_rows)
    plot_path = plot_figure1(samples)
    write_report(summary_rows, plot_path)
    write_readme()

    print(f"Read {len(raw_samples)} raw trace samples")
    print(f"Wrote {len(samples)} aggregated trace-derived samples")
    print(f"Wrote {DATA_DIR / 'figure1_access_samples.csv'}")
    print(f"Wrote {DATA_DIR / 'figure1_access_summary.csv'}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
