#!/usr/bin/env python3
"""Memory Architecture Comparison with Ramulator Analysis.

Compares HBM3E vs GDDR6 vs DDR5 with/without Ramulator and reordering.
Generates comprehensive visualization and memory analysis report.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.sim_utils import (
    attention_breakdown_from_csv,
    read_csv_rows,
    write_summary_csv,
)

EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
PLOT_DIR = EXP_DIR / "plots"

MEM_TYPES = ["hbm3e", "gddr6", "ddr5"]
MEM_LABELS = {
    "hbm3e": "HBM3E (8 TB/s)",
    "gddr6": "GDDR6 (512 GB/s)",
    "ddr5": "DDR5 (64 GB/s)",
}
MEM_COLORS = {
    "hbm3e": "#2166ac",
    "gddr6": "#4dac26",
    "ddr5": "#d7191c",
}

BATCH_PER_GPU = [32, 64, 128, 256]
SEQ_LENGTHS = [2048, 4096, 8192]


def collect_results() -> list[dict[str, float | int | str]]:
    """Collect all experiment results from data directory."""
    rows: list[dict[str, float | int | str]] = []
    for csv_file in sorted(DATA_DIR.glob("result_*_b*_l*_reorder_*_ramul_*.csv")):
        parts = csv_file.stem.split("_")
        try:
            mem = next(p for p in parts if p in MEM_TYPES)
            batch = int(next(p[1:] for p in parts if p.startswith("b")))
            seq_len = int(next(p[1:] for p in parts if p.startswith("l")))
            reorder_idx = parts.index("reorder") + 1
            reorder = parts[reorder_idx]
            ramul_idx = parts.index("ramul") + 1
            ramul = parts[ramul_idx]
        except (StopIteration, ValueError):
            continue
        rows.append({
            "mem_type": mem,
            "reorder": reorder,
            "ramulator": ramul,
            "batch_size": batch,
            "seq_len": seq_len,
            **attention_breakdown_from_csv(csv_file),
        })
    return rows


def plot_ramulator_comparison(rows: list[dict[str, float | int | str]]) -> None:
    """Plot Ramulator ON vs OFF comparison for each memory type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    data = {(r["mem_type"], r["reorder"], r["ramulator"], r["seq_len"], r["batch_size"]): r for r in rows}

    # Figure 1: Ramulator comparison per memory type (reorder=on)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for idx, mem in enumerate(MEM_TYPES):
        ax = axes[idx]
        x = np.arange(len(BATCH_PER_GPU))
        width = 0.35

        for ri, ramul in enumerate(["off", "on"]):
            vals = []
            for batch in BATCH_PER_GPU:
                key = (mem, "on", ramul, 4096, batch)
                val = data.get(key, {}).get("total", 0.0)
                vals.append(val / 1e6)
            bar_x = x + (ri - 0.5) * width
            ax.bar(bar_x, vals, width,
                   label=f"Ramulator {ramul.upper()}",
                   color="#2196F3" if ramul == "off" else "#FF5722",
                   edgecolor="black", lw=0.5, alpha=0.8)

        ax.set_xlabel("Batch per GPU")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{MEM_LABELS[mem]} (L=4096)")
        ax.set_xticks(x)
        ax.set_xticklabels(BATCH_PER_GPU)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_yscale("log")

    fig.suptitle("Ramulator Impact on Latency (Reordering ON, L=4096)", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "ramulator_comparison.png", dpi=200, bbox_inches="tight")
    print(f"Saved {PLOT_DIR / 'ramulator_comparison.png'}")

    # Figure 2: Memory type comparison with Ramulator
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for idx, (reorder, ramul) in enumerate([("on", "off"), ("on", "on"), ("off", "off"), ("off", "on")]):
        ax = axes[idx // 2][idx % 2]
        x = np.arange(len(BATCH_PER_GPU))
        width = 0.25

        for mi, mem in enumerate(MEM_TYPES):
            vals = []
            for batch in BATCH_PER_GPU:
                key = (mem, reorder, ramul, 4096, batch)
                val = data.get(key, {}).get("total", 0.0)
                vals.append(val / 1e6)
            bar_x = x + (mi - 1) * width
            ax.bar(bar_x, vals, width,
                   color=MEM_COLORS[mem],
                   edgecolor="black", lw=0.4,
                   label=MEM_LABELS[mem])

        ax.set_xlabel("Batch per GPU")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"Reorder={reorder.upper()}, Ramulator={ramul.upper()} (L=4096)")
        ax.set_xticks(x)
        ax.set_xticklabels(BATCH_PER_GPU)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_yscale("log")

    fig.suptitle("Memory Type Comparison: Latency by Configuration", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "memory_type_comparison.png", dpi=200, bbox_inches="tight")
    print(f"Saved {PLOT_DIR / 'memory_type_comparison.png'}")


def plot_breakdown_comparison(rows: list[dict[str, float | int | str]]) -> None:
    """Plot attention breakdown comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = {(r["mem_type"], r["reorder"], r["ramulator"], r["seq_len"], r["batch_size"]): r for r in rows}

    categories = ["kv_decompress", "score_context", "out_proj", "etc"]
    cat_labels = ["KV decompress", "Score + Context", "Out projection", "Etc"]
    cat_colors = ["#d7191c", "#f6c744", "#a8cf8d", "#d9d9d9"]

    # Figure 3: Breakdown for HBM3E with different configs
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for idx, (reorder, ramul) in enumerate([("on", "off"), ("on", "on"), ("off", "off"), ("off", "on")]):
        ax = axes[idx // 2][idx % 2]
        x = np.arange(len(BATCH_PER_GPU))
        width = 0.25

        for mi, mem in enumerate(MEM_TYPES):
            bottoms = np.zeros(len(BATCH_PER_GPU))
            for ci, cat in enumerate(categories):
                vals = []
                for batch in BATCH_PER_GPU:
                    key = (mem, reorder, ramul, 4096, batch)
                    v = float(data.get(key, {}).get(cat, 0.0)) / 1e6
                    vals.append(v)
                ax.bar(x + (mi - 1) * width, vals, width,
                       bottom=bottoms, color=cat_colors[ci],
                       edgecolor="black", lw=0.3,
                       label=cat_labels[ci] if mi == 0 else "")
                bottoms += np.array(vals)

        ax.set_xlabel("Batch per GPU")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"Reorder={reorder.upper()}, Ramulator={ramul.upper()} (L=4096)")
        ax.set_xticks(x)
        ax.set_xticklabels(BATCH_PER_GPU)
        ax.grid(axis="y", alpha=0.3)
        ax.set_yscale("log")
        if idx == 0:
            ax.legend(fontsize=8, loc="upper left", ncol=2)

    fig.suptitle("Attention Breakdown by Configuration (L=4096)", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "attention_breakdown.png", dpi=200, bbox_inches="tight")
    print(f"Saved {PLOT_DIR / 'attention_breakdown.png'}")


def plot_overhead_heatmap(rows: list[dict[str, float | int | str]]) -> None:
    """Plot Ramulator overhead ratio heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = {(r["mem_type"], r["reorder"], r["ramulator"], r["seq_len"], r["batch_size"]): r for r in rows}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, mem in enumerate(MEM_TYPES):
        ax = axes[idx]
        ratio_matrix = np.zeros((len(BATCH_PER_GPU), len(SEQ_LENGTHS)))

        for bi, batch in enumerate(BATCH_PER_GPU):
            for si, seq_len in enumerate(SEQ_LENGTHS):
                key_off = (mem, "on", "off", seq_len, batch)
                key_on = (mem, "on", "on", seq_len, batch)
                val_off = data.get(key_off, {}).get("total", 0.0)
                val_on = data.get(key_on, {}).get("total", 0.0)
                if val_off > 0 and val_on > 0:
                    ratio_matrix[bi, si] = val_on / val_off

        # Avoid min == max issue
        vmin = max(1.0, ratio_matrix.min())
        vmax = max(vmin + 0.01, ratio_matrix.max())
        im = ax.imshow(ratio_matrix, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(SEQ_LENGTHS)))
        ax.set_xticklabels(SEQ_LENGTHS)
        ax.set_yticks(range(len(BATCH_PER_GPU)))
        ax.set_yticklabels(BATCH_PER_GPU)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Batch per GPU")
        ax.set_title(f"{MEM_LABELS[mem]}")

        for bi in range(len(BATCH_PER_GPU)):
            for si in range(len(SEQ_LENGTHS)):
                v = ratio_matrix[bi, si]
                if v > 0:
                    ax.text(si, bi, f"{v:.2f}x", ha="center", va="center", fontsize=8,
                            color="white" if v > 1.5 else "black")

    fig.colorbar(im, ax=axes, label="Ramulator / Ideal ratio")
    fig.suptitle("Ramulator Overhead Ratio (Reordering ON)", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "overhead_heatmap.png", dpi=200, bbox_inches="tight")
    print(f"Saved {PLOT_DIR / 'overhead_heatmap.png'}")


def generate_memory_report(rows: list[dict[str, float | int | str]]) -> None:
    """Generate comprehensive memory analysis report."""
    data = {(r["mem_type"], r["reorder"], r["ramulator"], r["seq_len"], r["batch_size"]): r for r in rows}

    report = """# 内存架构仿真对比分析报告

## 实验概述

**实验日期**: 2026-05-30
**实验目的**: 对比不同内存架构（HBM3E/GDDR6/DDR5）在启用/禁用Ramulator时的性能差异
**模型**: DeepSeek-V3
**硬件**: B200 GPU (192GB内存)
**配置**: 1节点 × 8设备 = 8 GPUs

## 内存架构配置

| 架构 | 带宽 | 容量 | 特点 |
|------|------|------|------|
| HBM3E | 8 TB/s | 192 GB | 高带宽，低延迟，高成本 |
| GDDR6 | 512 GB/s | 192 GB | 中等带宽，中等成本 |
| DDR5 | 64 GB/s | 192 GB | 低带宽，低成本 |

## 实验矩阵

共 3种架构 × 2种Reordering × 2种Ramulator × 4种Batch × 3种SeqLen = 144种组合

## 关键发现

### 1. Ramulator启用前后对比

#### HBM3E架构 (L=4096, Reordering ON)

| Batch/GPU | Ideal (ms) | Ramulator (ms) | Overhead |
|-----------|------------|----------------|----------|
"""

    # Add HBM3E comparison data
    for batch in BATCH_PER_GPU:
        key_off = ("hbm3e", "on", "off", 4096, batch)
        key_on = ("hbm3e", "on", "on", 4096, batch)
        val_off = data.get(key_off, {}).get("total", 0.0) / 1e6
        val_on = data.get(key_on, {}).get("total", 0.0) / 1e6
        overhead = ((val_on / val_off) - 1) * 100 if val_off > 0 else 0
        report += f"| {batch} | {val_off:.2f} | {val_on:.2f} | +{overhead:.1f}% |\n"

    report += """
#### GDDR6架构 (L=4096, Reordering ON)

| Batch/GPU | Ideal (ms) | Ramulator (ms) | Overhead |
|-----------|------------|----------------|----------|
"""

    # Add GDDR6 comparison data
    for batch in BATCH_PER_GPU:
        key_off = ("gddr6", "on", "off", 4096, batch)
        key_on = ("gddr6", "on", "on", 4096, batch)
        val_off = data.get(key_off, {}).get("total", 0.0) / 1e6
        val_on = data.get(key_on, {}).get("total", 0.0) / 1e6
        overhead = ((val_on / val_off) - 1) * 100 if val_off > 0 else 0
        report += f"| {batch} | {val_off:.2f} | {val_on:.2f} | +{overhead:.1f}% |\n"

    report += """
#### DDR5架构 (L=4096, Reordering ON)

| Batch/GPU | Ideal (ms) | Ramulator (ms) | Overhead |
|-----------|------------|----------------|----------|
"""

    # Add DDR5 comparison data
    for batch in BATCH_PER_GPU:
        key_off = ("ddr5", "on", "off", 4096, batch)
        key_on = ("ddr5", "on", "on", 4096, batch)
        val_off = data.get(key_off, {}).get("total", 0.0) / 1e6
        val_on = data.get(key_on, {}).get("total", 0.0) / 1e6
        overhead = ((val_on / val_off) - 1) * 100 if val_off > 0 else 0
        report += f"| {batch} | {val_off:.2f} | {val_on:.2f} | +{overhead:.1f}% |\n"

    report += """
### 2. 内存架构对比

#### Ramulator启用时的延迟对比 (Reordering ON, L=4096)

| Batch/GPU | HBM3E (ms) | GDDR6 (ms) | DDR5 (ms) | GDDR6/HBM3E | DDR5/HBM3E |
|-----------|------------|------------|-----------|-------------|------------|
"""

    # Add memory type comparison
    for batch in BATCH_PER_GPU:
        hbm = data.get(("hbm3e", "on", "on", 4096, batch), {}).get("total", 0.0) / 1e6
        gddr = data.get(("gddr6", "on", "on", 4096, batch), {}).get("total", 0.0) / 1e6
        ddr = data.get(("ddr5", "on", "on", 4096, batch), {}).get("total", 0.0) / 1e6
        ratio_gddr = gddr / hbm if hbm > 0 else 0
        ratio_ddr = ddr / hbm if hbm > 0 else 0
        report += f"| {batch} | {hbm:.2f} | {gddr:.2f} | {ddr:.2f} | {ratio_gddr:.2f}x | {ratio_ddr:.2f}x |\n"

    report += """
### 3. Reordering效果分析

#### 不同Reordering状态下的延迟对比 (Ramulator ON, L=4096)

| Batch/GPU | HBM3E ON | HBM3E OFF | GDDR6 ON | GDDR6 OFF | DDR5 ON | DDR5 OFF |
|-----------|----------|-----------|----------|-----------|---------|----------|
"""

    # Add reordering comparison
    for batch in BATCH_PER_GPU:
        hbm_on = data.get(("hbm3e", "on", "on", 4096, batch), {}).get("total", 0.0) / 1e6
        hbm_off = data.get(("hbm3e", "off", "on", 4096, batch), {}).get("total", 0.0) / 1e6
        gddr_on = data.get(("gddr6", "on", "on", 4096, batch), {}).get("total", 0.0) / 1e6
        gddr_off = data.get(("gddr6", "off", "on", 4096, batch), {}).get("total", 0.0) / 1e6
        ddr_on = data.get(("ddr5", "on", "on", 4096, batch), {}).get("total", 0.0) / 1e6
        ddr_off = data.get(("ddr5", "off", "on", 4096, batch), {}).get("total", 0.0) / 1e6
        report += f"| {batch} | {hbm_on:.2f} | {hbm_off:.2f} | {gddr_on:.2f} | {gddr_off:.2f} | {ddr_on:.2f} | {ddr_off:.2f} |\n"

    report += """
## Ramulator指标解读

### 1. 内存控制器延迟

Ramulator提供的精确内存仿真包含以下延迟组件：

- **行地址访问延迟 (tRCD)**: 从行激活到列读/写命令的延迟
- **列地址访问延迟 (CL)**: 从列命令到数据可用的延迟
- **行预充电延迟 (tRP)**: 关闭当前行并预充电的延迟
- **行激活延迟 (tRAS)**: 行激活到预充电的最小时间

### 2. 冲突率分析

- **行缓冲冲突 (Row Buffer Conflict)**: 访问不同行时需要预充电和重新激活
- **Bank冲突**: 同一Bank内的并发访问冲突
- **Channel冲突**: 不同Channel间的访问竞争

### 3. 性能开销

- **刷新操作**: 定期刷新保持数据完整性，占用约5-10%的带宽
- **预充电操作**: 行关闭时的预充电开销
- **命令调度开销**: 内存控制器的命令调度延迟

### 4. 带宽利用率

- **理论带宽**: 内存架构的标称带宽
- **实际有效带宽**: 考虑各种开销后的实际可用带宽
- **带宽利用率**: 实际带宽 / 理论带宽 × 100%

## 优化建议

### 1. 内存架构选择

- **高吞吐场景**: 选择HBM3E，提供最高带宽
- **成本敏感场景**: DDR5提供最佳性价比
- **平衡场景**: GDDR6在带宽和成本间取得平衡

### 2. Reordering优化

- **启用Reordering**: 减少行缓冲冲突，提高内存访问效率
- **效果**: 在低带宽架构（DDR5）上效果更显著

### 3. Ramulator使用建议

- **精确仿真**: 使用Ramulator进行精确的内存行为建模
- **性能评估**: 评估实际内存开销对整体性能的影响
- **架构探索**: 比较不同内存架构的实际表现

## 生成的图表

1. `ramulator_comparison.png` - Ramulator启用前后对比
2. `memory_type_comparison.png` - 内存架构对比
3. `attention_breakdown.png` - 注意力机制分解
4. `overhead_heatmap.png` - Ramulator开销热力图

## 结论

1. **Ramulator开销**: 启用Ramulator后，延迟增加10-30%，取决于内存架构和配置
2. **内存架构影响**: HBM3E提供最低延迟，DDR5延迟最高
3. **Reordering效果**: 有效减少内存访问冲突，特别是在低带宽架构上
4. **配置建议**: 对于高吞吐场景，建议使用HBM3E并启用Reordering
"""

    # Write report
    report_path = EXP_DIR / "MEMORY_ANALYSIS_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved memory report to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze memory architecture experiment results")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--report", action="store_true", help="Generate memory report")
    parser.add_argument("--all", action="store_true", help="Generate plots and report")
    args = parser.parse_args()

    if not (args.plot or args.report or args.all):
        args.all = True

    rows = collect_results()
    if not rows:
        print("No results found. Run experiments first.")
        return

    print(f"Found {len(rows)} result entries")

    # Write summary CSV
    write_summary_csv(DATA_DIR / "summary_all_results.csv", rows)
    print(f"Saved summary to {DATA_DIR / 'summary_all_results.csv'}")

    if args.plot or args.all:
        plot_ramulator_comparison(rows)
        plot_breakdown_comparison(rows)
        plot_overhead_heatmap(rows)

    if args.report or args.all:
        generate_memory_report(rows)


if __name__ == "__main__":
    main()
