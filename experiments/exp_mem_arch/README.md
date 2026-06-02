# Exp Mem Arch: Memory Architecture Comparison with Ramulator

## 概述

本实验对比不同内存架构（HBM3E/GDDR6/DDR5）在启用/禁用Ramulator时的性能差异，分析Reordering优化对内存访问效率的影响。

## 实验矩阵

| 维度 | 值 | 数量 |
|------|-----|------|
| 内存架构 | HBM3E, GDDR6, DDR5 | 3 |
| Reordering | ON, OFF | 2 |
| Ramulator | ON, OFF | 2 |
| Batch/GPU | 32, 64, 128, 256 | 4 |
| Seq Length | 2048, 4096, 8192 | 3 |
| **总计** | | **144** |

## 内存架构配置

| 架构 | 带宽 | 容量 | 特点 |
|------|------|------|------|
| HBM3E | 8 TB/s | 192 GB | 高带宽，低延迟，高成本 |
| GDDR6 | 512 GB/s | 192 GB | 中等带宽，中等成本 |
| DDR5 | 64 GB/s | 192 GB | 低带宽，低成本 |

Ramulator ON 的 GDDR6/DDR5 实验默认使用 `system.ramulator_sample_stride=4096`，在 C++ 请求生成层每 4096 个有效 DRAM bundle 取 1 个代表性请求，并将 Ramulator 返回的时延和命令计数按实际采样比例放大。这样保持 DDR5/GDDR6 的组织不变，同时避免 DDR5/GDDR6 全量请求在 Ramulator 中长时间无法完成。

## 能耗模型说明

当前 LLMSimulator 的 CSV 能耗仍是命令计数乘固定能耗常数的模型。ACT/RD/WR 常数来自 FGDRAM MICRO 2017 Table 3 中的 HBM2/QB-HBM 估算；该论文没有给出 REF 能耗或 background/standby power。新增的 `ref_energy` 因此采用 activation-equivalent 估算，即 `ref_count * kREF_energy`，默认 `kREF_energy = kACT_energy`；`background_energy` 使用 `background_time * kBACKGROUND_power_nW_`，默认 background power 为 0 nW，需替换为 DRAMPower 或 datasheet 参数后才会产生非零能耗。

`background_time` 的口径是每个执行段的 DRAM 通电计费时间，使用 layer/request 的 `total_duration` 累加，不等同于 `memory_duration`。`memory_duration` 只表示访存服务时间；在计算和访存重叠或计算主导的阶段，二者不能混用。

## 使用方法

### 1. 运行实验

```bash
# 整理已有数据并运行缺失实验
bash experiments/exp_mem_arch/organize_and_run.sh

# 或运行所有实验
bash experiments/exp_mem_arch/run_experiments.sh
```

### 2. 分析结果

```bash
# 生成图表和报告
python3 experiments/exp_mem_arch/analyze_results.py --all

# 仅生成图表
python3 experiments/exp_mem_arch/analyze_results.py --plot

# 仅生成报告
python3 experiments/exp_mem_arch/analyze_results.py --report
```

## 输出文件

### 数据文件
- `data/result_{mem}_b{B}_l{L}_reorder_{on|off}_ramul_{on|off}.csv` - 实验结果
- `data/summary_all_results.csv` - 汇总数据

### 图表文件
- `plots/ramulator_comparison.png` - Ramulator启用前后对比
- `plots/memory_type_comparison.png` - 内存架构对比
- `plots/attention_breakdown.png` - 注意力机制分解
- `plots/overhead_heatmap.png` - Ramulator开销热力图

### 报告文件
- `MEMORY_ANALYSIS_REPORT.md` - 内存分析报告

## 核心对比维度

### 1. 同架构下Ramulator启用与否的差异

对比未启用Ramulator的简化内存模型与启用Ramulator的精确仿真模型：
- 吞吐量差异
- 延迟差异
- 内存带宽利用率
- 请求队列深度

### 2. 启用Ramulator后新增指标

- 内存控制器的行/列地址访问延迟
- 不同内存通道/rank/bank的冲突率
- 刷新操作、预充电操作带来的性能开销
- 重排序优化带来的请求命中/冲突改善率
- 实际有效带宽与理论带宽的差距

## 可视化说明

### 1. 柱状图/折线图
- 不同内存架构、reordering状态下的性能指标对比
- Ramulator启用前后的指标差异对比

### 2. 饼图/堆叠柱状图
- 关键内存操作阶段的延迟分布
- 利用率占比分析

### 3. 热力图
- Ramulator开销在不同配置下的分布
- 内存架构性能差异的可视化

## 参考实验

- `exp_mem/` - 内存类型影响实验（无Ramulator）
- `exp1/` - 注意力机制分解实验（含Ramulator对比）

## 模型配置

- 模型: DeepSeek-V3
- GPU: B200
- 节点: 1
- 设备: 8
- 精度: 1 byte
- 迭代: 3
