# Exp6：Figure 14 - GPU 与 GPU+PIM 对比

## 论文目标

Figure 14 评估 PIM-style execution 对 MoE 层的收益。论文结论是：在较低 batch 下，expert computation 更偏 memory-bound，PIM 更有帮助；batch 增大后，计算逐渐成为瓶颈，PIM 收益下降。

## 模拟器配置

主脚本：

```bash
python3 experiments/exp6/run_pim.py
```

执行模式：

| 模式 | 配置 |
| --- | --- |
| `gpu` | GPU-only |
| `gpu_pim` | `processor_type=GPU+PIM`，并设置 `use_low_unit_moe_only=true` |

PIM 参数：

| 参数 | 取值 |
| --- | --- |
| `pim_x` | `4` |
| `pim_op_b` | `8` |

这对应论文 Figure 14 中 Duplex-style PIM：MoE execution 由 PIM 处理，PIM 的 ridge point 约为 8，并使用约 4x GPU HBM 带宽。

默认 sweep：

| 参数 | 取值 |
| --- | --- |
| 模型 | `deepseekV3` |
| GPU 系统 | 32 B200 GPU |
| Sequence length | `1024, 4096, 16384` |
| Batch per GPU | `8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128` |
| 阶段 | decode |

论文 Figure 14 是 normalized throughput heatmap，横轴为 `Batch per GPU`，纵轴为 sequence length。脚本会转换为模拟器系统总 batch：`max_batch_size = batch_per_gpu * 32`。

## 运行方式

```bash
python3 experiments/exp6/run_pim.py --quick --all
python3 experiments/exp6/run_pim.py --run
python3 experiments/exp6/run_pim.py --plot
```

## 输出

- `data/result_mode_*_l*_bpg*.csv`
- `data/summary_pim.csv`
- `plots/figure14_pim.png`

绘图展示同一 sequence length 和 batch per GPU 下，GPU+PIM 相对 GPU-only 的 normalized throughput。

脚本也会读取旧命名的 `data/result_mode_*_l*_b*.csv`；新运行会按 `bpg` 命名，避免把 batch per GPU 和系统总 batch 混淆。
