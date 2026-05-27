# Exp7：Figure 3 - Roofline 解析图

## 论文目标

Figure 3 使用 roofline 图说明不同层的 arithmetic intensity（ArI）和性能边界。核心观察是：

- MHA core attention 的 ArI 约为 1，明显 memory-bound。
- GQA 有一定改善，但仍偏 memory-bound。
- MLA 经过 layer reordering 后，core attention 的 ArI 接近现代加速器 ridge point。
- FFN/MoE 层随 batch 增大，ArI 上升并逐渐接近或超过 ridge point。

## 实验性质

本实验是**解析图生成**，不是运行 LLMSimulator。脚本根据论文 Figure 3/Table I 的硬件参数和代表性 ArI 结论生成 roofline 风格图，用于对应 Figure 3 的可视化。

## 脚本

```bash
python3 experiments/exp7/run_roofline.py
```

默认参数：

| 参数 | 取值 |
| --- | --- |
| 硬件峰值 | 989.5 TFLOPS |
| 内存带宽 | 4800 GB/s |
| Ridge point | 约 206 Op/B |
| 代表点 | MHA、GQA、MLA、FFN B=64/1K、MoE B=64/1K |

## 运行方式

```bash
python3 experiments/exp7/run_roofline.py --all
python3 experiments/exp7/run_roofline.py --plot
```

`--run` 对该解析实验没有额外含义；脚本会直接生成 summary 和图片。

## 输出

- `data/summary_roofline.csv`
- `plots/figure3_roofline.png`

## 注意

论文 Figure 3 标注来自 real-machine measurement。本脚本用于复现图的分析结构和主要趋势，不声称重现实机测量的每个点位。
