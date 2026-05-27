# Exp5：Figure 12/13 - Expert Routing Skew

## 论文目标

Figure 12 研究 expert routing 分布变得倾斜时，throughput-latency 如何退化。Figure 13 对比 32 GPU x8 和 256 GPU 两种部署粒度，说明较小部署组在 skew 下可以缓解 accelerator-level load imbalance。

## 模拟器配置

主脚本：

```bash
python3 experiments/exp5/run_skew.py
```

默认 sweep：

| 参数 | 取值 |
| --- | --- |
| 模型 | `deepseekV3` |
| Sequence length | `2048` |
| Skewness `s` | `0.0, 0.2, 0.4, 0.6, 0.8` |
| Figure 12 total batch | `1152, 2304, 4608, 6912, 9216, 11520, 13824, 16128, 18432, 20736` |
| Figure 13 batch per GPU | `24, 48, 96, 144, 192, 240, 288, 336, 384, 408` |
| 系统 | `fig12_32gpu`, `fig13_32gpu_x8`, `fig13_256gpu` |
| Interconnect | 900GB/s |
| 阶段 | decode |

Figure 12 使用 32-GPU 系统并以系统总 batch 为横轴；`s=0.0` 是无 skew 基线，`0.2..0.8` 对应论文的 skew 退化趋势。Figure 13 对比 8 个 32-GPU 部署和单个 256-GPU 部署，以 batch per GPU 为横轴；脚本对 `32 GPU x8` 的吞吐乘以 `deployment_count=8` 后写入 `paper_system_throughput_tps`。脚本也兼容旧命名的 `_b` CSV。

## 运行方式

```bash
python3 experiments/exp5/run_skew.py --quick --all
python3 experiments/exp5/run_skew.py --run
python3 experiments/exp5/run_skew.py --plot
```

## 输出

- `data/result_system_*_s*_tb*.csv`
- `data/result_system_*_s*_bpg*.csv`
- `data/summary_skew.csv`
- `plots/figure12_13_skew.png`

summary 中还包含基于 Zipf 分布和 expert grouping 估算的 accelerator-level load imbalance ratio。
