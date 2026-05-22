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
| Batch size | `32, 64, 128, 256, 512, 1024` |
| 系统 | `32gpu_x8`, `256gpu` |
| Interconnect | 900GB/s |
| 阶段 | decode |

## 运行方式

```bash
python3 experiments/exp5/run_skew.py --quick --all
python3 experiments/exp5/run_skew.py --run
python3 experiments/exp5/run_skew.py --plot
```

## 输出

- `data/result_system_*_s*_b*.csv`
- `data/summary_skew.csv`
- `plots/figure12_13_skew.png`

summary 中还包含基于 Zipf 分布和 expert grouping 估算的 accelerator-level load imbalance ratio。
