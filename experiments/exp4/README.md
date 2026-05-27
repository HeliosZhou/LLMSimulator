# Exp4：Figure 10/11 - Interconnect 与部署规模

## 论文目标

Figure 10 展示当 interconnect 带宽较低时，MoE communication 会成为 decode 阶段的重要瓶颈。Figure 11 对比多组小规模 32-GPU 部署和单个 256-GPU 部署在不同 interconnect 带宽下的吞吐表现。

## 模拟器配置

主脚本：

```bash
python3 experiments/exp4/run_interconnect.py
```

系统配置：

| 标签 | 配置 |
| --- | --- |
| `fig10_32gpu_xdr` | Figure 10，32 GPU，100GB/s interconnect |
| `fig11_32gpu_x8_900` | Figure 11，8 个 32-GPU 部署，900GB/s |
| `fig11_256gpu_100` | Figure 11，单个 256-GPU 部署，100GB/s |
| `fig11_256gpu_300` | Figure 11，单个 256-GPU 部署，300GB/s |
| `fig11_256gpu_900` | Figure 11，单个 256-GPU 部署，900GB/s |

脚本中的 interconnect 档位：

- `infiniband_gen=800` 表示 100GB/s
- `infiniband_gen=2400` 表示 300GB/s
- `infiniband_gen=7200` 表示 900GB/s

默认 sweep：

| 参数 | 取值 |
| --- | --- |
| 模型 | `deepseekV3` |
| Figure 10 sequence length | `2048, 8192` |
| Figure 10 total batch | `1152, 2304, ..., 20736` |
| Figure 11 sequence length | `2048, 16384` |
| Figure 11 batch per GPU | `12, 24, ..., 420` |
| 阶段 | decode |

Figure 11 中 `32 GPU x8` 表示 8 个独立 32-GPU 部署。脚本会对该配置的模拟器吞吐乘以 `deployment_count=8`，并在 summary 中写入 `paper_system_throughput_tps`。

Figure 10 对应论文中 “32 B200 GPU + InfiniBand XDR 100GB/s” 的 decode breakdown/throughput 图；Figure 11 对应 “32 GPU x8” 与 “256 GPU” 在 900/300/100GB/s 互联下的部署粒度对比。脚本也兼容旧命名的 `_b` CSV，但新运行会按 `_tb` 或 `_bpg` 写出。

## 运行方式

```bash
python3 experiments/exp4/run_interconnect.py --quick --all
python3 experiments/exp4/run_interconnect.py --run
python3 experiments/exp4/run_interconnect.py --plot
```

## 输出

- `data/result_system_*_l*_tb*.csv`
- `data/result_system_*_l*_bpg*.csv`
- `data/summary_interconnect.csv`
- `plots/figure10_11_interconnect.png`

预期趋势：更高的 interconnect 带宽会降低 communication latency，并在较大 batch 下显著改善 throughput。
