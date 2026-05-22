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
| `32gpu_xdr` | 32 GPU，100GB/s inter-node bandwidth |
| `32gpu_nvlink` | 32 GPU，900GB/s inter-node bandwidth |
| `256gpu_100` | 256 GPU，100GB/s |
| `256gpu_300` | 256 GPU，300GB/s |
| `256gpu_900` | 256 GPU，900GB/s |

脚本中的 interconnect 档位：

- `infiniband_gen=800` 表示 100GB/s
- `infiniband_gen=2400` 表示 300GB/s
- `infiniband_gen=7200` 表示 900GB/s

默认 sweep：

| 参数 | 取值 |
| --- | --- |
| 模型 | `deepseekV3` |
| Sequence length | `2048, 8192` |
| Batch size | `32, 64, 128, 256, 512, 1024` |
| 阶段 | decode |

## 运行方式

```bash
python3 experiments/exp4/run_interconnect.py --quick --all
python3 experiments/exp4/run_interconnect.py --run
python3 experiments/exp4/run_interconnect.py --plot
```

## 输出

- `data/result_system_*_l*_b*.csv`
- `data/summary_interconnect.csv`
- `plots/figure10_11_interconnect.png`

预期趋势：更高的 interconnect 带宽会降低 communication latency，并在较大 batch 下显著改善 throughput。
