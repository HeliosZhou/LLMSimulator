# Exp0：Figure 2 - TPOT 与 Per-GPU Throughput 预实验

## 论文目标

Figure 2 对比 GPT-3、Llama4-Maverick 和 DeepSeek-R1 在不同 sequence length 与 batch size 下的：

- Time per output token（TPOT）
- Per-GPU throughput

这张图属于论文前置动机实验，用来说明 DeepSeek-R1 由于 MLA 和 MoE 的组合，在更大 batch 下可以获得更好的吞吐-延迟表现。

## 模拟器配置

主脚本：

```bash
python3 experiments/exp0/run_tpot_throughput.py
```

模型映射：

| 论文模型 | 模拟器模型名 |
| --- | --- |
| GPT-3 | `gpt3_175B` |
| Llama4-Maverick | `llama4_maverick` |
| DeepSeek-R1 | `deepseekV3` |

默认 sweep：

| 参数 | 取值 |
| --- | --- |
| GPU 系统 | 32 B200 GPU（`num_node=4`, `num_device=8`） |
| Sequence length | `2048, 8192` |
| Batch size | `32, 64, 128, 256` |
| 精度 | 2 bytes，近似 BF16 |
| 阶段 | decode |

## 运行方式

```bash
python3 experiments/exp0/run_tpot_throughput.py --quick --all
python3 experiments/exp0/run_tpot_throughput.py --run
python3 experiments/exp0/run_tpot_throughput.py --plot
```

## 输出

- `data/result_model_*_l*_b*.csv`
- `data/summary_tpot_throughput.csv`
- `plots/figure2_tpot_throughput.png`

summary 中包含：

- `latency_ns`：TPOT
- `throughput_tps`：系统总吞吐
- `per_gpu_throughput_tps`：每 GPU 吞吐，计算方式为 `throughput_tps / 32`

## 与 Figure 9 的区别

`exp3` 复现 Figure 9 的 throughput-latency 曲线；本实验复现 Figure 2 的前置对比图，重点是把 TPOT 和 per-GPU throughput 按模型、sequence length、batch size 分开展示。
