# Exp3：Figure 9 - 多模型吞吐-延迟对比

## 论文目标

Figure 9 对比 GPT-3、Llama4-Maverick 和 DeepSeek-R1 在 decode 阶段的 throughput-latency 曲线。核心观点是：MLA 减小 KV cache，MoE 降低激活计算量，两者结合让 DeepSeek-R1 能使用更大的有效 batch，从而获得更好的吞吐-延迟折中。

## 模拟器配置

主脚本：

```bash
python3 experiments/exp3/run_throughput_latency.py
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
| Total batch | GPT-3: `96, 192, ..., 1728`; Llama4/DeepSeek-R1: `1152, 2304, ..., 20736` |
| 精度 | 2 bytes，近似 BF16 |
| 阶段 | decode |

论文 Figure 9 的横轴是 throughput-latency 曲线，图中标注的 batch 是系统总 batch。脚本按 total batch 写入 `max_batch_size`，并使用论文图中的 batch 网格：GPT-3 图轴到 1728，Llama4-Maverick/DeepSeek-R1 图轴到 20736。

## 运行方式

```bash
python3 experiments/exp3/run_throughput_latency.py --quick --all
python3 experiments/exp3/run_throughput_latency.py --run
python3 experiments/exp3/run_throughput_latency.py --plot
```

## 输出

- `data/result_model_*_l*_tb*.csv`
- `data/summary_throughput_latency.csv`
- `plots/figure9_throughput_latency.png`

绘图中横轴为 TPOT（即 CSV 中的 `latency`），纵轴为 system throughput，计算方式为模拟器 CSV 中实际 `batchsize / latency`。脚本也会读取旧命名的 `data/result_model_*_l*_b*.csv`，但推荐重新跑一次完整 sweep 以获得论文对齐的 batch 点。
