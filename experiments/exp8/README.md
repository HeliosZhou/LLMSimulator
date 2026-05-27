# Exp8：Figure 4 - 模型权重与 KV Cache 容量对比

## 论文目标

Figure 4 对比 GPT-3、Llama4-Maverick 和 DeepSeek-R1 在 BF16 下的内存占用：

- 每 token 实际访问/计算的 activated parameters
- 完整模型参数占用，并拆分 attention weights 与 FFN/MoE weights
- 8M tokens 对应的 KV cache 占用

核心结论是：DeepSeek-R1 虽然总参数量更大，但 MLA 显著缩小 KV cache，MoE 又减少每 token 激活参数量，因此可以支持更大的 batch。

## 实验性质

本实验是**解析统计图生成**，不是运行 LLMSimulator。脚本使用论文正文给出的总参数、激活参数、KV cache/token 数值，并用模型维度公式估算 attention weight，剩余参数归入 FFN/MoE weight。

## 脚本

```bash
python3 experiments/exp8/run_memory_footprint.py
```

默认参数：

| 模型 | 总参数 | 激活参数 | KV cache / token |
| --- | --- | --- | --- |
| GPT-3 | 175B | 175B | 4.5 MB |
| Llama4-Maverick | 400B | 17B | 192 KB |
| DeepSeek-R1 | 671B | 37B | 68.6 KB |

KV cache 总量按 8M tokens 和 BF16 口径计算。

## 运行方式

```bash
python3 experiments/exp8/run_memory_footprint.py --all
python3 experiments/exp8/run_memory_footprint.py --plot
```

## 输出

- `data/summary_memory_footprint.csv`
- `plots/figure4_memory_footprint.png`

## 注意

该图用于复现论文 Figure 4 的容量关系和主要结论。模型参数分解不是由模拟器执行路径测量得到，而是由论文中的公开参数和 KV cache 公式整理得到。
