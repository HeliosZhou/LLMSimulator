# RoMe Figure 1 复现

本实验按论文 Figure 1 的思路，从 LLMSimulator runtime tensor trace 统计 DeepSeek-V3、Grok 1、Llama 3-405B 在 prefill 与 decode 阶段的 weight、activation、KV cache 访问数据大小分布。

## 口径

- 数据来源只使用 LLMSimulator runtime trace，不再保留模型参数公式估算 fallback。
- trace 由 `run_trace_figure1.py` 按论文 8-accelerator 上下文生成：DeepSeek-V3 使用 non-expert TP=1 / expert TP=1，Grok 1 使用 non-expert TP=8 / expert TP=1，Llama 3-405B 使用 non-expert TP=8。
- 当前 trace exporter 记录的是 `device_rank=0` 的 per-accelerator 执行路径；tensor shape 会受到 8-accelerator TP/EP/DP 配置影响，但 CSV 不是 8 个 device 的全量合并 trace。
- 绘图样本不是 raw trace 逐行直方统计，而是 trace-derived paper-style aggregation：去除输入重复计数，保留 operator/module 输出，KV cache 聚合为 logical cache block。
- C++ trace exporter 在 `Device::execution()` 与少数 module-level forward 中导出 tensor access。
- trace 中 `tag=weight/act/cache` 分别映射为 Figure 1 的 `weight/activation/KV cache`。
- 每个样本大小按 trace 中 tensor `shape * precision_byte` 计算。
- trace CSV 保留 `source/layer_type/module/tensor/process_tokens/device_rank`，用于回查箱线图样本来源。

## 生成文件

- `data/figure1_access_samples.csv`
- `data/figure1_access_summary.csv`
- `plots/rome_figure1_access_distribution.png`
- `plots/rome_figure1_access_distribution.pdf`

## 分组摘要

| 模型 | 阶段 | 类别 | 样本数 | 最小值 | 中位数 | 最大值 | >=4KiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek-V3 | prefill | weight | 890 | 448.00 KiB | 14.00 MiB | 883.75 MiB | 100.0% |
| DeepSeek-V3 | prefill | activation | 6763 | 512.00 KiB | 4.07 MiB | 1010.00 MiB | 100.0% |
| DeepSeek-V3 | prefill | kv_cache | 60 | 4.50 MiB | 4.50 MiB | 4.50 MiB | 100.0% |
| DeepSeek-V3 | decode | weight | 890 | 448.00 KiB | 14.00 MiB | 883.75 MiB | 100.0% |
| DeepSeek-V3 | decode | activation | 1596 | 64 B | 7.00 KiB | 1.00 MiB | 55.7% |
| DeepSeek-V3 | decode | kv_cache | 60 | 4.50 MiB | 4.50 MiB | 4.50 MiB | 100.0% |
| Grok 1 | prefill | weight | 386 | 96.00 KiB | 384.00 MiB | 1.50 GiB | 100.0% |
| Grok 1 | prefill | activation | 864 | 127.98 KiB | 95.99 MiB | 2.00 GiB | 100.0% |
| Grok 1 | prefill | kv_cache | 64 | 32.00 MiB | 32.00 MiB | 32.00 MiB | 100.0% |
| Grok 1 | decode | weight | 386 | 96.00 KiB | 384.00 MiB | 1.50 GiB | 100.0% |
| Grok 1 | decode | activation | 631 | 16 B | 12.00 KiB | 256.00 KiB | 79.7% |
| Grok 1 | decode | kv_cache | 64 | 32.00 MiB | 32.00 MiB | 32.00 MiB | 100.0% |
| Llama 3-405B | prefill | weight | 632 | 32.00 MiB | 104.00 MiB | 1.96 GiB | 100.0% |
| Llama 3-405B | prefill | activation | 1264 | 18.00 MiB | 128.00 MiB | 1002.00 MiB | 100.0% |
| Llama 3-405B | prefill | kv_cache | 126 | 16.00 MiB | 16.00 MiB | 16.00 MiB | 100.0% |
| Llama 3-405B | decode | weight | 632 | 32.00 MiB | 104.00 MiB | 1.96 GiB | 100.0% |
| Llama 3-405B | decode | activation | 1264 | 2.25 KiB | 16.00 KiB | 125.25 KiB | 90.0% |
| Llama 3-405B | decode | kv_cache | 126 | 16.00 MiB | 16.00 MiB | 16.00 MiB | 100.0% |

## 合理性检查

- 三个模型、prefill/decode、weight/activation/KV cache 共 18 个分组均有样本。
- raw trace 中大量重复输入和 simulator 内部 MoE expert 遍历会显著扭曲箱线图；当前报告使用聚合后的 trace-derived samples。
- Prefill activation 明显大于 decode activation，符合 prefill 处理整段 prompt、decode 单 token 生成的阶段差异。
- KV cache 样本均为 MiB 量级并全部大于 4 KiB，支撑 RoMe 使用 row-granularity access 的动机。
- DeepSeek-V3 decode activation 中仍有大量 64B/2KiB 小样本，来自 trace 中 LayerNorm/RoPE/element-wise 等 module-level 小 tensor；这比原先手写公式更接近真实执行路径，但也说明当前 trace 是 tensor/module 粒度，不是作者未公开的 kernel-level 原始 trace。

## 与原文 Figure 1 的关系

论文未公开 Figure 1 的逐样本 trace/CSV。因此当前结果是基于 LLMSimulator runtime trace 的复现口径，而不是作者原始数据的逐点复刻。若图形仍与论文有差异，主要可能来自 trace 取样粒度、并行切分口径、embedding/lm_head 是否纳入、MoE expert 触达口径等未公开细节。
