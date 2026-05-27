# Exp1：Figure 6 - MLA Attention 延迟分解

## 论文目标

Figure 6 分析 DeepSeek-R1 在 decode 阶段的 MLA attention block 延迟，对比：

- 使用 MLA layer reordering / absorption：`use_absorb=true`
- 不使用 MLA layer reordering：`use_absorb=false`

论文展示三部分结果：

- 归一化 attention 延迟：`w/o reordering / w/ reordering`
- 无重排时的 attention block 延迟分解
- 有重排时的 attention block 延迟分解

## 模拟器配置

主脚本：

```bash
python3 experiments/exp1/run_attention_breakdown.py
```

默认 sweep：

| 参数 | 取值 |
| --- | --- |
| 模型 | `deepseekV3` |
| GPU | `B200` |
| 阶段 | decode（`decode_mode=true`） |
| Batch per GPU | `32, 64, 128, 256` |
| Sequence length | `2048, 4096, 8192` |
| 是否重排 | `on, off` |
| 精度 | BF16 / 2 byte |
| 迭代次数 | 3 |

论文使用 32 张 B200 GPU。当前脚本按 `4 node x 8 device` 配置 32 张 B200，并将图中的 batch 解释为 per-GPU batch；运行时实际 `max_batch_size = batch_per_gpu * 32`。

## 运行方式

快速验证：

```bash
python3 experiments/exp1/run_attention_breakdown.py --quick --all
```

完整 sweep：

```bash
python3 experiments/exp1/run_attention_breakdown.py --run
```

基于已有数据绘图：

```bash
python3 experiments/exp1/run_attention_breakdown.py --plot
```

兼容入口：

```bash
python3 experiments/exp1/plot_attention_breakdown.py --plot
```

## 输出

生成文件：

- `data/result_b*_l*_absorb_*.csv`
- `data/summary_attention_breakdown.csv`
- `plots/figure6_attention_breakdown.png`

CSV 和 PNG 都是生成文件，已由 `.gitignore` 忽略。

## 指标映射

脚本将 LLMSimulator CSV 字段映射到 Figure 6 的几类延迟：

| Figure 类别 | CSV 字段 |
| --- | --- |
| KV decompress | 无重排时为 `kv_up_proj`；有重排时为 `tr_k_up_proj + v_up_proj` |
| Score + Context | `atten_sum + atten_gen` |
| Out projection | `o_proj` |
| Etc | 其他 projection、RoPE、layernorm、residual、communication |

预期趋势：无重排时 KV 解压和 core attention 随 `B`、`L` 增大成为主要瓶颈；重排后 KV 解压基本消失，attention 总延迟显著下降。
