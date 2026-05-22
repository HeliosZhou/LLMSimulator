# Exp2：Figure 8 - TP 对 MLA Attention 的影响

## 论文目标

Figure 8 评估 layer reordering 之后，tensor parallelism 是否还能有效降低 MLA attention 延迟。论文结论是：重排后的 MLA 中所有 head 共享压缩 KV cache，TP shard 仍需要访问 CKV，并且 arithmetic intensity 可能下降，因此 TP 对 attention latency 的收益有限。

## 模拟器配置

主脚本：

```bash
python3 experiments/exp2/run_tp_attention.py
```

默认 sweep：

| 参数 | 取值 |
| --- | --- |
| 模型 | `deepseekV3` |
| GPU | `B200` |
| 阶段 | decode |
| Sequence length | `4096` |
| Batch size | `32, 64, 128` |
| TP degree | `1, 2, 4, 8`，通过 `none_expert_tensor_degree` 设置 |
| 是否重排 | `on, off` |

## 运行方式

```bash
python3 experiments/exp2/run_tp_attention.py --quick --all
python3 experiments/exp2/run_tp_attention.py --run
python3 experiments/exp2/run_tp_attention.py --plot
```

## 输出

- `data/result_b*_l4096_tp*_absorb_*.csv`
- `data/summary_tp_attention.csv`
- `plots/figure8_tp_attention.png`

预期趋势：TP 会降低部分 projection 开销，但对重排后的 attention 总延迟无法带来线性加速。
