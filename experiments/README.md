# 《Rethinking LLM Inference Bottlenecks》复现说明

本目录用于组织论文《Rethinking LLM Inference Bottlenecks: Insights from Latent Attention and Mixture-of-Experts》（arXiv:2507.15465v3, 2026-01-29）中可复现的图表与实验。为了尽量做到与论文图号一一对应，当前目录同时包含两类内容：

- **模拟器实验**：调用 LLMSimulator 运行不同模型、batch、sequence length、并行度、interconnect 或 PIM 配置。
- **解析生成图**：不运行模拟器，而是根据论文参数、模型规模和解析公式生成图，例如 roofline 和 memory footprint。

## 图表覆盖范围

| 论文内容 | 当前状态 | 目录 | 复现方式 | 说明 |
| --- | --- | --- | --- | --- |
| Figure 1 | 记录为非实验图 | - | 不运行 | Decoder 结构示意图，没有对应仿真 sweep |
| Figure 2 | 已配置 | `exp0` | LLMSimulator sweep | TPOT 与 per-GPU throughput 对比 |
| Figure 3 | 已配置 | `exp7` | 解析生成 | Roofline 风格图，复现分析结构和主要趋势 |
| Figure 4 | 已配置 | `exp8` | 解析生成 | 权重、激活参数与 KV cache 容量对比 |
| Figure 5 | 记录为非实验图 | - | 不运行 | MLA 计算流程示意图 |
| Figure 6 | 已配置 | `exp1` | LLMSimulator sweep | MLA attention 有/无重排的延迟分解 |
| Figure 7 | 记录为非实验图 | - | 不运行 | TP 下 MLA 计算流程示意图 |
| Figure 8 | 已配置 | `exp2` | LLMSimulator sweep | TP degree 对 MLA attention 的影响 |
| Figure 9 | 已配置 | `exp3` | LLMSimulator sweep | 多模型 throughput-latency 曲线 |
| Figure 10 | 已配置 | `exp4` | LLMSimulator sweep | DeepSeek-R1 execution breakdown 与 interconnect 敏感性 |
| Figure 11 | 已配置 | `exp4` | LLMSimulator sweep | 32 GPU x8 与 256 GPU 部署粒度对比 |
| Figure 12 | 已配置 | `exp5` | LLMSimulator sweep | skewed expert routing 下的 throughput-latency |
| Figure 13 | 已配置 | `exp5` | LLMSimulator sweep | skew 下的 load imbalance 与部署粒度对比 |
| Figure 14 | 已配置 | `exp6` | LLMSimulator sweep | GPU-only 与 GPU+PIM 归一化吞吐对比 |
| Table I-V | 已在文档中说明 | - | 不运行 | 硬件参数、模型符号、公式和附录表格 |

因此，除 Figure 1/5/7 这类纯结构示意图和 Table I-V 这类表格外，论文中的主要结果图都已经配置了对应目录。

## 对照论文后的复现边界

- 论文 §VI 使用 B200、BF16、decode-only、prefill/decode disaggregation、默认 5th-gen NVLink（900GB/s），并在特别说明时使用 InfiniBand XDR（100GB/s）。模拟器实验均按这些口径设置。
- `exp1` 的 Figure 6 复现已经人工确认正确；其它实验已按论文图轴和实验设定校准脚本/README，但数值仍建议用完整 `--run --overwrite` 重新生成后再逐图核验。
- `exp7` 和 `exp8` 不是 LLMSimulator 逐点仿真，而是按论文 Table I、正文参数与公式生成的解析复现图；它们用于复现图形结构和主要趋势，不声称重现实机测量点。
- 部分目录中已有早期 CSV 使用旧文件名（例如 `_b` 而不是 `_tb`/`_bpg`）。脚本的 `collect()` 已兼容旧命名，新的完整复现实验会使用 README 中列出的新命名。

## 目录结构

| 目录 | 对应内容 | 主脚本 | 输出图 | 类型 |
| --- | --- | --- | --- | --- |
| `exp0` | Figure 2 | `run_tpot_throughput.py` | `figure2_tpot_throughput.png` | 模拟器实验 |
| `exp1` | Figure 6 | `run_attention_breakdown.py` | `figure6_attention_breakdown.png` | 模拟器实验 |
| `exp2` | Figure 8 | `run_tp_attention.py` | `figure8_tp_attention.png` | 模拟器实验 |
| `exp3` | Figure 9 | `run_throughput_latency.py` | `figure9_throughput_latency.png` | 模拟器实验 |
| `exp4` | Figure 10/11 | `run_interconnect.py` | `figure10_11_interconnect.png` | 模拟器实验 |
| `exp5` | Figure 12/13 | `run_skew.py` | `figure12_13_skew.png` | 模拟器实验 |
| `exp6` | Figure 14 | `run_pim.py` | `figure14_pim.png` | 模拟器实验 |
| `exp7` | Figure 3 | `run_roofline.py` | `figure3_roofline.png` | 解析生成图 |
| `exp8` | Figure 4 | `run_memory_footprint.py` | `figure4_memory_footprint.png` | 解析生成图 |
| `common` | 通用工具 | `sim_utils.py` | - | 公共代码 |

## 通用运行方式

运行模拟器实验前先构建 LLMSimulator：

```bash
cmake --build build -j 4
```

每个实验脚本都支持统一参数：

```bash
python3 experiments/expN/script.py --quick --all
python3 experiments/expN/script.py --run
python3 experiments/expN/script.py --plot
```

`--quick --all` 会运行小规模 smoke test 并绘图，适合改动后快速验证。

`--run` 会运行完整配置网格。部分实验包含 32/256 GPU 配置，耗时较长。

`--plot` 会读取已有 CSV 并重新生成 summary 和图片。

常用可选参数：

```bash
--overwrite          # 即使结果 CSV 已存在，也重新运行该配置点
--timeout 900        # 单个配置点的模拟器超时时间，单位为秒
```

对于 `exp7` 和 `exp8` 这类解析生成图，`--run` 不会调用模拟器，只会生成 summary 和图片。

## 生成文件与 Git

实验产生的 CSV、每个配置点的 YAML、绘图 PNG 和 Python 缓存都属于生成文件，已在 `.gitignore` 中忽略：

```text
experiments/**/data/*.csv
experiments/**/data/configs/*.yaml
experiments/**/plots/*.png
__pycache__/
```

建议纳入版本管理的是实验脚本、`README.md` 文档和 `common` 中的共享工具代码。

## 为复现新增的模拟器能力

为了覆盖论文实验，当前工程补充了两个小的模拟器能力：

- 新增 `gpt3_175B` 模型配置，用于 `exp0` 和 `exp3` 的 GPT-3 对比。
- 支持从 YAML 覆盖 `logic_x`、`logic_op_b`、`pim_x`、`pim_op_b`，并新增 300GB/s interconnect 档位（`infiniband_gen: 2400`），用于 `exp4` 和 `exp6`。

对应代码位于 `src/model/model_config.h` 和 `eval/test.cpp`。
