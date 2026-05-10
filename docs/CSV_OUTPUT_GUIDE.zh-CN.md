# LLMSimulator CSV 输出字段说明

本文档解释 `./run config.yaml` 生成的 CSV 文件应该如何阅读，以及 CSV 中每一列在源码中的来源。

本文以当前结果文件为主要例子：

```text
data/deepseekV3_synthesis_1024_10_GPU_N1_D8_TP1_DP8_maxbatch32_maxprocess524288_iter5_skew0_precision_byte1_parallel_execution0_decode.csv
```

这个 CSV 不是模型生成文本，而是一次仿真实验的统计结果表。它记录了仿真过程中的调度状态、单步时间、算子耗时、能耗拆分和 OOM 状态。

## 1. CSV 是在哪里生成的

整体调用链如下：

1. `eval/test.cpp` 读取 YAML 配置。
2. `eval/test.cpp` 创建 `SystemConfig`、`ModelConfig`、`Scheduler`、`Cluster` 和 `Model`。
3. `eval/test.cpp` 根据配置拼出输出 CSV 文件名。
4. `Cluster::runIteration()` 创建 CSV 文件并写入表头。
5. `Cluster::runIterationMixed()` 或 `Cluster::runIterationSumGenSplit()` 执行仿真迭代。
6. 每轮仿真把统计结果填入 `Stat`。
7. `Cluster::exportToCSV()` 把 `Stat` 按列写入 CSV。

关键源码位置：

| 作用 | 文件 | 行范围 |
|---|---|---|
| 读取配置文件 | `eval/test.cpp` | 14-39 |
| 解析硬件、优化、模型、输入输出长度 | `eval/test.cpp` | 41-226 |
| 创建 `Scheduler`、`Cluster`、`Model` | `eval/test.cpp` | 224-233 |
| 拼接输出 CSV 文件名 | `eval/test.cpp` | 247-337 |
| 触发仿真并导出 CSV | `eval/test.cpp` | 342-354 |
| CSV 表头 | `src/hardware/cluster.cpp` | 424-436 |
| CSV 行写入顺序 | `src/hardware/cluster.cpp` | 371-421 |
| 每行数据结构 `Stat` | `src/hardware/stat.h` | 8-75 |
| 时间、能耗单位别名 | `src/common/type.h` | 6-10 |

单位：

| 类型 | 源码类型 | 单位 |
|---|---|---|
| 时间 | `time_ns` | ns |
| 能耗 | `energy_nJ` | nJ |
| 硬件指标 | `hw_metric` | double，具体含义由对应计算逻辑决定 |

## 2. 当前 CSV 文件名的含义

当前文件名：

```text
deepseekV3_synthesis_1024_10_GPU_N1_D8_TP1_DP8_maxbatch32_maxprocess524288_iter5_skew0_precision_byte1_parallel_execution0_decode.csv
```

可以拆成下面这些实验参数：

| 文件名片段 | 含义 | 当前值 | 源码位置 |
|---|---|---:|---|
| `deepseekV3` | 模型名 | `deepseekV3` | `eval/test.cpp:24`, `eval/test.cpp:163-181` |
| `synthesis` | 输入数据类型，表示合成请求 | `synthesis` | `eval/test.cpp:31`, `eval/test.cpp:204-213` |
| `1024_10` | 输入长度和输出长度 | input 1024, output 10 | `eval/test.cpp:32-34`, `eval/test.cpp:206-209` |
| `GPU` | 处理器类型 | `GPU` | `eval/test.cpp:25-26`, `eval/test.cpp:140-157` |
| `N1_D8` | 节点数和每节点设备数 | 1 node, 8 devices | `eval/test.cpp:28-29`, `eval/test.cpp:91-92` |
| `TP1_DP8` | 非专家部分 tensor parallel 和 data parallel | TP=1, DP=8 | `eval/test.cpp:184-187`, `eval/test.cpp:242-243` |
| `maxbatch32` | 最大 batch size | 32 | `eval/test.cpp:36`, `eval/test.cpp:224-226` |
| `maxprocess524288` | 每个 sum/prefill 阶段最多处理 token 数 | 524288 | `eval/test.cpp:38`, `eval/test.cpp:219-226` |
| `iter5` | 仿真迭代轮数 | 5 | `eval/test.cpp:34`, `eval/test.cpp:239-243` |
| `skew0` | MoE expert 选择偏斜度，文件名中是 `skewness * 10` 的整数 | 0 | `eval/test.cpp:193-194` |
| `precision_byte1` | 数值精度字节数 | 1 byte | `eval/test.cpp:196-199`, `eval/test.cpp:245` |
| `parallel_execution0` | 是否并行执行 high/low processor | off | `eval/test.cpp:99-100` |
| `decode` | decode 模式 | on | `eval/test.cpp:121-138`, `eval/test.cpp:278-305` |

注意：如果 `max_process_token` 在配置里写成 `0`，源码会把它改成 `65536 * 8 = 524288`。所以你当前文件名里是 `maxprocess524288`。对应代码在 `eval/test.cpp:219-226`。

## 3. 每一行 CSV 代表什么

CSV 中每一行是一条统计记录。主要由 `type` 列区分：

| `type` | 含义 | 生成位置 |
|---|---|---|
| `t2t` | token-to-token 过程统计，通常表示一次 decode step 或一次仿真迭代的阶段性统计 | `src/hardware/cluster.cpp:488-518`, `src/hardware/cluster.cpp:562-585` |
| `t2ft` | time-to-first-token，首 token 延迟统计 | `src/hardware/cluster.cpp:629-635` |
| `e2e` | end-to-end，请求完整完成延迟统计 | `src/hardware/cluster.cpp:621-628` |
| `sum` | disaggregated system 下 sum/prefill machine 的阶段统计 | `src/hardware/cluster.cpp:590-603` |

你当前这份 `decode.csv` 前几行都是 `t2t`。这表示它主要在记录 decode 阶段每个生成步的累计时间、单步延迟、当前 batch 状态、算子时间和能耗。

在当前样例里，前几行 `seqlen` 从 `1025`、`1026`、`1027` 增长。原因是 decode 模式下新请求会从 `current_len = input_len` 开始，每轮生成 1 个 token。

相关源码：

| 作用 | 文件 | 行范围 |
|---|---|---|
| `Sequence` 初始化 `current_len = 0`, `total_len = input_len + output_len - 1` | `src/scheduler/sequence.cpp` | 6-30 |
| decode 模式把 `current_len` 设置为 `input_len` | `src/scheduler/scheduler.cpp` | 84-90 |
| generation 序列每轮处理 1 个 token | `src/scheduler/scheduler.cpp` | 277-289 |
| 每轮执行后更新 `current_len` | `src/scheduler/sequence.cpp` | 35-50 |

## 4. 行类型和整体时间列

| CSV 列 | `Stat` 字段 | 含义 | 单位 | 主要源码位置 |
|---|---|---|---|---|
| `iter_info` | `iter_info` | 统计记录类型标记。`1` 通常表示仿真过程中的阶段性记录，`0` 通常表示请求级延迟记录，如 `t2ft/e2e`。 | 无 | `src/hardware/cluster.cpp:488-490`, `src/hardware/cluster.cpp:617-619` |
| `split` | `split` | disaggregated system 相关标记。普通 mixed 执行通常为 0；`config.disagg_system` 时会被置为 1。 | 无 | `src/hardware/cluster.cpp:493-495` |
| `type` | `type` | 记录类型：`t2t`、`t2ft`、`e2e`、`sum`。 | 字符串 | `src/hardware/cluster.cpp:488-490`, `src/hardware/cluster.cpp:590-603`, `src/hardware/cluster.cpp:621-635` |
| `time` | `time` | 当前记录对应的累计仿真时间。对 `t2t` 来说，通常是多轮迭代累加后的 total time。 | ns | `src/hardware/cluster.cpp:486-492`, `src/hardware/cluster.cpp:558-566`, `src/hardware/cluster.cpp:617-619` |
| `latency` | `latency` | 当前记录的延迟。`t2t` 下是本轮执行耗时；`t2ft` 是首 token 延迟；`e2e` 是请求总延迟。 | ns | `src/hardware/cluster.cpp:658-681`, `src/hardware/cluster.cpp:621-635` |
| `queueing_delay` | `queueing_delay` | 请求进入 running queue 前的排队时间。只有启用注入率相关逻辑时才明显非零。 | ns | `src/scheduler/scheduler.cpp:357-380`, `src/hardware/cluster.cpp:621-635` |
| `arrival_time` | `arrival_time` | 请求到达时间。 | ns | `src/scheduler/scheduler.cpp:68-75`, `src/hardware/cluster.cpp:621-635` |
| `seq_queue_size` | `seq_queue_size` | 当前仍在等待进入 running queue 的请求数量。 | 个 | `src/hardware/cluster.cpp:497-508`, `src/hardware/cluster.cpp:568-579` |

## 5. 请求、batch 和序列状态列

| CSV 列 | `Stat` 字段 | 含义 | 单位 | 主要源码位置 |
|---|---|---|---|---|
| `input_len` | `input_len` | 请求输入长度。对 `t2ft/e2e` 记录会填入具体请求的 input length；对普通 `t2t` 行通常保持默认 0。 | token | `src/hardware/cluster.cpp:621-635`, `src/scheduler/sequence.cpp:6-14` |
| `output_len` | `output_len` | 请求输出长度。对 `e2e` 记录会填入；部分 `t2t/t2ft` 行可能为 0。 | token | `src/hardware/cluster.cpp:621-635`, `src/scheduler/sequence.cpp:6-14` |
| `num_sum_iter` | `num_sum_iter` | prefill/sum 阶段处理了多少轮。`Sequence::update()` 在 sum stage 时递增。 | 次 | `src/scheduler/sequence.cpp:35-50`, `src/hardware/cluster.cpp:621-635` |
| `mixed` | `is_mixed` | 当前是否处在 mixed stage。源码用 `scheduler->hasSumSeq()` 判断是否存在 sum/prefill 序列。 | 0/1 | `src/hardware/cluster.cpp:658-681`, `src/scheduler/scheduler.cpp:24-31` |
| `batchsize` | `batchsize` | 当前 running queue 中总请求数。 | 个 | `src/hardware/cluster.cpp:658-665`, `src/scheduler/sequence.cpp:97-105` |
| `numtoken` | `process_token` | 当前迭代实际处理的 token 数。decode 阶段一般每个 active generation sequence 处理 1 个 token。 | token | `src/hardware/cluster.cpp:658-665`, `src/scheduler/scheduler.cpp:260-308`, `src/scheduler/sequence.cpp:99-126` |
| `num_sum_seq` | `sum_seq` | 当前处于 sum/prefill 阶段的序列数。 | 个 | `src/hardware/cluster.cpp:658-665`, `src/scheduler/sequence.cpp:59-68` |
| `num_gen_seq` | `gen_seq` | 当前处于 generation/decode 阶段的序列数。 | 个 | `src/hardware/cluster.cpp:658-665`, `src/scheduler/sequence.cpp:70-78` |
| `seqlen` | `average_seq_len` | generation 序列的平均当前长度。decode 模式下通常随生成步增长。 | token | `src/hardware/cluster.cpp:658-665`, `src/scheduler/sequence.cpp:128-140` |
| `sum_attention_opb` | `sum_attention_opb` | sum/prefill attention 的 operation per byte，来自 `AttentionSum` 时间戳的平均 `opb`。 | OP/B | `src/hardware/cluster.cpp:828-836`, `src/hardware/cluster.cpp:1072-1080`, `src/module/timeboard.h:50` |

## 6. 注意力和投影算子时间列

这些列单位都是 ns。它们不是直接由 attention kernel 写 CSV，而是模块执行时产生 `TimeStamp`，之后 `Cluster::setTimeBreakDown()` 按名字搜索时间戳并聚合。

时间戳机制：

| 作用 | 文件 | 行范围 |
|---|---|---|
| 模块执行前后 push/pop time stamp | `src/module/module_graph.cpp` | 177-188 |
| 根据执行前后 `StatusBoard` 计算 duration 和 energy delta | `src/module/timeboard.cpp` | 20-76 |
| 按名字查找 time stamp | `src/module/timeboard.cpp` | 78-87 |
| duration 定义为 `end_time - start_time` | `src/module/timeboard.h` | 32-34 |

普通 attention 模型和 MLA 模型会填不同列。当前 `deepseekV3` 属于 MLA 路径，所以 `qkvgen` 通常为 0，而 `q_down_proj`、`kv_down_proj`、`kr_proj`、`q_up_proj`、`qr_proj` 等列更重要。

| CSV 列 | 含义 | 普通 attention 来源 | MLA / DeepSeekV3 来源 | 聚合源码位置 |
|---|---|---|---|---|
| `qkvgen` | 普通注意力中的 QKV projection 时间。DeepSeekV3 MLA 不走整体 QKV projection，因此常为 0。 | `src/module/layer.cpp:15-20`, `src/module/layer.cpp:47-50` | 不适用 | `src/hardware/cluster.cpp:701-749` |
| `q_down_proj` | MLA 中 query latent down projection。 | 不适用 | `src/module/layer.cpp:65-67`, `src/module/layer.cpp:210` | `src/hardware/cluster.cpp:867-944` |
| `kv_down_proj` | MLA 中 KV latent down projection。 | 不适用 | `src/module/layer.cpp:74-76`, `src/module/layer.cpp:211` | `src/hardware/cluster.cpp:867-951` |
| `kr_proj` | MLA 中 rope key projection。 | 不适用 | `src/module/layer.cpp:78-80`, `src/module/layer.cpp:212` | `src/hardware/cluster.cpp:867-958` |
| `q_up_proj` | MLA 中 query up projection。 | 不适用 | absorb: `src/module/layer.cpp:96-100`, `src/module/layer.cpp:224`; baseline: `src/module/layer.cpp:140-144`, `src/module/layer.cpp:270` | `src/hardware/cluster.cpp:871-965` |
| `qr_proj` | MLA 中 query rope projection。 | 不适用 | absorb: `src/module/layer.cpp:102-106`, `src/module/layer.cpp:225`; baseline: `src/module/layer.cpp:146-150`, `src/module/layer.cpp:271` | `src/hardware/cluster.cpp:871-972` |
| `kv_up_proj` | baseline MLA 中 KV up projection。absorb MLA 不使用该列，通常为 0。 | 不适用 | `src/module/layer.cpp:162-166`, `src/module/layer.cpp:272` | `src/hardware/cluster.cpp:873-979` |
| `tr_k_up_proj` | absorb MLA 中 transposed K up projection。 | 不适用 | `src/module/layer.cpp:112-116`, `src/module/layer.cpp:228` | `src/hardware/cluster.cpp:876-986` |
| `v_up_proj` | absorb MLA 中 V up projection。 | 不适用 | `src/module/layer.cpp:124-128`, `src/module/layer.cpp:232` | `src/hardware/cluster.cpp:876-993` |
| `atten_sum` | sum/prefill attention 时间。 | `src/module/parallel.cpp:154-158`, `src/module/parallel.cpp:191` | `src/module/parallel.cpp:299-303`, `src/module/parallel.cpp:335` | `src/hardware/cluster.cpp:751-756`, `src/hardware/cluster.cpp:995-1000` |
| `atten_gen` | generation/decode attention 时间。 | `src/module/parallel.cpp:160-164`, `src/module/parallel.cpp:192` | `src/module/parallel.cpp:305-309`, `src/module/parallel.cpp:336` | `src/hardware/cluster.cpp:758-763`, `src/hardware/cluster.cpp:1002-1007` |
| `o_proj` | attention output projection 时间。 | `src/module/layer.cpp:29-33`, `src/module/layer.cpp:49` | absorb: `src/module/layer.cpp:130-133`, `src/module/layer.cpp:233`; baseline: `src/module/layer.cpp:175-179`, `src/module/layer.cpp:282` | `src/hardware/cluster.cpp:765-770`, `src/hardware/cluster.cpp:1009-1014` |

## 7. FFN、MoE、通信和小算子时间列

| CSV 列 | 含义 | 主要源码位置 | 聚合源码位置 |
|---|---|---|---|
| `ffn` | 非 MoE decoder 中普通 feedforward 时间。对 MoE decoder 主路径可能为 0，因为专家部分走 `expertFFN`。 | `src/module/decoder.cpp:50-60`, `src/module/decoder.cpp:80-86`; FFN 具体实现在 `src/module/layer.cpp:290` 之后 | `src/hardware/cluster.cpp:772-777`, `src/hardware/cluster.cpp:1016-1021` |
| `expert_ffn` | MoE expert FFN 计算时间，不包括 MoE scatter/gather/all-reduce 通信时间。源码会从 `expertFFN` 总时间中减去 MoE 通信时间。 | `src/module/decoder.cpp:132-158`, `src/module/expert.cpp:13-134`, `src/module/expert.cpp:136-207` | `src/hardware/cluster.cpp:779-798`, `src/hardware/cluster.cpp:1023-1042` |
| `communication` | 通信时间，当前聚合 `all_reduce`、`moe_scatter`、`moe_gather` 等时间戳。 | `src/module/communication.cpp`, `src/module/expert.cpp:44-47`, `src/module/expert.cpp:103-112`, `src/module/expert.cpp:148-194` | `src/hardware/cluster.cpp:707-713`, `src/hardware/cluster.cpp:800-803`, `src/hardware/cluster.cpp:886-892`, `src/hardware/cluster.cpp:1044-1047` |
| `rope` | RoPE 位置编码时间，包括 `k_rope`、`q_rope`。 | `src/module/layer.cpp:82-84`, `src/module/layer.cpp:108-110`, `src/module/layer.cpp:152-154`, `src/module/layer.cpp:213/226/255/271` | `src/hardware/cluster.cpp:715-716`, `src/hardware/cluster.cpp:805-808`, `src/hardware/cluster.cpp:894-895`, `src/hardware/cluster.cpp:1049-1052` |
| `layernorm` | LayerNorm 时间。DeepSeekV3 MLA 会额外统计 latent q/kv layernorm。 | `src/module/decoder.cpp:26-48`, `src/module/decoder.cpp:108-130`, `src/module/layer.cpp:86-92`, `src/module/layer.cpp:215-216` | `src/hardware/cluster.cpp:718-719`, `src/hardware/cluster.cpp:810-813`, `src/hardware/cluster.cpp:897-900`, `src/hardware/cluster.cpp:1054-1057` |
| `residual` | residual add 时间。注意当前 `MoEDecoder` 中 `residual_1/residual_2` 构造为 `LayerNorm::Create`，但时间戳名字仍会按模块名聚合。 | `src/module/decoder.cpp:42-47`, `src/module/decoder.cpp:65-67`, `src/module/decoder.cpp:124-139` | `src/hardware/cluster.cpp:721-722`, `src/hardware/cluster.cpp:815-818`, `src/hardware/cluster.cpp:902-903`, `src/hardware/cluster.cpp:1059-1062` |

## 8. 能耗列说明

能耗单位都是 nJ。

总能耗统计链路：

1. 每个算子执行后产生 `ExecStatus`，其中包含 DRAM command count、FLOPs、duration 等。
2. `TopModuleGraph::set_pop_status()` 根据 `ExecStatus` 累加 `StatusBoard` 的时间和能耗。
3. `TopModuleGraph::getDeviceEnergy()` 返回单设备总能耗数组。
4. `Cluster::getTotalEnergy()` 对所有设备求和。
5. `Cluster::runIteration*()` 把能耗数组填入 `Stat`。
6. `exportToCSV()` 写入 CSV。

关键源码：

| 作用 | 文件 | 行范围 |
|---|---|---|
| `ExecStatus` 和 `StatusBoard` 字段定义 | `src/module/status.h` | 13-117 |
| 每个 module pop 时累加时间和能耗 | `src/module/module_graph.cpp` | 237-307 |
| 单设备能耗数组 | `src/module/module_graph.cpp` | 327-332 |
| 多设备求和 | `src/hardware/cluster.cpp` | 349-360 |
| 将总能耗写入 `Stat` | `src/hardware/cluster.cpp` | 497-506, 568-577 |
| 按算子时间戳拆分 FC/Attn/MoE 能耗 | `src/hardware/cluster.cpp` | 737-826, 932-1070 |

| CSV 列 | 含义 | 计算方式 / 来源 |
|---|---|---|
| `act_energy` | DRAM ACT 能耗 | `exec_status.act_count * kACT_energy_j_`，见 `src/module/module_graph.cpp:258-260` |
| `read_energy` | DRAM READ 能耗 | `exec_status.read_count * kREAD_energy_j_`，见 `src/module/module_graph.cpp:260-262` |
| `write_energy` | DRAM WRITE 能耗 | `exec_status.write_count * kWRITE_energy_j_`，见 `src/module/module_graph.cpp:262-263` |
| `all_act_energy` | all-bank ACT 类能耗 | `exec_status.all_act_count * kALL_ACT_energy_j_`，见 `src/module/module_graph.cpp:265-266` |
| `all_read_energy` | all-bank READ 类能耗 | `exec_status.all_read_count * kALL_READ_energy_j_`，见 `src/module/module_graph.cpp:267-268` |
| `all_write_energy` | all-bank WRITE 类能耗 | `exec_status.all_write_count * kALL_WRITE_energy_j_`，见 `src/module/module_graph.cpp:269-270` |
| `mac_energy` | 计算 MAC 能耗 | `exec_status.flops * kMAC_energy_j_`，见 `src/module/module_graph.cpp:272-274` |
| `total_energy` | 总能耗 | 上面 DRAM 能耗和 MAC 能耗求和，见 `src/module/module_graph.cpp:327-332` |
| `fc_dram` | 全连接/投影/FFN 相关 DRAM 能耗 | 在 `setTimeBreakDown()` 中对 QKV/O/FFN 等时间戳调用 `getDramEnergy()` 聚合 |
| `fc_comp` | 全连接/投影/FFN 相关计算能耗 | 在 `setTimeBreakDown()` 中对 QKV/O/FFN 等时间戳调用 `getCompEnergy()` 聚合 |
| `attn_dram` | attention 相关 DRAM 能耗 | 普通 attention 或 MLA attention/projection 时间戳的 DRAM 能耗聚合 |
| `attn_comp` | attention 相关计算能耗 | 普通 attention 或 MLA attention/projection 时间戳的 MAC 能耗聚合 |
| `moe_dram` | MoE expert FFN 相关 DRAM 能耗 | 对每个设备的 `expertFFN` 时间戳聚合，见 `src/hardware/cluster.cpp:783-792`, `src/hardware/cluster.cpp:1027-1036` |
| `moe_comp` | MoE expert FFN 相关计算能耗 | 同上，取 `getCompEnergy()` |
| `OOM` | 是否发生 out of memory | `Cluster::checkMemorySize()` 设置 `out_of_memory`，`setTimeBreakDown()` 写入 `stat.isOOM` |

`fc_dram/fc_comp/attn_dram/attn_comp/moe_dram/moe_comp` 是按时间戳标签重新归类后的能耗，不是 CSV 前面几个总能耗字段的简单相邻列拆分。它们适合用来分析“能耗主要来自 FC、attention 还是 MoE”。

## 9. 每个 CSV 字段和源码的总对照表

| CSV 列 | 数据来源 | 赋值位置 | 写 CSV 位置 |
|---|---|---|---|
| `iter_info` | `Stat::iter_info` | `src/hardware/cluster.cpp:488-490`, `src/hardware/cluster.cpp:562-564`, `src/hardware/cluster.cpp:617-619` | `src/hardware/cluster.cpp:373` |
| `split` | `Stat::split` | `src/hardware/cluster.cpp:493-495` | `src/hardware/cluster.cpp:373` |
| `type` | `Stat::type` | `src/hardware/cluster.cpp:488-490`, `src/hardware/cluster.cpp:590-595`, `src/hardware/cluster.cpp:621-635` | `src/hardware/cluster.cpp:374` |
| `time` | `Stat::time` | `src/hardware/cluster.cpp:486-492`, `src/hardware/cluster.cpp:558-566`, `src/hardware/cluster.cpp:617-619` | `src/hardware/cluster.cpp:374` |
| `latency` | `Stat::latency` | `src/hardware/cluster.cpp:658-681`, `src/hardware/cluster.cpp:621-635` | `src/hardware/cluster.cpp:375` |
| `queueing_delay` | `Sequence::queueing_delay` | `src/scheduler/scheduler.cpp:357-380`, `src/hardware/cluster.cpp:621-635` | `src/hardware/cluster.cpp:376` |
| `arrival_time` | `Sequence::arrival_time` | `src/scheduler/scheduler.cpp:68-75`, `src/hardware/cluster.cpp:621-635` | `src/hardware/cluster.cpp:377` |
| `seq_queue_size` | `scheduler->sequence_queue.size()` | `src/hardware/cluster.cpp:497-508`, `src/hardware/cluster.cpp:568-579` | `src/hardware/cluster.cpp:378` |
| `input_len` | `Sequence::input_len` | `src/scheduler/sequence.cpp:6-14`, `src/hardware/cluster.cpp:621-635` | `src/hardware/cluster.cpp:379` |
| `output_len` | `Sequence::output_len` | `src/scheduler/sequence.cpp:6-14`, `src/hardware/cluster.cpp:621-635` | `src/hardware/cluster.cpp:380` |
| `num_sum_iter` | `Sequence::num_sum_iter` | `src/scheduler/sequence.cpp:35-50`, `src/hardware/cluster.cpp:621-635` | `src/hardware/cluster.cpp:381` |
| `mixed` | `Stat::is_mixed` | `src/hardware/cluster.cpp:658-681` | `src/hardware/cluster.cpp:382` |
| `batchsize` | `scheduler->getBatchSize()` | `src/hardware/cluster.cpp:658-665`, helpers in `src/scheduler/sequence.cpp:97-126` | `src/hardware/cluster.cpp:383` |
| `numtoken` | `scheduler->getNumProcessToken()` | `src/hardware/cluster.cpp:658-665`, `src/scheduler/scheduler.cpp:260-308` | `src/hardware/cluster.cpp:384` |
| `num_sum_seq` | `scheduler->getSumSize()` | `src/hardware/cluster.cpp:658-665`, `src/scheduler/sequence.cpp:59-68` | `src/hardware/cluster.cpp:385` |
| `num_gen_seq` | `scheduler->getGenSize()` | `src/hardware/cluster.cpp:658-665`, `src/scheduler/sequence.cpp:70-78` | `src/hardware/cluster.cpp:385` |
| `seqlen` | `scheduler->getAverageSeqlen()` | `src/hardware/cluster.cpp:658-665`, `src/scheduler/sequence.cpp:128-140` | `src/hardware/cluster.cpp:386` |
| `sum_attention_opb` | `TimeStamp::getOpb()` on `AttentionSum` | `src/hardware/cluster.cpp:828-836`, `src/hardware/cluster.cpp:1072-1080` | `src/hardware/cluster.cpp:387` |
| `qkvgen` | `attn_qkv_proj` duration | `src/hardware/cluster.cpp:701-749` | `src/hardware/cluster.cpp:388` |
| `q_down_proj` | `attn_q_down_proj` duration | `src/hardware/cluster.cpp:867-944` | `src/hardware/cluster.cpp:389` |
| `kv_down_proj` | `attn_kv_down_proj` duration | `src/hardware/cluster.cpp:867-951` | `src/hardware/cluster.cpp:390` |
| `kr_proj` | `attn_kr_proj` duration | `src/hardware/cluster.cpp:867-958` | `src/hardware/cluster.cpp:391` |
| `q_up_proj` | `attn_q_up_proj` duration | `src/hardware/cluster.cpp:871-965` | `src/hardware/cluster.cpp:392` |
| `qr_proj` | `attn_qr_proj` duration | `src/hardware/cluster.cpp:871-972` | `src/hardware/cluster.cpp:393` |
| `kv_up_proj` | `attn_kv_up_proj` duration | `src/hardware/cluster.cpp:873-979` | `src/hardware/cluster.cpp:394` |
| `tr_k_up_proj` | `attn_tr_k_up_proj` duration | `src/hardware/cluster.cpp:876-986` | `src/hardware/cluster.cpp:395` |
| `v_up_proj` | `attn_v_up_proj` duration | `src/hardware/cluster.cpp:876-993` | `src/hardware/cluster.cpp:396` |
| `atten_sum` | `AttentionSum` duration | `src/hardware/cluster.cpp:751-756`, `src/hardware/cluster.cpp:995-1000` | `src/hardware/cluster.cpp:397` |
| `atten_gen` | `AttentionGen` duration | `src/hardware/cluster.cpp:758-763`, `src/hardware/cluster.cpp:1002-1007` | `src/hardware/cluster.cpp:398` |
| `o_proj` | `attn_o_proj` duration | `src/hardware/cluster.cpp:765-770`, `src/hardware/cluster.cpp:1009-1014` | `src/hardware/cluster.cpp:399` |
| `ffn` | `feedforward` duration | `src/hardware/cluster.cpp:772-777`, `src/hardware/cluster.cpp:1016-1021` | `src/hardware/cluster.cpp:399` |
| `expert_ffn` | `expertFFN` duration minus MoE communication duration | `src/hardware/cluster.cpp:779-798`, `src/hardware/cluster.cpp:1023-1042` | `src/hardware/cluster.cpp:400` |
| `communication` | `all_reduce/moe_scatter/moe_gather` duration | `src/hardware/cluster.cpp:800-803`, `src/hardware/cluster.cpp:1044-1047` | `src/hardware/cluster.cpp:401` |
| `rope` | `k_rope/q_rope` duration | `src/hardware/cluster.cpp:805-808`, `src/hardware/cluster.cpp:1049-1052` | `src/hardware/cluster.cpp:402` |
| `layernorm` | layernorm module duration | `src/hardware/cluster.cpp:810-813`, `src/hardware/cluster.cpp:1054-1057` | `src/hardware/cluster.cpp:403` |
| `residual` | residual module duration | `src/hardware/cluster.cpp:815-818`, `src/hardware/cluster.cpp:1059-1062` | `src/hardware/cluster.cpp:404` |
| `act_energy` | total device ACT energy | `src/hardware/cluster.cpp:497-506`, `src/module/module_graph.cpp:258-260` | `src/hardware/cluster.cpp:405` |
| `read_energy` | total device READ energy | `src/hardware/cluster.cpp:497-506`, `src/module/module_graph.cpp:260-262` | `src/hardware/cluster.cpp:406` |
| `write_energy` | total device WRITE energy | `src/hardware/cluster.cpp:497-506`, `src/module/module_graph.cpp:262-263` | `src/hardware/cluster.cpp:407` |
| `all_act_energy` | total all-bank ACT energy | `src/hardware/cluster.cpp:497-506`, `src/module/module_graph.cpp:265-266` | `src/hardware/cluster.cpp:408` |
| `all_read_energy` | total all-bank READ energy | `src/hardware/cluster.cpp:497-506`, `src/module/module_graph.cpp:267-268` | `src/hardware/cluster.cpp:409` |
| `all_write_energy` | total all-bank WRITE energy | `src/hardware/cluster.cpp:497-506`, `src/module/module_graph.cpp:269-270` | `src/hardware/cluster.cpp:410` |
| `mac_energy` | total MAC energy | `src/hardware/cluster.cpp:497-506`, `src/module/module_graph.cpp:272-274` | `src/hardware/cluster.cpp:411` |
| `total_energy` | total DRAM + MAC energy | `src/hardware/cluster.cpp:497-506`, `src/module/module_graph.cpp:327-332` | `src/hardware/cluster.cpp:412` |
| `fc_dram` | FC category DRAM energy | `src/hardware/cluster.cpp:737-826`, `src/hardware/cluster.cpp:932-1070` | `src/hardware/cluster.cpp:413` |
| `fc_comp` | FC category compute energy | `src/hardware/cluster.cpp:737-826`, `src/hardware/cluster.cpp:932-1070` | `src/hardware/cluster.cpp:414` |
| `attn_dram` | attention category DRAM energy | `src/hardware/cluster.cpp:737-826`, `src/hardware/cluster.cpp:932-1070` | `src/hardware/cluster.cpp:415` |
| `attn_comp` | attention category compute energy | `src/hardware/cluster.cpp:737-826`, `src/hardware/cluster.cpp:932-1070` | `src/hardware/cluster.cpp:416` |
| `moe_dram` | MoE category DRAM energy | `src/hardware/cluster.cpp:783-792`, `src/hardware/cluster.cpp:1027-1036` | `src/hardware/cluster.cpp:417` |
| `moe_comp` | MoE category compute energy | `src/hardware/cluster.cpp:783-792`, `src/hardware/cluster.cpp:1027-1036` | `src/hardware/cluster.cpp:418` |
| `OOM` | `Cluster::out_of_memory` | `src/hardware/cluster.cpp:75-298`, `src/hardware/cluster.cpp:820-826`, `src/hardware/cluster.cpp:1064-1070` | `src/hardware/cluster.cpp:419` |

## 10. 如何读当前这份 DeepSeekV3 decode CSV

当前实验配置大致是：

```yaml
model_name: deepseekV3
gpu_gen: B100
num_node: 1
num_device: 8
processor_type: GPU
input_len: 1024
output_len: 10
max_batch_size: 32
decode_mode: on
use_ramulator: off
compressed_kv: on
use_absorb: on
use_flash_mla: on
```

读表时建议注意：

1. 这是 decode 阶段，不是 prefill 阶段。`num_sum_seq` 通常为 0，`num_gen_seq` 会接近 batch size。
2. `deepseekV3` 使用 MLA，所以重点看 `q_down_proj/kv_down_proj/kr_proj/q_up_proj/qr_proj/tr_k_up_proj/v_up_proj/atten_gen/o_proj`，不要重点看 `qkvgen`。
3. `use_absorb: on` 时，`tr_k_up_proj` 和 `v_up_proj` 是关键列，`kv_up_proj` 通常为 0 或不重要。
4. `expert_ffn` 很大时，说明 MoE expert FFN 是主要耗时来源；但通信开销要单独看 `communication`。
5. `OOM=0` 表示该配置没有触发内存超限；如果为 `1`，需要回看 `Cluster::checkMemorySize()` 的估算逻辑。
6. `time` 是累计时间，`latency` 对 `t2t` 行更接近当前 step 的执行耗时。对比不同配置时，常用 `latency` 看单步 decode，常用最后一行 `time` 看累计成本。

## 11. 做实验时建议重点比较的列

| 研究目标 | 重点列 |
|---|---|
| 单步 decode 延迟 | `type`, `latency`, `seqlen`, `batchsize`, `num_gen_seq` |
| 总体仿真耗时 | `time`, 最后一批 `e2e` 行 |
| attention 瓶颈 | `atten_gen`, `atten_sum`, `q_down_proj`, `kv_down_proj`, `q_up_proj`, `tr_k_up_proj`, `v_up_proj`, `attn_dram`, `attn_comp` |
| MoE 瓶颈 | `expert_ffn`, `communication`, `moe_dram`, `moe_comp` |
| 通信瓶颈 | `communication`，并结合 `num_node`, `num_device`, `TP/DP` 配置看 |
| 能耗分析 | `total_energy`, `mac_energy`, `fc_dram`, `attn_dram`, `moe_dram` |
| 是否爆内存 | `OOM`，以及运行日志中的 ACT/Weight/Cache/Total |

