# LLMSimulator 建模与运行机制说明

这份文档回答三个问题：

1. 这个仿真器如何对大语言模型做结构建模。
2. 它如何对 GPU 以及相关内存/互连系统做硬件建模。
3. 它如何把“一个 LLM 请求批次”映射成 GPU 上的仿真执行过程，并最终得到时延、能耗和 CSV 结果。

本文基于当前仓库源码整理，重点引用以下目录：

- `eval/`：程序入口、配置解析、实验输出命名
- `src/model/`：LLM 顶层结构与模型参数模板
- `src/module/`：Transformer、Attention、MoE、通信等模块化表示
- `src/scheduler/`：请求、batch、prefill/decode 状态推进
- `src/hardware/`：设备、集群、执行器、算子硬件模型
- `src/dram/`：DRAM 接口、地址映射、Ramulator 集成

## 1. 总体设计思路

这个工程不是直接运行一个真实的神经网络框架，也不是调用 CUDA kernel 去算真实张量。它的核心方法是：

1. 先把一个 LLM 拆成模块图。
2. 每个模块只保留系统仿真关心的信息：
   - 输入输出张量形状
   - 并行切分方式
   - 应该映射到哪类硬件单元
   - 理论 FLOPs
   - 理论访存量
   - 通信/同步关系
3. 每个模块执行时不做数值计算，而是调用硬件执行模型返回一个 `ExecStatus`。
4. `ExecStatus` 再被 `TopModuleGraph` 累积成设备级时间线、能耗和利用率。
5. `Cluster` 把所有设备结果汇总，形成每次迭代的统计项并输出到 CSV。

可以把它理解成：

- `model/ + module/` 负责“这个模型由哪些算子构成”
- `scheduler/` 负责“这轮有哪些 token 在跑、属于 prefill 还是 decode”
- `hardware/` 负责“这些算子在 GPU/Logic/PIM 上要花多少时间和能耗”

## 2. 程序入口与主流程

一次仿真的主入口在 [eval/test.cpp](/home/zsy/LLMSimulator/eval/test.cpp:14)。

主流程可以按下面理解：

1. 读取 YAML 配置。
   - [eval/test.cpp](/home/zsy/LLMSimulator/eval/test.cpp:14)
2. 解析系统配置 `SystemConfig`。
   - GPU 代际、节点数、卡数、互连带宽、是否使用 Ramulator、prefill/decode 模式等。
   - [eval/test.cpp](/home/zsy/LLMSimulator/eval/test.cpp:41)
3. 解析模型配置 `ModelConfig`。
   - 选择内置模型模板，再覆写并行度、precision、MLA/压缩 KV 等选项。
   - [eval/test.cpp](/home/zsy/LLMSimulator/eval/test.cpp:161)
4. 创建 `Scheduler`。
   - [eval/test.cpp](/home/zsy/LLMSimulator/eval/test.cpp:224)
5. 创建 `Cluster`。
   - [eval/test.cpp](/home/zsy/LLMSimulator/eval/test.cpp:228)
6. 创建 `Model`，把整个 LLM 结构实例化到每个 device 上。
   - [eval/test.cpp](/home/zsy/LLMSimulator/eval/test.cpp:230)
7. 做显存/容量检查。
   - [eval/test.cpp](/home/zsy/LLMSimulator/eval/test.cpp:232)
8. 建立模块依赖关系。
   - [eval/test.cpp](/home/zsy/LLMSimulator/eval/test.cpp:233)
9. 调用 `Cluster::runIteration()` 进入真正的仿真循环。
   - 文件名构造从 [eval/test.cpp](/home/zsy/LLMSimulator/eval/test.cpp:247) 开始
   - 实际运行调用在 [eval/test.cpp](/home/zsy/LLMSimulator/eval/test.cpp:342)

这条链路说明：这个工程是“先静态搭图，再动态推进 batch 状态”。

## 3. 大语言模型是如何建模的

### 3.1 `ModelConfig`：先把模型压缩成一组系统参数

LLM 的结构参数集中在 [src/model/model_config.h](/home/zsy/LLMSimulator/src/model/model_config.h:9)。

这里没有保存真实权重，而是保存系统仿真所需的结构参数：

- `hidden_dim`：隐藏维度
- `head_dim`：单头维度
- `num_layers`：层数
- `num_heads` / `num_kv_heads`：Q 头数与 KV 头数
- `intermediate_dim`：普通 FFN 中间维度
- `expert_intermediate_dim`：MoE expert FFN 中间维度
- `num_routed_expert` / `num_shared_expert`：MoE 专家数
- `expert_freq`：多少层出现一次 MoE
- `top_k`：每个 token 路由到多少个 expert
- `ffn_way`：FFN 结构是 2-way 还是 3-way
- `q_lora_rank` / `kv_lora_rank` / `qk_nope_head_dim` / `qk_rope_head_dim`：MLA 建模参数
- `compressed_kv` / `use_absorb`：DeepSeek V3 风格 MLA 路径开关
- `precision_byte`：权重/激活字节数
- `input_len` / `output_len`：这次实验的输入输出长度

源码位置：

- 参数定义：[src/model/model_config.h](/home/zsy/LLMSimulator/src/model/model_config.h:12)
- 成员字段：[src/model/model_config.h](/home/zsy/LLMSimulator/src/model/model_config.h:56)
- 内置模型模板：
  - `mixtral`：[src/model/model_config.h](/home/zsy/LLMSimulator/src/model/model_config.h:91)
  - `deepseekV3`：[src/model/model_config.h](/home/zsy/LLMSimulator/src/model/model_config.h:107)
  - `llama3_405B`：[src/model/model_config.h](/home/zsy/LLMSimulator/src/model/model_config.h:111)

这一步本质上是把“一个模型架构”压缩成“若干张量维度、专家配置、注意力配置和并行配置”。

### 3.2 `Model`：把 LLM 实例化到每一张卡

`Model` 的作用不是做计算，而是为 cluster 中每个 device 构造一份顶层模块图。

源码位置：[src/model/model.h](/home/zsy/LLMSimulator/src/model/model.h:14)

关键逻辑：

- 遍历 `cluster->num_total_device`
- 给每张设备设置 `model_config`
- 在该设备上创建一个 `LLM` 顶层模块
- 调用 `model_distribute()` 触发一次“假 forward”，借此把模块图、张量和依赖关系全部注册到 `TopModuleGraph`

对应代码：

- per-device 构造：[src/model/model.h](/home/zsy/LLMSimulator/src/model/model.h:16)
- 触发图构建的 `model_distribute()`：[src/model/model.h](/home/zsy/LLMSimulator/src/model/model.h:37)

这一步非常关键：它并不是为了求输出，而是为了把未来要跑的执行图预先铺好。

### 3.3 `LLM`：顶层模型结构

`LLM` 的构造在 [src/model/llm.cpp](/home/zsy/LLMSimulator/src/model/llm.cpp:10)。

顶层结构是：

1. `Embedding_layer`
2. 多层 `Decoder` 或 `MoE_decoder`
3. `lm_head`

源码位置：

- embedding 创建：[src/model/llm.cpp](/home/zsy/LLMSimulator/src/model/llm.cpp:18)
- 普通模型逐层组网：[src/model/llm.cpp](/home/zsy/LLMSimulator/src/model/llm.cpp:22)
- DeepSeekV3 特化路径：[src/model/llm.cpp](/home/zsy/LLMSimulator/src/model/llm.cpp:42)
- `lm_head`：[src/model/llm.cpp](/home/zsy/LLMSimulator/src/model/llm.cpp:59)
- forward 顺序：[src/model/llm.cpp](/home/zsy/LLMSimulator/src/model/llm.cpp:64)

这里可以看到这个工程如何表达不同模型：

- 普通 dense 层用 `Decoder`
- MoE 层用 `MoEDecoder`
- DeepSeekV3 用 `first_k_dense` 决定前几层 dense、后几层 MoE

### 3.4 `Decoder`：一层 Transformer block 的建模

普通 Decoder 在 [src/module/decoder.cpp](/home/zsy/LLMSimulator/src/module/decoder.cpp:12)。

它拆成如下模块：

1. `input_layer_norm`
2. `attention`
3. `residual_1`
4. `post_attn_layer_norm`
5. `feedforward`
6. `all_reduce`
7. `residual_2`

源码：

- 模块创建：[src/module/decoder.cpp](/home/zsy/LLMSimulator/src/module/decoder.cpp:26)
- 执行顺序：[src/module/decoder.cpp](/home/zsy/LLMSimulator/src/module/decoder.cpp:70)

这里的重点不是数值，而是把一层 Transformer 拆成可以单独计时的系统模块。

### 3.5 `MoEDecoder`：MoE 层的建模

MoE 层在 [src/module/decoder.cpp](/home/zsy/LLMSimulator/src/module/decoder.cpp:92)。

它和普通层的最大区别在于：

- attention 之后不是普通 FFN
- 而是 `ExpertFFN`

对应代码：

- `ExpertFFN` 挂接：[src/module/decoder.cpp](/home/zsy/LLMSimulator/src/module/decoder.cpp:132)
- forward 执行 expert 路径：[src/module/decoder.cpp](/home/zsy/LLMSimulator/src/module/decoder.cpp:143)

### 3.6 Attention 的建模

#### 标准 Attention

标准 attention 在 [src/module/layer.cpp](/home/zsy/LLMSimulator/src/module/layer.cpp:11)。

它被拆成：

1. `attn_qkv_proj`
2. `self_attention`
3. `attn_o_proj`
4. `all_reduce`

代码位置：

- 模块搭建：[src/module/layer.cpp](/home/zsy/LLMSimulator/src/module/layer.cpp:11)
- forward 顺序：[src/module/layer.cpp](/home/zsy/LLMSimulator/src/module/layer.cpp:40)

`self_attention` 又会进一步拆为四段，见 [src/module/parallel.cpp](/home/zsy/LLMSimulator/src/module/parallel.cpp:127)：

1. `AttentionSplit`
2. `AttentionSum`
3. `AttentionGen`
4. `AttentionMerge`

执行顺序见 [src/module/parallel.cpp](/home/zsy/LLMSimulator/src/module/parallel.cpp:177)。

这里的设计非常重要：它把注意力分成 prefill/sum 阶段和 decode/gen 阶段。

- `AttentionSum` 主要对应已经可并行处理的一段上下文累积
- `AttentionGen` 对应逐 token decode 时读取 KV cache 的部分

#### MLA / DeepSeekV3 Attention

DeepSeekV3 的 MLA 路径在 [src/module/layer.cpp](/home/zsy/LLMSimulator/src/module/layer.cpp:56)。

这个模块不是一个简单的 QKV 投影，而是显式建模以下子步骤：

- `attn_q_down_proj`
- `attn_kv_down_proj`
- `attn_kr_proj`
- `k_rope`
- `latent_q_layer_norm`
- `latent_kv_layer_norm`

随后根据 `use_absorb` 分成两条路径：

1. absorb MLA
   - `attn_q_up_proj`
   - `attn_qr_proj`
   - `q_rope`
   - `attn_tr_k_up_proj`
   - `attn_mla_absorbed`
   - `attn_v_up_proj`
   - `attn_o_proj`
   - `all_reduce`
   - 对应 [src/module/layer.cpp](/home/zsy/LLMSimulator/src/module/layer.cpp:95)
2. baseline MLA
   - `c_kv_restore`
   - `attn_q_up_proj`
   - `attn_qr_proj`
   - `attn_kv_up_proj`
   - `multi_latent_attention`
   - `attn_o_proj`
   - `all_reduce`
   - 对应 [src/module/layer.cpp](/home/zsy/LLMSimulator/src/module/layer.cpp:138)

forward 执行顺序在：

- absorb 路径：[src/module/layer.cpp](/home/zsy/LLMSimulator/src/module/layer.cpp:199)
- baseline MLA 路径：[src/module/layer.cpp](/home/zsy/LLMSimulator/src/module/layer.cpp:239)

这也是为什么你的 CSV 会有 `q_down_proj`、`kv_down_proj`、`tr_k_up_proj`、`v_up_proj` 这些专门的列。

### 3.7 FFN 和 MoE 的建模

普通 FFN 在 [src/module/layer.cpp](/home/zsy/LLMSimulator/src/module/layer.cpp:292)。

其核心结构是：

1. `ffn_up_proj`
2. `activation`
3. `ffn_down_proj`
4. 可选 `all_reduce`

MoE 的核心在 [src/module/expert.cpp](/home/zsy/LLMSimulator/src/module/expert.cpp:13)。

它没有把 expert 当作一个黑盒，而是显式建模：

1. `gate_fn`
2. `gate_update`
3. `moe_scatter`
4. `moe_route`
5. 多个 `expert_FFN_x`
6. `moe_all_reduce_for_e_tp`
7. `moe_gather`
8. `moe_all_reduce_for_gather`
9. 多个 `sync`
10. 可选 shared experts

代码位置：

- ExpertFFN 结构搭建：[src/module/expert.cpp](/home/zsy/LLMSimulator/src/module/expert.cpp:13)
- expert 执行顺序：[src/module/expert.cpp](/home/zsy/LLMSimulator/src/module/expert.cpp:136)

这说明该仿真器对 MoE 的建模非常系统化，不仅考虑 FFN 计算，还考虑：

- token 路由
- scatter / gather
- expert tensor parallel all-reduce
- shared expert
- 多次同步点

## 4. GPU 模块是如何建模的

### 4.1 `SystemConfig`：GPU 平台参数

硬件总参数在 [src/hardware/hardware_config.h](/home/zsy/LLMSimulator/src/hardware/hardware_config.h:16)。

GPU 建模核心参数包括：

- `compute_peak_flops`
- `memory_bandwidth`
- `memory_capacity`
- `device_ict_bandwidth` / `device_ict_latency`
- `node_ict_bandwidth` / `node_ict_latency`
- `num_node`
- `num_device`
- `processor_type`
- `logic_x` / `logic_op_b`
- `pim_x` / `pim_op_b`
- `use_ramulator`
- `parallel_execution`

几种 GPU 代际模板：

- A100：[src/hardware/hardware_config.h](/home/zsy/LLMSimulator/src/hardware/hardware_config.h:163)
- H100：[src/hardware/hardware_config.h](/home/zsy/LLMSimulator/src/hardware/hardware_config.h:201)
- B100：[src/hardware/hardware_config.h](/home/zsy/LLMSimulator/src/hardware/hardware_config.h:239)
- B200：[src/hardware/hardware_config.h](/home/zsy/LLMSimulator/src/hardware/hardware_config.h:277)

从这里能看出，该工程对 GPU 的抽象层次是：

- 算力峰值
- 显存带宽
- 显存容量
- 卡间互连
- 节点间互连

也就是说，它更偏“系统性能模型”，不是对 SM、warp、CTA 的微结构仿真。

### 4.2 `Device`：单张 GPU 的状态与内存系统

每个 device 的实现位于 [src/hardware/device.cpp](/home/zsy/LLMSimulator/src/hardware/device.cpp:21)。

一个 `Device` 内部维护：

- 该卡的 `compute_peak_flops`
- `memory_bandwidth`
- `memory_capacity`
- `TopModuleGraph`
- `DRAMInterface`
- `MMapController`
- `StatusBoard`

关键逻辑：

- 根据 `gpu_gen` 选择 DRAM 配置文件
  - H100 用 `dram_config_HBM3_80GB.yaml`
  - B100/B200 用 `dram_config_HBM3E_192GB.yaml`
  - [src/hardware/device.cpp](/home/zsy/LLMSimulator/src/hardware/device.cpp:33)
- 初始化 `DRAMInterface`
  - [src/hardware/device.cpp](/home/zsy/LLMSimulator/src/hardware/device.cpp:59)
- 初始化地址映射 `MMapController`
  - [src/hardware/device.cpp](/home/zsy/LLMSimulator/src/hardware/device.cpp:60)
- `Device::execution()` 中调用 `cluster->executor.execution()`
  - [src/hardware/device.cpp](/home/zsy/LLMSimulator/src/hardware/device.cpp:151)

这说明单卡并不是一个简单的“时间计数器”，而是挂着 DRAM 与模块图状态的仿真实体。

### 4.3 `Executor`：把模块类型映射到硬件执行模型

执行分发中心是 [src/hardware/executor.cpp](/home/zsy/LLMSimulator/src/hardware/executor.cpp:13)。

它做两层映射：

1. 按 layer type 选择哪类执行函数
2. 按 processor type 选择 GPU / LOGIC / PIM 版本

初始化表：

- linear：[src/hardware/executor.cpp](/home/zsy/LLMSimulator/src/hardware/executor.cpp:28)
- activation：[src/hardware/executor.cpp](/home/zsy/LLMSimulator/src/hardware/executor.cpp:46)
- attention sum/gen/mixed：[src/hardware/executor.cpp](/home/zsy/LLMSimulator/src/hardware/executor.cpp:56)
- MLA/Absorb MLA：[src/hardware/executor.cpp](/home/zsy/LLMSimulator/src/hardware/executor.cpp:85)

实际执行入口：

- `Executor::execution()`：[src/hardware/executor.cpp](/home/zsy/LLMSimulator/src/hardware/executor.cpp:135)
- `Executor::executePType()`：[src/hardware/executor.cpp](/home/zsy/LLMSimulator/src/hardware/executor.cpp:157)

这里有一个关键策略：

- 如果一个 layer 可以映射到多种 processor type，执行器会遍历所有候选类型
- 选择 `total_duration` 更小的那个 `ExecStatus`

见 [src/hardware/executor.cpp](/home/zsy/LLMSimulator/src/hardware/executor.cpp:145)。

这意味着它不是固定把所有算子都丢给 GPU，而是允许在 GPU / LOGIC / PIM 之间比较最优执行时间。

### 4.4 算子硬件模型：如何估计时间和能耗

#### Linear

GPU 线性层模型在 [src/hardware/linear_impl.cpp](/home/zsy/LLMSimulator/src/hardware/linear_impl.cpp:16)。

核心公式：

- FLOPs：`2 * m * k * n`
- Memory size：`(m*k + k*n + m*n) * precision_byte`
- Compute duration：`FLOPs / compute_peak_flops`
- Memory duration：`memory_size / memory_bandwidth`
- 总时间：`max(compute_duration, memory_duration)`

对应代码：

- 形状与 FLOPs：[src/hardware/linear_impl.cpp](/home/zsy/LLMSimulator/src/hardware/linear_impl.cpp:23)
- 计算时间与访存时间：[src/hardware/linear_impl.cpp](/home/zsy/LLMSimulator/src/hardware/linear_impl.cpp:30)
- ideal memory 路径：[src/hardware/linear_impl.cpp](/home/zsy/LLMSimulator/src/hardware/linear_impl.cpp:52)
- total duration：[src/hardware/linear_impl.cpp](/home/zsy/LLMSimulator/src/hardware/linear_impl.cpp:61)

这就是该工程最基本的 roofline 风格建模。

#### AttentionSum

GPU prefill attention 模型入口在 [src/hardware/attention_sum_impl.cpp](/home/zsy/LLMSimulator/src/hardware/attention_sum_impl.cpp:16)。

关键点：

- 遍历 `sequences_metadata->get_sum()`，即仍处于 sum/prefill 阶段的序列
- 对每条序列分别估算 scoring 阶段的 FLOPs 和 memory size
- 按 query 读、KV 读、score 写的方式估算 DRAM 访问

对应代码：

- sum 序列选择：[src/hardware/attention_sum_impl.cpp](/home/zsy/LLMSimulator/src/hardware/attention_sum_impl.cpp:48)
- scoring 维度定义：`m/k/n`
  - [src/hardware/attention_sum_impl.cpp](/home/zsy/LLMSimulator/src/hardware/attention_sum_impl.cpp:59)
- FLOPs / memory：[src/hardware/attention_sum_impl.cpp](/home/zsy/LLMSimulator/src/hardware/attention_sum_impl.cpp:63)
- Ramulator 路径：[src/hardware/attention_sum_impl.cpp](/home/zsy/LLMSimulator/src/hardware/attention_sum_impl.cpp:74)
- ideal memory 路径：[src/hardware/attention_sum_impl.cpp](/home/zsy/LLMSimulator/src/hardware/attention_sum_impl.cpp:112)

#### AttentionGen

GPU decode attention 模型入口在 [src/hardware/attention_gen_impl.cpp](/home/zsy/LLMSimulator/src/hardware/attention_gen_impl.cpp:16)。

关键点：

- 只处理 `get_gen()` 的序列
- 将 decode attention 分成三段：
  - scoring
  - softmax
  - context
- 显式读取 `k_cache` 和 `v_cache`

对应代码：

- gen 序列选择：[src/hardware/attention_gen_impl.cpp](/home/zsy/LLMSimulator/src/hardware/attention_gen_impl.cpp:50)
- scoring 阶段：[src/hardware/attention_gen_impl.cpp](/home/zsy/LLMSimulator/src/hardware/attention_gen_impl.cpp:60)
- K cache 读取：[src/hardware/attention_gen_impl.cpp](/home/zsy/LLMSimulator/src/hardware/attention_gen_impl.cpp:84)
- softmax 阶段：[src/hardware/attention_gen_impl.cpp](/home/zsy/LLMSimulator/src/hardware/attention_gen_impl.cpp:103)
- context 阶段：[src/hardware/attention_gen_impl.cpp](/home/zsy/LLMSimulator/src/hardware/attention_gen_impl.cpp:121)
- V cache 读取：[src/hardware/attention_gen_impl.cpp](/home/zsy/LLMSimulator/src/hardware/attention_gen_impl.cpp:149)

#### MLA

MLA 的 GPU 模型入口：

- baseline MLA sum：[src/hardware/attention_sum_impl.cpp](/home/zsy/LLMSimulator/src/hardware/attention_sum_impl.cpp:657)
- baseline MLA gen：[src/hardware/attention_gen_impl.cpp](/home/zsy/LLMSimulator/src/hardware/attention_gen_impl.cpp:489)
- absorb MLA sum：[src/hardware/attention_sum_impl.cpp](/home/zsy/LLMSimulator/src/hardware/attention_sum_impl.cpp:2051)
- absorb MLA gen：[src/hardware/attention_gen_impl.cpp](/home/zsy/LLMSimulator/src/hardware/attention_gen_impl.cpp:1469)

这些实现比标准 attention 更复杂，因为它们会：

- 处理 latent KV
- 处理 rope 分支
- 处理 restore 或 absorbed path
- 在 decode 场景中对 cache 的形态进行特别建模

### 4.5 是否用 Ramulator

很多算子模型都有两条路径：

1. `use_ramulator = false`
   - 用理想化带宽模型
   - 直接按 `memory_size / memory_bandwidth` 估时
2. `use_ramulator = true`
   - 用 `issueRamulator()` 发出细粒度 DRAM 请求
   - 由 DRAM 子系统返回更细的访存时延和计数

例如：

- linear：[src/hardware/linear_impl.cpp](/home/zsy/LLMSimulator/src/hardware/linear_impl.cpp:42)
- attention sum：[src/hardware/attention_sum_impl.cpp](/home/zsy/LLMSimulator/src/hardware/attention_sum_impl.cpp:74)
- attention gen：[src/hardware/attention_gen_impl.cpp](/home/zsy/LLMSimulator/src/hardware/attention_gen_impl.cpp:84)

所以这个工程的访存建模既可以是高层 bandwidth model，也可以接入更细的 DRAM 调度模型。

## 5. 仿真运行是怎样推进的

### 5.1 `Scheduler`：生成和推进请求状态

调度器入口在 [src/scheduler/scheduler.cpp](/home/zsy/LLMSimulator/src/scheduler/scheduler.cpp:4)。

它维护两类东西：

- `sequence_queue`：还没进入 batch 的请求
- `running_queue`：已经在每个 DP rank 上运行的 batch

#### 请求初始化

合成 workload 的请求生成在：

- `pushDummySeq()`：[src/scheduler/scheduler.cpp](/home/zsy/LLMSimulator/src/scheduler/scheduler.cpp:42)

这里会根据配置调整请求状态：

- `prefill_mode` 下，把 `output_len` 改成 `input_len`
  - [src/scheduler/scheduler.cpp](/home/zsy/LLMSimulator/src/scheduler/scheduler.cpp:84)
- `decode_mode` 下，把 `current_len` 直接设为 `input_len`
  - [src/scheduler/scheduler.cpp](/home/zsy/LLMSimulator/src/scheduler/scheduler.cpp:88)

这正是 decode 仿真为什么一开始 `seqlen` 就是 `input_len`。

#### 每轮该处理多少 token

每轮的 batch 元数据生成在 `setMetadata()`：

- [src/scheduler/scheduler.cpp](/home/zsy/LLMSimulator/src/scheduler/scheduler.cpp:260)

核心规则：

- 对 gen 序列，通常一次处理 `1` 个 token
  - [src/scheduler/scheduler.cpp](/home/zsy/LLMSimulator/src/scheduler/scheduler.cpp:280)
- 对 sum 序列，按 `max_process_token` 在多个 sum 序列之间均分
  - [src/scheduler/scheduler.cpp](/home/zsy/LLMSimulator/src/scheduler/scheduler.cpp:294)

#### 每轮之后更新序列状态

- 普通 mixed 更新：
  - [src/scheduler/scheduler.cpp](/home/zsy/LLMSimulator/src/scheduler/scheduler.cpp:311)
- sum/gen split 更新：
  - [src/scheduler/scheduler.cpp](/home/zsy/LLMSimulator/src/scheduler/scheduler.cpp:334)

#### 把等待队列填进 running queue

- [src/scheduler/scheduler.cpp](/home/zsy/LLMSimulator/src/scheduler/scheduler.cpp:357)

这里会更新 queueing delay，并按 DP batch 轮转装入。

### 5.2 `Sequence`：每个请求跟踪什么状态

单个请求的状态在 [src/scheduler/sequence.cpp](/home/zsy/LLMSimulator/src/scheduler/sequence.cpp:6)。

关键字段和含义：

- `input_len`
- `output_len`
- `current_len`
- `total_len`
- `num_process_token`
- `sum_stage`
- `arrival_time`
- `first_token_time`
- `end_token_time`
- `queueing_delay`
- `num_sum_iter`

状态更新规则在 `Sequence::update()`：

- [src/scheduler/sequence.cpp](/home/zsy/LLMSimulator/src/scheduler/sequence.cpp:35)

逻辑是：

- 每轮增加 `end_token_time`
- 若仍在 sum 阶段，则也增加 `first_token_time`
- `current_len += num_process_token`
- 一旦 `current_len == input_len`，就从 sum 阶段切到 gen 阶段

因此：

- `t2ft` 本质上就是走完整个 sum 阶段的累计时间
- `e2e` 则是 sum + gen 的总时间

### 5.3 `TopModuleGraph`：模块依赖图如何执行

`TopModuleGraph` 是静态模块图的运行器，源码在 [src/module/module_graph.cpp](/home/zsy/LLMSimulator/src/module/module_graph.cpp:115)。

#### 它保存什么

- 一个有序的 `module_graph`
- 每个 module 的 push/pop 时间戳
- 当前设备状态 `StatusBoard`

#### 如何推进

- `run()` 从当前模块指针开始往后扫
  - [src/module/module_graph.cpp](/home/zsy/LLMSimulator/src/module/module_graph.cpp:149)
- 每个 module 执行前调用 `set_stamp()`
  - [src/module/module_graph.cpp](/home/zsy/LLMSimulator/src/module/module_graph.cpp:177)
- 如果模块 ready，则执行 `module->forward(...)`
  - [src/module/module_graph.cpp](/home/zsy/LLMSimulator/src/module/module_graph.cpp:27)

#### 同步如何处理

对于 `sync` 模块：

- 先检查依赖张量是否全部 ready
- 再把所有相关 device 的时间推进到最大值

对应代码：

- ready 检查：[src/module/module_graph.cpp](/home/zsy/LLMSimulator/src/module/module_graph.cpp:54)
- `sync_devices()`：[src/module/module_graph.cpp](/home/zsy/LLMSimulator/src/module/module_graph.cpp:74)

#### 时间和能耗如何累计

在 module pop 的时候，会从 `device->getExecStatus()` 取回这次执行的硬件统计，并更新：

- `device_time`
- `high_time` / `low_time`
- `act_energy` / `read_energy` / `write_energy`
- `all_*`
- `mac_energy`

对应代码：

- `set_pop_status()`：[src/module/module_graph.cpp](/home/zsy/LLMSimulator/src/module/module_graph.cpp:223)

尤其要注意：

- `high_time` 表示 GPU 时间
- `low_time` 表示 Logic/PIM 时间

定义在 [src/module/status.h](/home/zsy/LLMSimulator/src/module/status.h:61)。

这就是它支持异构并行执行的基础。

### 5.4 `Cluster`：一次迭代如何在所有 GPU 上跑起来

集群主循环在 [src/hardware/cluster.cpp](/home/zsy/LLMSimulator/src/hardware/cluster.cpp:1084)。

`Cluster::run()` 的流程非常直接：

1. `setPerformExecution(true)`
2. `restartModuleGraph()`
3. 只要任一节点还有未完成模块，就循环：
   - 遍历每个 `Node`
   - 每个 `Node` 再遍历它的每个 `Device`
   - 每个 `Device` 执行自己的 `TopModuleGraph`

对应代码：

- `Cluster::run()`：[src/hardware/cluster.cpp](/home/zsy/LLMSimulator/src/hardware/cluster.cpp:1084)
- `Node::run()`：[src/hardware/node.cpp](/home/zsy/LLMSimulator/src/hardware/node.cpp:33)
- `Device::run()`：[src/hardware/device.cpp](/home/zsy/LLMSimulator/src/hardware/device.cpp:71)

因此，一轮仿真不是“直接算完一整个 batch”，而是不断推进各 device 的模块图直到都没有剩余模块。

### 5.5 `Cluster::runIteration()`：多轮 token 级仿真

真正的实验循环在：

- `runIteration()`：[src/hardware/cluster.cpp](/home/zsy/LLMSimulator/src/hardware/cluster.cpp:424)
- mixed 系统：[src/hardware/cluster.cpp](/home/zsy/LLMSimulator/src/hardware/cluster.cpp:459)
- sum/gen split 系统：[src/hardware/cluster.cpp](/home/zsy/LLMSimulator/src/hardware/cluster.cpp:527)

每轮大致流程：

1. `scheduler->setMetadata()`
2. `run(metadata)`，跑完这一轮所有模块
3. 从 `get_device(0)->status.device_time` 取这轮延迟
4. 汇总能耗 `getTotalEnergy()`
5. 填 `Stat`
6. `scheduler->updateScheduler(time)` 更新请求状态
7. 从等待队列补新请求

代码：

- metadata 生成：[src/hardware/cluster.cpp](/home/zsy/LLMSimulator/src/hardware/cluster.cpp:476)
- 实际执行：[src/hardware/cluster.cpp](/home/zsy/LLMSimulator/src/hardware/cluster.cpp:477)
- 取一轮时间：[src/hardware/cluster.cpp](/home/zsy/LLMSimulator/src/hardware/cluster.cpp:478)
- 累计能耗：[src/hardware/cluster.cpp](/home/zsy/LLMSimulator/src/hardware/cluster.cpp:497)
- 统计填充：[src/hardware/cluster.cpp](/home/zsy/LLMSimulator/src/hardware/cluster.cpp:509)

### 5.6 显存/OOM 检查是怎样做的

在正式跑之前，`Cluster::checkMemorySize()` 会做容量估算：

- [src/hardware/cluster.cpp](/home/zsy/LLMSimulator/src/hardware/cluster.cpp:75)

它不是逐地址精确分配，而是用解析式估计：

- activation size
- weight size
- KV cache size

而且对以下情况给出不同公式：

- 标准 attention
- compressed KV
- absorb MLA
- decode 模式
- prefill/mixed 模式

如果超容量：

- 可直接退出
- 或自动缩小 `max_batch_size`

对应逻辑在 [src/hardware/cluster.cpp](/home/zsy/LLMSimulator/src/hardware/cluster.cpp:241)。

## 6. 从配置到 GPU 仿真的完整链路

把整个系统串起来，可以概括为下面 10 步：

1. `eval/test.cpp` 读取 YAML，得到 `SystemConfig` 和 `ModelConfig`。
2. `Scheduler` 根据 batch、大模型长度、prefill/decode 规则生成人工请求或 trace 请求。
3. `Cluster` 根据 `num_node` 和 `num_device` 创建 `Node`/`Device`。
4. 每个 `Device` 初始化自己的 DRAM 接口与地址映射器。
5. `Model` 在每个 `Device` 上构建一份 `LLM` 模块图。
6. `LLM` 进一步展开为 embedding、decoder、attention、FFN、MoE、通信、sync 等模块。
7. `TopModuleGraph` 根据这些模块记录静态执行顺序与依赖关系。
8. 每轮仿真时，`Scheduler` 决定哪些序列跑 sum，哪些跑 gen，每条序列处理多少 token。
9. `Cluster::run()` 推动每个 device 的模块图执行；每个模块执行时由 `Executor` 调度到 GPU/Logic/PIM 的时延与访存模型。
10. `Cluster` 汇总每轮的 token 级延迟、t2ft/e2e、能耗、通信与 breakdown，并写入 CSV。

## 7. 代码目录到功能的对照

如果你接下来要继续读源码，建议按这个顺序：

1. 入口与配置
   - [eval/test.cpp](/home/zsy/LLMSimulator/eval/test.cpp:14)
2. 模型结构参数
   - [src/model/model_config.h](/home/zsy/LLMSimulator/src/model/model_config.h:9)
3. 顶层 LLM 组网
   - [src/model/model.h](/home/zsy/LLMSimulator/src/model/model.h:14)
   - [src/model/llm.cpp](/home/zsy/LLMSimulator/src/model/llm.cpp:10)
4. Transformer / MoE / MLA 模块
   - [src/module/decoder.cpp](/home/zsy/LLMSimulator/src/module/decoder.cpp:12)
   - [src/module/layer.cpp](/home/zsy/LLMSimulator/src/module/layer.cpp:11)
   - [src/module/parallel.cpp](/home/zsy/LLMSimulator/src/module/parallel.cpp:127)
   - [src/module/expert.cpp](/home/zsy/LLMSimulator/src/module/expert.cpp:13)
5. 调度与请求状态
   - [src/scheduler/scheduler.cpp](/home/zsy/LLMSimulator/src/scheduler/scheduler.cpp:4)
   - [src/scheduler/sequence.cpp](/home/zsy/LLMSimulator/src/scheduler/sequence.cpp:6)
6. 设备与硬件执行
   - [src/hardware/hardware_config.h](/home/zsy/LLMSimulator/src/hardware/hardware_config.h:16)
   - [src/hardware/device.cpp](/home/zsy/LLMSimulator/src/hardware/device.cpp:21)
   - [src/hardware/executor.cpp](/home/zsy/LLMSimulator/src/hardware/executor.cpp:13)
   - [src/hardware/linear_impl.cpp](/home/zsy/LLMSimulator/src/hardware/linear_impl.cpp:16)
   - [src/hardware/attention_sum_impl.cpp](/home/zsy/LLMSimulator/src/hardware/attention_sum_impl.cpp:16)
   - [src/hardware/attention_gen_impl.cpp](/home/zsy/LLMSimulator/src/hardware/attention_gen_impl.cpp:16)
7. 集群运行与结果统计
   - [src/hardware/cluster.cpp](/home/zsy/LLMSimulator/src/hardware/cluster.cpp:75)
   - [src/hardware/cluster.cpp](/home/zsy/LLMSimulator/src/hardware/cluster.cpp:424)
   - [src/hardware/cluster.cpp](/home/zsy/LLMSimulator/src/hardware/cluster.cpp:1084)

## 8. 你可以怎样理解这个仿真器

最简洁的理解方式是：

- 它不是“跑神经网络数值”的 simulator
- 它是“跑神经网络结构在某个硬件系统上的执行代价”的 simulator

因此它的建模对象不是 logits，而是：

- token 数
- 序列长度
- 模块形状
- FLOPs
- DRAM 流量
- KV cache 容量
- expert 路由
- all-reduce / scatter / gather
- 节点内外互连
- GPU/Logic/PIM 之间的执行选择

如果你下一步要继续深入，最值得继续看的两个方向是：

1. `src/module/`：理解“模型结构如何转成模块图”
2. `src/hardware/*_impl.cpp`：理解“每类模块的时延和访存公式具体是什么”

## 9. 相关文档

- 项目整体说明：[docs/PROJECT_OVERVIEW.zh-CN.md](/home/zsy/LLMSimulator/docs/PROJECT_OVERVIEW.zh-CN.md:1)
- 运行说明：[docs/RUN_GUIDE.zh-CN.md](/home/zsy/LLMSimulator/docs/RUN_GUIDE.zh-CN.md:1)
- CSV 字段说明：[docs/CSV_OUTPUT_GUIDE.zh-CN.md](/home/zsy/LLMSimulator/docs/CSV_OUTPUT_GUIDE.zh-CN.md:1)
