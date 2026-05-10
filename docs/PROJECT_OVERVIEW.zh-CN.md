# LLMSimulator 工程说明

## 1. 工程定位

LLMSimulator 是一个面向大语言模型推理系统研究的 C++ 周期级模拟器。它的核心任务不是“真的跑出模型回答”，而是把一次 LLM 推理请求在系统里的执行过程拆开，映射成一组可调度、可计时、可统计的模块执行与硬件事件，然后据此估算整套系统的性能、延迟、通信开销、显存压力和能耗。

如果把它和常见工具放在一起理解，可以把它看成下面这种定位：

- 它不像 `PyTorch`、`vLLM`、`TensorRT-LLM` 那样直接提供真实推理执行
- 它更像一个“LLM 系统研究模拟平台”
- 它适合做体系结构论文、系统优化实验、硬件设计空间探索

换句话说，这个工程关心的问题通常是：

- 某个大模型在某种 GPU 系统上大概会跑多快
- 多卡、多节点扩展后通信会不会成为瓶颈
- MoE、MLA、KV Cache 复用、Flash Attention 这些机制会怎样影响成本
- GPU + Logic 或 GPU + PIM 这样的异构设计是否值得
- DRAM/HBM 参数变化以后，Attention 或 FFN 的瓶颈会不会转移

所以，这个工程本质上是一个“为了回答系统问题而构建的模拟器”。

## 2. 这个工程解决什么问题

从代码和配置来看，LLMSimulator 主要解决下面几类问题。

### 2.1 模型结构带来的系统代价

不同模型结构会导致完全不同的执行和访存行为。例如：

- Dense 模型和 MoE 模型的 FFN 开销差异
- MHA、GQA、MQA、MLA 在 Attention 阶段的计算和 KV Cache 访问差异
- 路由专家数量、`top_k`、专家层分布对延迟和能耗的影响
- DeepSeekV3 这类前若干层 dense、后续大量 MoE 的混合结构对系统调度的影响

LLMSimulator 通过显式建模 Embedding、Decoder、Attention、Route、Expert、Communication 等模块，把模型结构变化转换成系统层面的变化。

### 2.2 并行策略和部署策略的收益

一个模型即使参数相同，不同并行方式的效果也会很不一样。这个工程重点支持评估：

- Expert Tensor Parallel
- Non-Expert Tensor Parallel
- Data Parallel 对 batch 和吞吐的影响
- 多卡、多节点规模变化带来的通信开销
- 是否启用并行执行优化
- 是否使用 hetero sub-batch
- 是否采用 disaggregated system 风格的系统组织

这类问题通常很难只靠公式估算，模拟器的价值就在于可以把多种策略放在统一框架下做可比实验。

### 2.3 内存系统和 PIM 架构的影响

LLM 推理里很多瓶颈都不是纯算力问题，而是内存带宽、访存模式和数据移动问题。这个工程通过自带 DRAM 建模层，以及对修改版 Ramulator2 的集成，可以更细粒度地观察：

- HBM/DRAM 配置变化后整体性能怎么变
- 内存访问是否成为某些层的主瓶颈
- PIM 指令是否真的能抵消数据搬运开销
- Logic-PIM、bank-level PIM、bank-group-level PIM 相比传统 GPU-only 方案是否有系统收益

### 2.4 服务负载和 batching 行为的影响

除了模型和硬件本身，真实推理系统还受到请求形态影响。这个工程通过 `scheduler` 层对序列和批处理进行建模，可以研究：

- 输入长度、输出长度变化对成本的影响
- batch size 增大后的吞吐/延迟变化
- 持续到达请求下的 queueing delay
- continuous batching 行为
- prefill 和 decode 两种阶段的差异
- KV cache 复用率变化带来的收益

因此它也适合做 serving 场景下的系统趋势分析，尽管它本身并不是 serving 框架。

## 3. 代码结构总览

整个仓库是一个 C++ + CMake 项目。核心目录可以理解为从“模型定义”一路走到“硬件执行与统计输出”的完整链条。

### 3.1 顶层文件

- `CMakeLists.txt`：顶层构建入口，组织各个子模块并生成 `run` 可执行文件
- `config.yaml`：主模拟配置文件
- `dram_config.yaml`、`dram_config_HBM3_80GB.yaml`、`dram_config_HBM3E_192GB.yaml`：内存系统相关配置
- `README.md`、`README.zh-CN.md`：简要说明
- `patch/ramulator2_pim.patch`：对 Ramulator2 的定制补丁

### 3.2 `src/model`

这一层负责描述“要模拟的模型是什么”。

关键内容包括：

- `model_config.h`
- `model.h`
- `llm.h`
- `llm.cpp`

这里定义了 `ModelConfig`，并内置了若干模型模板，例如：

- `mixtral`
- `openMoE`
- `llama7bMoE`
- `llama3_405B`
- `grok1`
- `deepseekV3`
- `llama4_scout`
- `llama4_maverick`

`ModelConfig` 里包含大量和系统建模直接相关的参数，例如：

- `hidden_dim`
- `head_dim`
- `num_layers`
- `num_heads`
- `num_kv_heads`
- `max_seq_len`
- `intermediate_dim`
- `expert_intermediate_dim`
- `num_routed_expert`
- `top_k`
- `expert_freq`
- `first_k_dense`
- `q_lora_rank`
- `kv_lora_rank`
- `compressed_kv`
- `use_absorb`

这些参数不是只为了“表示模型长什么样”，而是直接决定后续模块规模、算子形态、KV Cache 行为和执行代价。

### 3.3 `src/module`

这一层负责把一个 LLM 拆成一组模块化的执行单元，是整个工程最接近“神经网络结构”的部分。

你会看到大量模块实现，例如：

- `embedding`
- `decoder`
- `layer`
- `attention`
- `route`
- `expert`
- `linear`
- `layernorm`
- `residual`
- `rope`
- `lm_head`
- `communication`
- `parallel`
- `compressed_kv_restore`

从职责上理解：

- `Embedding` 负责输入嵌入阶段
- `Decoder` / `MoEDecoder` 负责一层或一类解码层结构
- `Attention` 负责注意力相关计算
- `Route` 与 `Expert` 负责 MoE 路由与专家执行
- `Communication` 负责并行系统里的数据通信建模
- `Parallel` 负责与张量并行等机制相关的逻辑

这一层的一个重要特点是：它不是只描述数学意义上的前向传播，而是已经把很多“系统里要付出的代价”也抽象进模块行为里了。

### 3.4 `src/scheduler`

这一层负责“请求如何进入系统、如何组成 batch、如何随时间推进”。

关键对象包括：

- `Sequence`
- `BatchedSequence`
- `Scheduler`

这一层负责的事情大致包括：

- 生成 synthetic 请求
- 读取真实专家路由数据
- 维护等待队列和运行队列
- 按 batch / token 组织调度信息
- 维护 prefill / decode 过程中的序列状态
- 统计请求到达时间和排队行为

如果把整个工程看成一个 serving 系统的简化研究版，那么 `scheduler` 就是其中“工作负载与调度器”的抽象。

### 3.5 `src/hardware`

这一层负责把模型执行映射到抽象硬件上，是整个模拟器的“系统平台层”。

主要对象包括：

- `SystemConfig`
- `Cluster`
- `Node`
- `Device`
- `Executor`
- `Stat`

可以这样理解：

- `SystemConfig` 描述硬件平台参数
- `Cluster` 描述整个集群
- `Node` 描述单机节点
- `Device` 描述单个设备，如 GPU/Logic/PIM 单元
- `Executor` 负责根据 layer 类型调用相应执行路径
- `Stat` 汇总每轮模拟统计结果

这里建模的不是单纯“有几张卡”，而是较完整的系统属性，包括：

- 节点内互连延迟与带宽
- 节点间互连延迟与带宽
- 设备计算峰值 FLOPS
- 设备显存带宽
- 显存容量
- GPU / LOGIC / PIM 的不同执行能力

### 3.6 `src/dram`

这一层负责更细粒度的 DRAM / PIM 建模。

主要内容包括：

- `dram_interface`
- `dram_request`
- `memory_object`
- `memory_config`
- `pim_request`

这里的职责包括：

- 把高层模块的访存需求转换为 DRAM 请求
- 在需要时接入 Ramulator2 做更细粒度模拟
- 统计 ACT/READ/WRITE 等命令行为
- 为后续能耗统计提供基础

### 3.7 `eval`

`eval/test.cpp` 是这个工程最重要的运行入口。它负责：

1. 读取 YAML 配置
2. 选择模型模板
3. 组装系统配置
4. 创建 `Scheduler`
5. 创建 `Cluster`
6. 创建并分发 `Model`
7. 检查容量限制
8. 运行若干次模拟迭代
9. 导出 CSV 统计结果
10. 可选导出 gantt 数据和时间线日志

如果你想快速理解“程序从哪里开始，最后生成了什么”，优先看这个文件最有效。

## 4. 程序是怎么运行起来的

从执行链路看，整个工程大致遵循下面的流程。

### 4.1 读取配置

程序启动后，`eval/test.cpp` 会优先读取传入的 YAML 配置文件；如果命令行没有传参，就默认读取当前目录下的 `config.yaml`。

配置主要分为几类：

- `model`
- `system`
- `serving`
- `simulation`
- `log`

### 4.2 选择模型模板

根据 `model.model_name`，程序会从内置模板中选择对应 `ModelConfig`。当前代码中可以识别的模型包括：

- `mixtral`
- `openMoE`
- `llama7bMoE`
- `llama3_405B`
- `grok1`
- `deepseekV3`
- `llama4_scout`
- `llama4_maverick`

然后再根据 YAML 中的附加配置覆盖其中部分行为，例如：

- 并行度
- 精度字节数
- `compressed_kv`
- `use_absorb`
- `skewness`
- 输入输出长度

### 4.3 选择硬件平台

根据 `system.gpu_gen`，程序会选择预定义的硬件模板，例如：

- `A100`
- `H100`
- `B100`
- `B200`

随后再根据配置继续调整：

- `nvlink_gen`
- `infiniband_gen`
- `num_node`
- `num_device`
- `processor_type`
- 若干优化开关

因此，硬件平台并不是一个固定写死的对象，而是“预置模板 + YAML 定制”的组合。

### 4.4 创建调度器和集群

接下来程序会创建：

- `Scheduler`
- `Cluster`

`Scheduler` 负责工作负载和 batch 行为，`Cluster` 负责底层执行平台。然后 `Model` 会被分发到 cluster 里的各个设备上，建立完整的模块图和依赖关系。

### 4.5 执行模拟

在真正开始迭代前，程序会先做一次容量检查，例如判断是否 out-of-memory。

之后进入 `runIteration` 流程。每次迭代会推进请求、执行模块、累计时间、累计能耗并记录统计结果。

如果开启了：

- `print_log`
- `export_gantt`

还会输出时间板日志或 gantt 数据。

### 4.6 导出结果

最终结果会以 CSV 文件形式写入输出目录。输出文件名里通常包含大量实验参数，例如：

- 模型名
- 数据名
- 输入长度
- 输出长度
- 处理器类型
- 节点数
- 设备数
- TP/DP 规模
- batch 限制
- process token 限制
- 迭代次数
- skewness
- precision byte
- 是否 parallel execution
- 是否 Ramulator
- 当前是 prefill / decode / 默认模式

这说明作者希望它能直接服务于“批量跑实验，然后整理结果”的研究工作流。

## 5. 当前支持哪些模型与机制

从源码中能直接确认，当前工程已经支持或显式建模了下列能力。

### 5.1 模型类型

- Dense Transformer
- MoE Transformer
- Dense + MoE 混合分层结构

### 5.2 Attention 机制

- MHA
- GQA
- MQA
- MLA

### 5.3 MoE 机制

- Routed experts
- `top_k` 专家选择
- 专家层周期性分布
- DeepSeekV3 风格的“前 K 层 dense，后续层 MoE”

### 5.4 KV 相关机制

- `compressed_kv`
- `reuse_kv_cache`
- `kv_cache_reuse_rate`

### 5.5 执行优化开关

- `use_flash_attention`
- `use_flash_mla`
- `parallel_execution`
- `hetero_subbatch`
- `disagg_system`
- `use_low_unit_moe_only`
- `prefill_mode`
- `decode_mode`

这些能力的意义在于，它不是只支持“一个静态模型”，而是允许我们把很多现代 LLM 系统设计点映射到模拟参数里。

## 6. 当前支持哪些硬件与系统实验

### 6.1 GPU 代际

当前代码显式提供了：

- `A100`
- `H100`
- `B100`
- `B200`

每种模板都包含：

- 计算峰值
- 显存带宽
- 显存容量
- 设备内互连参数
- 节点间互连参数

### 6.2 处理器类型

当前配置支持的处理器类型组合包括：

- `GPU`
- `LOGIC`
- `GPU+LOGIC`
- `GPU+PIM`

这说明该工程并不是只模拟标准 GPU 系统，也尝试研究异构系统执行。

### 6.3 互连配置

支持配置：

- `nvlink_gen`
- `infiniband_gen`

因此可以研究：

- 节点内带宽受限场景
- 多节点扩展场景
- 通信成本对 TP/DP 设计的影响

### 6.4 容量与带宽约束

系统会检查内存容量相关问题，并支持：

- `exit_out_of_memory`
- `mem_cap_limit`

这使它不只是“算执行时间”，还可以回答“这套部署方案是否装得下”。

## 7. 可以完成哪些典型任务

这一部分更偏使用视角，回答“拿这个工程可以做什么实验”。

### 7.1 比较不同模型在同一系统上的代价

例如：

- `mixtral` 和 `deepseekV3` 在 8 卡 B100 上哪个 decode 更贵
- `llama3_405B` 和 `grok1` 的 Attention / FFN 占比差异

### 7.2 比较不同硬件代际的收益

例如：

- 同一模型从 H100 切换到 B100，延迟下降多少
- 带宽提升是否真的换来等比例收益

### 7.3 比较不同并行度和部署规模

例如：

- TP 从 1 提升到 2、4 后收益是否被通信吃掉
- 多节点扩展后 DP 吞吐是否更优

### 7.4 比较 prefill 和 decode 阶段行为

例如：

- 长 prompt prefill 时瓶颈在计算还是显存带宽
- decode 模式下 KV cache 复用是否带来明显收益

### 7.5 比较异构系统设计

例如：

- `GPU` 与 `GPU+PIM` 的对比
- `GPU+LOGIC` 是否更适合某类 MoE 工作负载

### 7.6 比较不同访存建模精度

例如：

- 理想内存模型和 `use_ramulator` 细粒度模型下结果差多少
- DRAM 命令级行为是否改变结论

### 7.7 比较不同路由负载分布

例如：

- 专家负载均衡和偏斜负载下，MoE 系统的尾延迟差异
- `skewness` 增大后，专家热点是否更严重

## 8. 输出结果通常包含什么

从 `Stat` 结构体可以看出，这个工程不仅输出一个总时间，而是会输出较完整的系统统计。

### 8.1 时延与吞吐相关

- 总执行时间
- 单请求延迟
- 排队延迟
- 到达时间
- 队列大小

### 8.2 batch 与序列相关

- 当前 batch size
- 处理 token 数
- sum/gen 序列数
- 平均序列长度
- 输入长度
- 输出长度

### 8.3 模块时间拆解

- `qkv_gen`
- `atten_sum`
- `atten_gen`
- `o_proj`
- `ffn`
- `expert_ffn`
- `communication`
- `rope`
- `layernorm`
- `residual`

对于 MLA，还会有更细的拆分，例如：

- `q_down_proj`
- `kv_down_proj`
- `q_up_proj`
- `qr_proj`
- `kv_up_proj`
- `tr_k_up_proj`
- `v_up_proj`

### 8.4 能耗相关

- `act_energy`
- `read_energy`
- `write_energy`
- `all_act_energy`
- `all_read_energy`
- `all_write_energy`
- `mac_energy`
- `total_energy`

还会进一步按功能块拆解：

- `FC_DRAM_energy`
- `FC_COMP_energy`
- `Attn_DRAM_energy`
- `Attn_COMP_energy`
- `MoE_DRAM_energy`
- `MoE_COMP_energy`

### 8.5 其他状态

- 是否 OOM
- 是否 mixed stage
- split 信息
- 请求类型标记

因此它的输出非常适合拿去做论文图表、表格统计和瓶颈分析。

## 9. 配置文件怎么理解

`config.yaml` 是这个工程的主要实验入口。理解它，基本就理解了工程如何被使用。

### 9.1 `model`

示例：

```yaml
model:
  model_name: deepseekV3
```

这一段决定选择哪个内置模型模板。

### 9.2 `system`

这一段决定模拟的硬件系统，例如：

- GPU 代际
- NVLink 代际
- InfiniBand 代际
- 节点数、设备数
- 处理器类型
- 并行与优化开关

示例中的：

- `gpu_gen: B100`
- `num_node: 1`
- `num_device: 8`

表示当前默认实验在 1 节点 8 卡 B100 风格系统上进行。

### 9.3 `serving`

这一段偏向服务负载与 batch 限制，例如：

- `max_batch_size`
- `max_process_token`

用于控制调度器可以同时处理多少序列和 token。

### 9.4 `simulation`

这一段偏向实验输入特征，例如：

- 数据来源
- 输入长度
- 输出长度
- 精度字节数
- 偏度
- 迭代次数
- 请求注入率
- OOM 行为

如果 `data: synthesis`，说明使用合成请求，而不是外部专家路由数据。

### 9.5 `log`

这一段控制输出行为，例如：

- 是否打印日志
- 是否导出 gantt
- CSV 输出目录
- gantt 目录

## 10. 当前默认配置代表什么实验

结合仓库里的 `config.yaml`，当前默认配置大致表示：

- 模型：`deepseekV3`
- 系统：`B100`
- 拓扑：`1` 节点、`8` 设备
- 处理器：`GPU`
- 运行阶段：`decode_mode: on`
- 输入数据：`synthesis`
- 输入长度：`1024`
- 输出长度：`10`
- 精度：`1 byte`
- 启用：
  - `compressed_kv`
  - `use_absorb`
  - `use_flash_mla`
  - `use_flash_attention`
  - `reuse_kv_cache`

因此，默认配置并不是一个“随便的 demo”，而更像一个针对现代大模型 decode 阶段的研究实验模板。

## 11. 这个工程适合谁使用

比较适合下面几类人：

- 做计算机体系结构、系统、编译优化相关研究的同学
- 想分析 LLM 推理瓶颈的研究人员
- 想比较不同硬件配置成本与收益的工程师
- 想快速验证“某种系统设计是否可能有收益”的团队

如果你的目标是：

- 部署在线大模型服务
- 做真实模型推理加速
- 输出模型文本结果

那这个工程就不是最合适的选择。

## 12. 它不擅长什么

为了避免误用，也需要明确它的边界。

### 12.1 它不负责真实数值正确性推理

这个工程并不追求像深度学习框架那样给出真实 token 输出，它主要关心系统成本，而不是文本正确性。

### 12.2 它不是训练框架

代码整体围绕推理、请求、KV cache、MoE 路由、系统执行展开，不是用来做大模型训练模拟的主框架。

### 12.3 它的结论依赖建模假设

作为模拟器，结果质量依赖：

- 模型抽象是否合理
- 硬件参数是否准确
- 内存模型是否贴近目标平台
- 请求分布是否代表真实场景

因此它更适合做趋势分析、相对比较和设计空间探索，而不是把某个绝对数值当成真实线上 SLA。

## 13. 建议如何阅读这个工程

如果你准备继续深入代码，比较推荐的阅读顺序是：

1. 先看 [README.zh-CN.md](/home/zsy/LLMSimulator/README.zh-CN.md:1) 和当前文档
2. 再看 [config.yaml](/home/zsy/LLMSimulator/config.yaml:1)，理解实验入口
3. 接着看 [eval/test.cpp](/home/zsy/LLMSimulator/eval/test.cpp:1)，理解主流程
4. 再看 [src/model/model_config.h](/home/zsy/LLMSimulator/src/model/model_config.h:1)，理解内置模型模板
5. 然后看 `src/module`，理解模型模块如何构图
6. 再看 `src/hardware` 和 `src/scheduler`，理解执行与调度
7. 最后看 `src/dram` 和 Ramulator2 接口，理解细粒度内存建模

如果你的目标是“先能跑实验”，优先关注：

- `config.yaml`
- `eval/test.cpp`
- 输出 CSV 的字段

如果你的目标是“改模型或加机制”，优先关注：

- `src/model`
- `src/module`

如果你的目标是“改硬件建模”，优先关注：

- `src/hardware`
- `src/dram`

## 14. 一句话总结

LLMSimulator 是一个面向大语言模型推理系统研究的 C++ 周期级模拟平台，用来回答“某个模型在某种硬件、并行策略、调度方式和内存系统下，性能、延迟、通信和能耗会怎样变化”，非常适合做系统论文实验、硬件设计探索和 LLM 推理架构对比，但并不是一个直接拿来部署真实推理服务的框架。
