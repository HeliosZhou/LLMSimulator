# LLMSimulator 数学机理与代码对照

<div align="center">
  <h2>这个仿真器到底怎样“模拟一个 LLM”？</h2>
  <p>
    答案不是“算出真实 token”，而是“把 LLM 前向传播翻译成可计时、可计量、可调度的系统事件”。
  </p>
</div>

<table>
  <tr>
    <td width="25%" valign="top"><b>模型数学</b><br/>Linear、Attention、MLA、MoE</td>
    <td width="25%" valign="top"><b>张量形状</b><br/>batch、seq_len、head_dim、expert token</td>
    <td width="25%" valign="top"><b>系统代价</b><br/>FLOPs、bytes、通信量、KV cache</td>
    <td width="25%" valign="top"><b>硬件时间</b><br/>Roofline、Ramulator、GPU/LOGIC/PIM</td>
  </tr>
</table>

## 1. 先说结论

LLMSimulator 对 LLM 的仿真可以概括成下面这个公式链：

```text
LLM 请求
  -> Scheduler 决定本轮处理哪些 token
  -> Model / Module 把 LLM 展开成模块图
  -> 每个模块根据张量形状得到 FLOPs 和访存量
  -> Executor 在 GPU / LOGIC / PIM 候选执行器中估算时间
  -> TopModuleGraph 累计每张卡的时间线和能耗
  -> Cluster 汇总本轮 token 延迟、吞吐、能耗并写入 CSV
```

它不保存真实权重值，不执行真实矩阵乘法，也不计算真实 logits。它保存的是：

- 张量形状：例如 `[m, k]`、`[seq_len, head_dim]`
- 精度字节数：`precision_byte`
- 模块类型：Linear、Attention、Activation、AllReduce、MoE Route 等
- 硬件参数：峰值 FLOPs、HBM 带宽、互连带宽、PIM/Logic 带宽放大系数
- 请求状态：当前序列长度、prefill/decode 阶段、本轮处理 token 数

最核心的时间模型是 Roofline：

```text
T_compute = FLOPs / peak_flops
T_memory  = memory_bytes / memory_bandwidth
T_layer   = max(T_compute, T_memory)
```

源码中统一换算成 ns：

```cpp
compute_duration = total_flops / compute_peak_flops * 1000 * 1000 * 1000;
memory_duration = total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;
total_duration = std::max(compute_duration, memory_duration);
```

对应代码：[src/hardware/linear_impl.cpp](../src/hardware/linear_impl.cpp#L27-L62)。

## 2. 一条请求如何变成本轮 token

LLM 推理有两个阶段：

- prefill / sum：处理 prompt，上下文长度快速增长，一轮可处理多个 token
- decode / gen：自回归生成，通常每条序列一轮只生成 1 个 token

调度器给每条 `Sequence` 维护这些状态：

| 状态 | 含义 |
|---|---|
| `input_len` | prompt 长度 |
| `output_len` | 要生成的长度 |
| `current_len` | 当前已经处理到的长度 |
| `num_process_token` | 本轮要处理几个 token |
| `sum_stage` | 是否还在 prefill / sum 阶段 |

本轮 token 数由 `Scheduler::setMetadata()` 决定：

- 对 decode 序列：`num_process_token = 1`
- 对 prefill 序列：把 `max_process_token` 在多个 sum 序列之间均分

源码入口：

- 本轮 metadata 生成：[src/scheduler/scheduler.cpp](../src/scheduler/scheduler.cpp#L260)
- 本轮之后更新序列：[src/scheduler/scheduler.cpp](../src/scheduler/scheduler.cpp#L311)
- sum/gen split 更新：[src/scheduler/scheduler.cpp](../src/scheduler/scheduler.cpp#L334)

可以把 `num_process_token` 理解成所有后续公式里的 `m`，也就是“本轮这个模块实际要处理多少 token”。

## 3. LLM 数学结构如何变成模块图

模型结构不是在运行时动态猜出来的，而是在构建 `Model` 时展开成模块图。

```text
LLM
  Embedding
  Decoder_0
    LayerNorm
    Attention
    Residual
    LayerNorm
    FFN 或 ExpertFFN
    AllReduce
    Residual
  ...
  LMHead
```

关键源码：

- 模型参数模板：[src/model/model_config.h](../src/model/model_config.h#L9)
- 每张卡构造一份 LLM 图：[src/model/model.h](../src/model/model.h#L14)
- 顶层 LLM 组网：[src/model/llm.cpp](../src/model/llm.cpp#L10)
- 普通 Decoder：[src/module/decoder.cpp](../src/module/decoder.cpp#L12)
- Attention / FFN / MLA 组网：[src/module/layer.cpp](../src/module/layer.cpp#L11)
- MoE 组网：[src/module/expert.cpp](../src/module/expert.cpp#L13)

每个 `Module` 被调用时都会进入 `TopModuleGraph`：

```cpp
device->top_module_graph->push_module_graph(getptr(), input);
return_tensor = forward(input, sequences_metadata);
device->top_module_graph->pop_module_graph(return_tensor);
```

对应代码：[src/module/module.cpp](../src/module/module.cpp#L8-L15)。

这表示一次“前向传播”在仿真器里不是为了求数值输出，而是为了注册和执行一串可统计的模块事件。

## 4. 张量大小的基本数学

所有访存量都来自张量形状和精度：

```text
bytes(tensor) = precision_byte * product(shape)
```

源码就是 `Tensor::getSize()`：

```cpp
long size = precision_byte;
for (int dim : shape) {
  size *= dim;
}
```

对应代码：[src/module/tensor.cpp](../src/module/tensor.cpp#L15-L21)。

因此，只要某个模块改变了张量 shape，后续 `getSize()` 就会直接改变访存估算。例如 Attention 里会临时把输入张量 shape 改成 query、score、KV cache 的形状，再调用内存模型。

## 5. 通用硬件执行模型

每个可执行模块最终都会调用：

```cpp
device->execution(LayerType::..., tensor_list, sequences_metadata, layer_info);
```

然后进入 `Executor`：

- `Device::execution()`：[src/hardware/device.cpp](../src/hardware/device.cpp#L152)
- `Executor::execution()`：[src/hardware/executor.cpp](../src/hardware/executor.cpp#L135)
- `Executor::executePType()`：[src/hardware/executor.cpp](../src/hardware/executor.cpp#L157)

`Executor` 做两件事：

1. 根据 `LayerType` 找到对应执行函数，例如 Linear 找 `LinearExecutionGPU`。
2. 如果 `layer_info.processor_type` 里有多个候选处理器，就分别估算时间，取 `total_duration` 最小的结果。

简化伪代码：

```cpp
for (auto type : layer_info.processor_type) {
  status = executePType(layer_type, tensor_list, sequences_metadata, type, ...);
  if (duration == 0 || duration > status.total_duration) {
    optimal_status = status;
    duration = status.total_duration;
  }
}
device->setExecStatus(optimal_status);
```

对应代码：[src/hardware/executor.cpp](../src/hardware/executor.cpp#L135-L153)。

## 6. Linear：矩阵乘如何仿真

LLM 里绝大部分投影都可以落到 Linear：

```text
Y = XW
X: [m, k]
W: [k, n]
Y: [m, n]
```

数学代价：

```text
FLOPs = 2 * m * k * n
Bytes = (m*k + k*n + m*n) * precision_byte
T_compute = FLOPs / compute_peak_flops
T_memory  = Bytes / memory_bandwidth
T_total   = max(T_compute, T_memory)
```

代码对应关系：

| 数学量 | 代码 |
|---|---|
| `m = input.shape[0]` | [src/hardware/linear_impl.cpp](../src/hardware/linear_impl.cpp#L23) |
| `k = input.shape[1]` | [src/hardware/linear_impl.cpp](../src/hardware/linear_impl.cpp#L24) |
| `n = weight.shape[1]` | [src/hardware/linear_impl.cpp](../src/hardware/linear_impl.cpp#L25) |
| `2*m*k*n` | [src/hardware/linear_impl.cpp](../src/hardware/linear_impl.cpp#L27) |
| input + weight + output bytes | [src/hardware/linear_impl.cpp](../src/hardware/linear_impl.cpp#L28) |
| `max(compute, memory)` | [src/hardware/linear_impl.cpp](../src/hardware/linear_impl.cpp#L61) |

<details>
  <summary><b>为什么是 2*m*k*n？</b></summary>

  矩阵乘 `Y[m,n] = X[m,k] @ W[k,n]` 中，每个输出元素需要 `k` 次乘法和 `k` 次加法，约为 `2k` FLOPs。共有 `m*n` 个输出元素，所以总计 `2*m*k*n`。
</details>

### GPU、LOGIC、PIM 的差别

同一个 Linear 可以被三类处理器估算：

| 执行器 | 峰值算力来源 | 带宽来源 | 代码入口 |
|---|---|---|---|
| GPU | `config.compute_peak_flops` | `config.memory_bandwidth` | `LinearExecutionGPU` |
| LOGIC | `logic_memory_bandwidth * logic_op_b` | `logic_memory_bandwidth` | `LinearExecutionLogic` |
| PIM | `pim_memory_bandwidth * pim_op_b` | `pim_memory_bandwidth` | `LinearExecutionPIM` |

代码入口：

- GPU：[src/hardware/linear_impl.cpp](../src/hardware/linear_impl.cpp#L16)
- LOGIC：[src/hardware/linear_impl.cpp](../src/hardware/linear_impl.cpp#L77)
- PIM：[src/hardware/linear_impl.cpp](../src/hardware/linear_impl.cpp#L140)

当 `precision_byte == 1` 时，LOGIC/PIM 路径会把等效算力乘 2，用来近似 FP8 更高吞吐。

## 7. FFN：由多个 Linear 拼出来

普通 Transformer FFN 在仿真器里不是一个单独的大公式，而是拆成 Linear 和 Activation。

### 2-way FFN

```text
up = Linear(x)              # hidden_dim -> intermediate_dim * activation_factor
act = Activation(up)
out = Linear(act)           # intermediate_dim -> hidden_dim
```

代码：

- 构造：[src/module/layer.cpp](../src/module/layer.cpp#L292)
- forward：[src/module/layer.cpp](../src/module/layer.cpp#L328)

### 3-way / SwiGLU 风格 FFN

```text
gate = Linear(x)
act  = Activation(gate)
up   = Linear(x)
out  = Linear(up)
```

代码：

- 构造：[src/module/layer.cpp](../src/module/layer.cpp#L351)
- forward：[src/module/layer.cpp](../src/module/layer.cpp#L415)

注意：当前代码中 `Activation` 的硬件模型主要把它视为访存型算子。GPU 路径下 `total_flops = 0`，访存为 `gate_output + input + output`；LOGIC/PIM 路径会用 `total_memory_size` 近似一份操作量。

对应代码：[src/hardware/activation_impl.cpp](../src/hardware/activation_impl.cpp#L16)。

## 8. 标准 Attention：prefill 和 decode 的差别

标准 attention 数学形式：

```text
score = QK^T
prob  = softmax(score)
out   = prob V
```

仿真器把它拆成三段：

1. Scoring：`QK^T`
2. Softmax：scale + mask + softmax
3. Context：`prob V`

### 8.1 Prefill / AttentionSum

入口：

- 模块层：[src/module/attention.cpp](../src/module/attention.cpp#L83)
- 硬件层：[src/hardware/attention_sum_impl.cpp](../src/hardware/attention_sum_impl.cpp#L16)

对于每条 sum 序列：

```text
m = seq.num_process_token
k = head_dim
n = seq.current_len + seq.num_process_token
```

Scoring：

```text
FLOPs_score = 2 * m * k * n * num_heads
Bytes_score = (m*k*num_heads + k*n*num_kv_heads + m*n*num_heads) * precision_byte
```

代码：

- `m/k/n`：[src/hardware/attention_sum_impl.cpp](../src/hardware/attention_sum_impl.cpp#L59-L61)
- FLOPs：[src/hardware/attention_sum_impl.cpp](../src/hardware/attention_sum_impl.cpp#L63)
- Bytes：[src/hardware/attention_sum_impl.cpp](../src/hardware/attention_sum_impl.cpp#L66-L68)

Softmax：

```text
FLOPs_softmax = 7 * m * n * num_heads
```

代码：[src/hardware/attention_sum_impl.cpp](../src/hardware/attention_sum_impl.cpp#L145-L163)。

Context：

```text
m = seq.num_process_token
k = seq.current_len + seq.num_process_token
n = head_dim

FLOPs_context = 2 * m * k * n * num_heads
Bytes_context = (m*k*num_heads + k*n*num_kv_heads + m*n*num_heads) * precision_byte
```

代码：

- `m/k/n`：[src/hardware/attention_sum_impl.cpp](../src/hardware/attention_sum_impl.cpp#L212-L214)
- FLOPs：[src/hardware/attention_sum_impl.cpp](../src/hardware/attention_sum_impl.cpp#L216)
- Bytes：[src/hardware/attention_sum_impl.cpp](../src/hardware/attention_sum_impl.cpp#L219-L221)

每一段都按 Roofline 取 `max(compute_duration, memory_duration)`，再把三段加起来。

### 8.2 Decode / AttentionGen

入口：

- 模块层：[src/module/attention.cpp](../src/module/attention.cpp#L11)
- 硬件层：[src/hardware/attention_gen_impl.cpp](../src/hardware/attention_gen_impl.cpp#L16)

decode 的关键是 `m` 通常为 1，但 `n = current_len + 1` 会随着上下文增长。因此 decode attention 的瓶颈经常来自 KV cache 读取。

Scoring：

```text
m = seq.num_process_token        # decode 中通常为 1
k = head_dim
n = seq.current_len + seq.num_process_token

FLOPs_score = 2 * m * k * n * attention_group_size * num_kv_heads
```

源码里是对每个 KV head 循环：

```cpp
for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
  flops = m * k * n * 2.0 * attention_group_size;
}
```

对应代码：[src/hardware/attention_gen_impl.cpp](../src/hardware/attention_gen_impl.cpp#L60-L80)。

KV cache 读取：

```cpp
k_cache->setShape({accumul_len, head_dim * num_kv_heads});
issueRamulator(..., k_cache);
```

对应代码：[src/hardware/attention_gen_impl.cpp](../src/hardware/attention_gen_impl.cpp#L84-L98)。

Context 阶段同理读取 `v_cache`：

- FLOPs 与 bytes：[src/hardware/attention_gen_impl.cpp](../src/hardware/attention_gen_impl.cpp#L121-L145)
- V cache 读取：[src/hardware/attention_gen_impl.cpp](../src/hardware/attention_gen_impl.cpp#L149-L163)

<details>
  <summary><b>为什么 decode 更容易 memory-bound？</b></summary>

  decode 每轮只有 1 个新 token，矩阵乘的 `m` 很小，GPU 很难把算力打满；但它仍要读取长度为 `current_len` 的 K/V cache。上下文越长，cache bytes 越大，所以延迟常被 HBM/PIM/Logic 带宽决定。
</details>

## 9. MLA：低秩 KV 与吸收路径

MLA 的目的之一是减少 KV cache 压力。普通 attention 存完整 K/V，而 MLA 会存 latent KV：

```text
c_kv = x W_DKV
k/v  = c_kv W_UKV
```

相关模型参数在 `ModelConfig`：

| 参数 | 含义 |
|---|---|
| `q_lora_rank` | Q 的低秩维度 |
| `kv_lora_rank` | latent KV 维度 |
| `qk_nope_head_dim` | 非 RoPE 的 Q/K 维度 |
| `qk_rope_head_dim` | RoPE 维度 |
| `compressed_kv` | 是否使用压缩 KV |
| `use_absorb` | 是否启用 absorb MLA |

代码入口：

- MLA 模块组网：[src/module/layer.cpp](../src/module/layer.cpp#L56)
- 非吸收 decode 模块：[src/module/attention.cpp](../src/module/attention.cpp#L345)
- 非吸收 prefill 模块：[src/module/attention.cpp](../src/module/attention.cpp#L396)
- 吸收 decode 模块：[src/module/attention.cpp](../src/module/attention.cpp#L475)
- 吸收 prefill 模块：[src/module/attention.cpp](../src/module/attention.cpp#L530)

### 9.1 非吸收 MLA

非吸收路径可以理解为：

```text
score = Q * K^T
K,V   = restore(CKV)
```

也就是说，decode 时要把压缩的 `CKV` 恢复到 attention 需要的 K/V 形态。这个路径在模块层会出现 `c_kv_restore`、`attn_kv_up_proj`、`multi_latent_attention` 等模块。

组网代码：[src/module/layer.cpp](../src/module/layer.cpp#L138)。

硬件执行入口：

- `LayerType::MLA_GEN`：[src/hardware/executor.cpp](../src/hardware/executor.cpp#L190)
- `MultiLatentAttentionGenExecutionGPU`：[src/hardware/attention_gen_impl.cpp](../src/hardware/attention_gen_impl.cpp#L489)

### 9.2 吸收 MLA

吸收路径利用矩阵乘结合律，把一部分上投影吸收到 query 侧：

```text
非吸收: score = Q * (CKV * W_UK)^T
吸收:   score = (Q * W_UK^T) * CKV^T
```

这样做的系统含义是：

- 不必显式恢复完整 K cache 再做 scoring
- scoring 可以直接在 latent KV 空间里做
- decode 阶段对长上下文的访存和中间张量压力更可控

模块层路径包括：

- `attn_tr_k_up_proj`
- `attn_mla_absorbed`
- `attn_v_up_proj`
- `attn_o_proj`

组网代码：[src/module/layer.cpp](../src/module/layer.cpp#L95)。

硬件执行入口：

- `LayerType::ABSORBED_MLA_GEN`：[src/hardware/executor.cpp](../src/hardware/executor.cpp#L199)
- `AbsorbMLAGenExecutionGPU`：[src/hardware/attention_gen_impl.cpp](../src/hardware/attention_gen_impl.cpp#L1469)

### 9.3 吸收路径的公式直觉

在吸收 decode 里，注意力被拆成 NoPE 和 RoPE 两部分：

```text
NoPE scoring:
  k = kv_lora_rank
  n = current_len

RoPE scoring:
  k = qk_rope_head_dim
  n = current_len

Context:
  k = current_len
  n = kv_lora_rank
```

源码中可以看到：

- NoPE scoring 使用 `kv_lora_rank`：[src/hardware/attention_gen_impl.cpp](../src/hardware/attention_gen_impl.cpp#L1993)
- RoPE scoring 使用 `qk_rope_head_dim`：[src/hardware/attention_gen_impl.cpp](../src/hardware/attention_gen_impl.cpp#L2066)
- Context 输出 latent 维度：[src/hardware/attention_gen_impl.cpp](../src/hardware/attention_gen_impl.cpp#L2194)

这就是 `use_absorb` 对数学图和系统代价同时产生影响的地方。

## 10. MoE：专家路由如何进入公式

MoE 层不是简单地把 FFN 参数乘上专家数。每个 token 只路由到 `top_k` 个专家。

数学上可以写成：

```text
gate = x W_gate
selected_experts = TopK(gate, top_k)
y = sum_{e in selected_experts} gate_e * Expert_e(x)
```

仿真器重点建模的是系统代价：

- gate projection 的 Linear 代价
- token 分配到各 expert 的数量
- MoE scatter / gather 通信
- expert FFN 计算
- expert tensor parallel 的 all-reduce
- shared expert 的额外 FFN

模块结构在 `ExpertFFN` 里：

```text
gate_fn
gate_update
moe_scatter
moe_route
expert_FFN_*
moe_all_reduce_for_e_tp
moe_gather
moe_all_reduce_for_gather
shared_expert_FFN_*
```

对应代码：

- MoE 构造：[src/module/expert.cpp](../src/module/expert.cpp#L13)
- MoE forward：[src/module/expert.cpp](../src/module/expert.cpp#L136)
- 专家 token 聚合：[src/module/route.cpp](../src/module/route.cpp#L149)
- 路由输入张量：[src/module/route.cpp](../src/module/route.cpp#L211)

### Expert token 数如何影响计算

每个 expert 的 FFN 输入 token 数来自 `sequences_metadata->num_token_in_expert[e]`。因此 expert 的 Linear 公式仍然是：

```text
FLOPs_expert_linear = 2 * m_expert * hidden_dim * expert_intermediate_dim
```

区别只是 `m` 不再是 batch token 总数，而是该 expert 本轮收到的 token 数。

这也是 MoE 性能很依赖路由偏斜的原因：如果 token 集中到少数 expert，某些设备的 `m_expert` 会变大，导致负载不均。

## 11. 通信：AllReduce 和 MoE Scatter/Gather

### 11.1 AllReduce

`AllReduce` 的通信时间模型是 ring 风格的简化估算：

```text
size_per_device = input_bytes / num_devices
one_hop = latency + size_per_device / bandwidth
hop = 2 * (num_devices - 1)
T_allreduce = one_hop * hop
```

源码：

- `hop = (device_list.size() - 1) * 2`：[src/module/communication.cpp](../src/module/communication.cpp#L31)
- 单 hop 时间：[src/module/communication.cpp](../src/module/communication.cpp#L34)
- 总时间：[src/module/communication.cpp](../src/module/communication.cpp#L38)

### 11.2 MoE Scatter / Gather

MoE scatter 根据 expert 所在设备统计跨设备 token 数：

```text
comm_bytes = num_comm_token * hidden_dim * precision_byte
```

然后区分：

- 节点内通信：`device_ict_bandwidth` / `device_ict_latency`
- 节点间通信：`node_ict_bandwidth` / `node_ict_latency`
- prefill/mixed：节点内和节点间可并行，取较大者
- decode：多节点时主要走节点间带宽，单节点走设备内互连

源码入口：[src/module/communication.cpp](../src/module/communication.cpp#L100)。

## 12. 访存模型：理想带宽 vs Ramulator

很多硬件执行函数都有两条路径：

### 理想带宽模型

直接使用带宽模型：

```text
T_memory = bytes / bandwidth
```

源码入口：[src/hardware/layer_impl.cpp](../src/hardware/layer_impl.cpp#L27)。

### Ramulator 模型

把张量访存转换成 `DRAMRequest`，交给 DRAM 接口运行：

```cpp
DRAMRequest::Ptr dram_request = DRAMRequest::Create(dram_request_type);
dram_request->AddOperand(tensor->getMemoryObject(), pim_operand_type);
device->run_ramulator(dram_request);
exec_status = device->dram_interface->getExecStatus();
```

源码入口：[src/hardware/layer_impl.cpp](../src/hardware/layer_impl.cpp#L9)。

这两条路径的区别是：

| 模式 | 速度 | 精度 | 适用 |
|---|---|---|---|
| `use_ramulator = false` | 快 | 高层估计 | 大量设计空间搜索 |
| `use_ramulator = true` | 慢 | 更细粒度 DRAM 行为 | 研究内存命令、PIM、DRAM 时序 |

## 13. 时间、能耗和利用率如何累计

每个硬件执行函数返回 `ExecStatus`：

| 字段 | 含义 |
|---|---|
| `total_duration` | 本模块最终耗时 |
| `compute_duration` | 计算时间 |
| `memory_duration` | 访存时间 |
| `flops` | 本模块 FLOPs |
| `memory_size` | 本模块访存量 |
| `compute_util` | 相对峰值算力利用率 |
| `memory_util` | 相对峰值带宽利用率 |
| `act_count/read_count/write_count` | DRAM 命令统计 |

定义代码：[src/module/status.h](../src/module/status.h#L13)。

`TopModuleGraph::set_pop_status()` 在模块执行结束后把 `ExecStatus` 加到设备时间线上：

- 普通执行：`status.device_time += exec_status.total_duration`
- 异构并行：GPU 走 `high_time`，LOGIC/PIM 走 `low_time`
- 能耗：用 DRAM command count 乘以对应能耗参数

源码：[src/module/module_graph.cpp](../src/module/module_graph.cpp#L223)。

这解释了为什么同一个 batch 的时间不是简单地把所有模块串起来。启用 `parallel_execution` 时，GPU 和 PIM/Logic 可以分别累积时间，再在 split/merge/sync 点对齐。

## 14. 每轮仿真的闭环

最终闭环在 `Cluster::runIteration()`：

```text
for iter:
  metadata = scheduler->setMetadata()
  cluster->run(metadata)
  time = device_0.status.device_time
  energy = getTotalEnergy()
  scheduler->updateScheduler(time)
  write CSV row
```

源码：

- 迭代入口：[src/hardware/cluster.cpp](../src/hardware/cluster.cpp#L424)
- mixed 执行：[src/hardware/cluster.cpp](../src/hardware/cluster.cpp#L459)
- sum/gen split 执行：[src/hardware/cluster.cpp](../src/hardware/cluster.cpp#L527)
- 单轮模块图执行：[src/hardware/cluster.cpp](../src/hardware/cluster.cpp#L1084)

因此，LLMSimulator 的“仿真一个 LLM”不是一次性计算完整输出，而是反复执行：

```text
本轮 token 状态 -> 模块图执行 -> 硬件代价估算 -> 序列状态推进
```

直到所有请求完成。

## 15. 一张总表：数学对象到代码的对应关系

| 数学/系统对象 | 仿真含义 | 主要代码 |
|---|---|---|
| `bytes = precision_byte * product(shape)` | 张量大小 | `src/module/tensor.cpp` |
| `2*m*k*n` | Linear FLOPs | `src/hardware/linear_impl.cpp` |
| `max(FLOPs/C, bytes/B)` | Roofline 时间 | `src/hardware/*_impl.cpp` |
| `QK^T` | Attention scoring | `src/hardware/attention_sum_impl.cpp`、`attention_gen_impl.cpp` |
| `softmax` | scale/mask/softmax 近似 FLOPs | `src/hardware/attention_*_impl.cpp` |
| `prob V` | Attention context | `src/hardware/attention_*_impl.cpp` |
| `CKV` | MLA latent KV cache | `src/module/attention.cpp`、`src/hardware/attention_*_impl.cpp` |
| `score = (QW)CKV^T` | absorb MLA | `src/module/layer.cpp`、`attention_gen_impl.cpp` |
| `TopK(gate)` | MoE token 路由 | `src/module/route.cpp` |
| `m_expert` | 每个 expert 的 token 数 | `src/module/expert.cpp`、`src/module/route.cpp` |
| ring all-reduce | TP 通信时间 | `src/module/communication.cpp` |
| DRAM request | 细粒度内存仿真 | `src/hardware/layer_impl.cpp`、`src/dram/` |
| `ExecStatus` | 单模块硬件统计 | `src/module/status.h` |
| `device_time` | 单卡时间线 | `src/module/module_graph.cpp` |

## 16. 最小心智模型

如果只记一件事，可以记这个：

```text
LLMSimulator = LLM 模块图 + token 调度状态 + Roofline/DRAM 硬件模型
```

每个模块都在回答同一个问题：

```text
给定本轮 token 数、序列长度、张量维度、精度和硬件参数，
这个 LLM 子结构会产生多少 FLOPs、多少 bytes、多少通信，
最终让 GPU/LOGIC/PIM 的时间线前进多少 ns？
```

