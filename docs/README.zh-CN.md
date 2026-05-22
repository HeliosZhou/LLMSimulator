# LLMSimulator 文档导航

<div align="center">
  <h2>从 LLM 数学结构到 GPU / PIM 系统代价的仿真链路</h2>
  <p>
    这个目录里的文档按“先理解是什么，再理解怎么跑，最后理解公式与代码”的顺序组织。
  </p>
</div>

<table>
  <tr>
    <td width="33%" valign="top">
      <h3>先看全局</h3>
      <p>如果你想快速知道这个工程解决什么问题、代码目录怎么分层，先读这里。</p>
      <p><a href="./PROJECT_OVERVIEW.zh-CN.md"><b>工程说明</b></a></p>
    </td>
    <td width="33%" valign="top">
      <h3>再看机理</h3>
      <p>如果你关心 LLM 是如何被拆成公式、张量、FLOPs、访存和硬件时间的，读这里。</p>
      <p><a href="./LLM_SIMULATION_MATH.zh-CN.md"><b>数学机理与代码对照</b></a></p>
    </td>
    <td width="33%" valign="top">
      <h3>最后跑实验</h3>
      <p>如果你已经理解仿真目标，想编译运行、改配置和看 CSV 输出，读这里。</p>
      <p><a href="./RUN_GUIDE.zh-CN.md"><b>运行指南</b></a></p>
    </td>
  </tr>
</table>

## 推荐阅读路径

1. [工程说明](./PROJECT_OVERVIEW.zh-CN.md)：明确 LLMSimulator 不是数值推理框架，而是 LLM 推理系统模拟器。
2. [建模与运行机制说明](./SIMULATION_ARCHITECTURE.zh-CN.md)：理解从 `ModelConfig` 到 `Cluster::runIteration()` 的整体流程。
3. [数学机理与代码对照](./LLM_SIMULATION_MATH.zh-CN.md)：逐项理解 Linear、Attention、MLA、MoE、通信、显存与时间统计如何计算。
4. [运行指南](./RUN_GUIDE.zh-CN.md)：编译、运行、改配置。
5. [CSV 输出说明](./CSV_OUTPUT_GUIDE.zh-CN.md)：理解实验结果文件中每一列的含义。

<details>
  <summary><b>一句话理解这个仿真器</b></summary>

  LLMSimulator 不计算真实 logits，也不产生文本；它把一次 LLM 推理请求拆成模块图，再用张量形状、FLOPs、访存量、互连带宽、DRAM 行为和调度状态估算每轮 token 的延迟、能耗和资源利用率。
</details>

## 核心源码速查

| 你想理解的问题 | 推荐入口 |
|---|---|
| 配置如何读入 | `eval/test.cpp` |
| 模型参数在哪里定义 | `src/model/model_config.h` |
| LLM 如何展开成 embedding、decoder、lm_head | `src/model/llm.cpp` |
| Transformer block 如何拆模块 | `src/module/decoder.cpp`、`src/module/layer.cpp` |
| Attention / MLA 如何分 sum 和 gen | `src/module/attention.cpp`、`src/module/parallel.cpp` |
| MoE 如何路由、scatter/gather、执行 expert | `src/module/expert.cpp`、`src/module/route.cpp` |
| 模块如何映射到 GPU/LOGIC/PIM 执行函数 | `src/hardware/executor.cpp` |
| Linear / Attention 的 FLOPs 和访存公式在哪里 | `src/hardware/linear_impl.cpp`、`src/hardware/attention_*_impl.cpp` |
| 每轮 token 仿真如何推进 | `src/hardware/cluster.cpp`、`src/scheduler/scheduler.cpp` |

