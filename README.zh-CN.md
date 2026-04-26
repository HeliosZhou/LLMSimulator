# LLMSimulator

## 概述
LLMSimulator 是一个基于 C++ 的周期级精确模拟器，面向大语言模型（Large Language Models, LLM）的图执行过程进行建模。该模拟器支持当前主流的大语言模型，例如 DeepSeek、Llama、Mixtral 等。除了多头注意力（Multi-Head Attention, MHA）机制外，LLMSimulator 还支持分组查询注意力（Grouped-Query Attention, GQA）、多查询注意力（Multi-Query Attention, MQA）以及多头潜在注意力（Multi-head Latent Attention, MLA）。此外，LLMSimulator 还具备对专家混合模型（Mixture of Experts, MoE）的模拟能力。

LLMSimulator 集成了经过修改的 [Ramulator 2.0](https://github.com/CMU-SAFARI/ramulator2)，用于进行更细粒度的内存系统建模。它可以评估多种 GPU 代际架构，例如 H100、B100 和 B200，同时也支持多种 PIM（Processing-in-Memory）架构，包括 bank-level PIM、bank-group-level PIM 和 Logic-PIM。

主要特性：
- 支持灵活配置输入长度、输出长度、批大小、请求注入速率以及多节点硬件配置
- 支持对多种内存系统下的能耗与性能指标进行建模

## 前置依赖
- 编译器：`g++` 11.4.0
- `cmake`
- `clang++`

LLMSimulator 已在以下系统环境下完成测试。

## 快速开始
### 构建 LLMSimulator
1. 克隆仓库
```bash
$ git clone https://github.com/scale-snu/LLMSimulator.git
$ cd LLMSimulator
$ git submodule update --init --recursive
```

2. 应用补丁
```bash
$ cd src/dram/ramulator2
$ git apply ../../../patch/ramulator2_pim.patch
$ cd ../../../
```

3. 编译可执行文件
```bash
$ mkdir build && cd build
$ cmake ..
$ make -j
```

### 运行方法
LLMSimulator 提供配置文件 `config.yaml`，你可以根据自己的需求进行修改。修改并保存 `config.yaml` 后，可以使用以下命令运行：

```bash
$ ./run > test.log
```

## 联系方式
Sungmin Yun  sungmin.yun@snu.ac.kr

Kwanhee Kyung  kwanhee.kyung@scale.snu.ac.kr

Juhwan Cho  juhwan.cho@snu.ac.kr

## 说明
本模拟器构建于 MICRO 2024 论文 “Duplex: A Device for Large Language Models with Mixture of Experts, Grouped Query Attention, and Continuous Batching” 中介绍的模拟器之上。
