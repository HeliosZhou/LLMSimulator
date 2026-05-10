# LLMSimulator 运行指南

## 1. 这份文档解决什么问题

这份文档专门回答一个问题：`LLMSimulator` 应该怎么编译、怎么运行、怎么改配置，以及运行结果会输出到哪里。

如果你还不清楚这个工程本身是什么，可以先看：

- [工程说明](PROJECT_OVERVIEW.zh-CN.md)

如果你已经知道它是一个 LLM 系统模拟器，那么这份文档可以直接带你把它跑起来。

## 2. 先说结论：最推荐的运行方式

这个工程最稳妥的使用方式是：

1. 在仓库根目录完成构建
2. 进入 `build/` 目录
3. 修改 `build/config.yaml`
4. 在 `build/` 目录下执行 `./run`

推荐这样做的原因是：

- 可执行文件 `run` 默认生成在 `build/`
- 构建时会自动把 `config.yaml`、`dram_config_HBM3_80GB.yaml`、`dram_config_HBM3E_192GB.yaml` 复制到 `build/`
- 默认输出路径 `../data/` 是相对运行目录解析的，从 `build/` 运行更符合仓库当前配置习惯

一句话说，推荐工作流就是：

```bash
cd build
./run config.yaml
```

## 3. 运行前准备

### 3.1 环境依赖

根据仓库当前说明，至少需要：

- `g++` 11.4.0 或兼容版本
- `cmake`
- `clang++`
- `make`

如果是 Ubuntu / Debian 类系统，通常可以先检查版本：

```bash
g++ --version
cmake --version
clang++ --version
make --version
```

### 3.2 子模块

这个工程依赖 `Ramulator2` 子模块。如果是第一次从远程仓库拉代码，需要初始化子模块：

```bash
git submodule update --init --recursive
```

如果你当前这个仓库已经能看到 `src/dram/ramulator2/` 目录，并且之前已经构建成功过，那这一步一般就不用重复做。

### 3.3 Ramulator2 补丁

仓库当前的构建流程假设你会给 `src/dram/ramulator2` 打上一个补丁：

```bash
cd src/dram/ramulator2
git apply ../../../patch/ramulator2_pim.patch
cd ../../../
```

这一步通常只需要做一次，适用于“刚 clone 下来的干净仓库”。

如果你已经打过补丁，再执行可能会失败或提示重复应用。这种情况下不要反复强打，先看当前子模块是不是已经处于可编译状态。

## 4. 如何编译

### 4.1 从零开始编译

在仓库根目录执行：

```bash
mkdir -p build
cd build
cmake ..
make -j
```

编译成功后，你通常会看到这些关键文件出现在 `build/`：

- `run`
- `libllm_system.so`
- `config.yaml`
- `dram_config_HBM3_80GB.yaml`
- `dram_config_HBM3E_192GB.yaml`

其中：

- `run` 是主程序
- `config.yaml` 是运行配置
- 两个 `dram_config_*.yaml` 是 DRAM 建模配置文件

### 4.2 如果只是改配置，不需要重新编译

这一点很重要。

如果你只是修改：

- `config.yaml`
- `dram_config_HBM3_80GB.yaml`
- `dram_config_HBM3E_192GB.yaml`

那么通常不需要重新执行 `cmake` 或 `make`，直接重新运行 `./run` 即可。

只有在你修改了 C++ 源码，比如：

- `src/model/*`
- `src/module/*`
- `src/hardware/*`
- `src/scheduler/*`
- `src/dram/*`

这类文件时，才需要重新编译。

## 5. 第一次运行前建议先做的事

### 5.1 创建输出目录

仓库当前默认配置里有这样一项：

```yaml
log:
  output_directory: ../data/
```

这表示程序会把 CSV 结果输出到“相对当前运行目录的 `../data/`”。

如果你从 `build/` 目录运行，那么这个路径实际对应：

- 仓库根目录下的 `data/`

但是当前仓库里默认并没有这个目录，所以建议你第一次运行前先手动创建：

```bash
mkdir -p data
```

如果你计划导出 gantt，也建议提前创建目标目录。例如默认是：

```yaml
gantt_directory: ./gantt
```

如果你从 `build/` 运行，它对应的是：

- `build/gantt/`

可以提前创建：

```bash
mkdir -p build/gantt
```

### 5.2 明确你要从哪个目录运行

这个工程存在不少“相对路径”：

- `config.yaml`
- 输出目录
- gantt 目录
- DRAM 配置文件

所以最推荐的习惯是始终这样跑：

```bash
cd build
./run config.yaml
```

不要在仓库根目录、别的目录、或者 IDE 的任意工作目录里随手执行 `build/run`，否则相对路径可能会和你预期不一致。

## 6. 最常见的运行方法

### 6.1 使用默认配置运行

如果你已经完成构建，并且位于仓库根目录：

```bash
mkdir -p data
cd build
./run config.yaml
```

这会使用 `build/config.yaml` 作为输入配置。

### 6.2 把日志重定向到文件

如果你不想让输出刷满终端，可以这样运行：

```bash
mkdir -p data
cd build
./run config.yaml > test.log
```

这样：

- 屏幕输出会进入 `build/test.log`
- CSV 结果仍然会按照 `config.yaml` 的设置写到输出目录

### 6.3 使用自定义配置文件运行

如果你有自己的一份配置，比如：

- `build/my_config.yaml`

那么可以这样运行：

```bash
cd build
./run my_config.yaml
```

或者你把配置放在仓库根目录：

```bash
cd build
./run ../config.yaml
```

但实际使用中，我更推荐你直接编辑 `build/config.yaml`，因为这样最不容易被路径问题坑到。

## 7. 默认配置到底会跑什么

仓库当前默认的 `config.yaml` 大致表示：

- 模型：`deepseekV3`
- GPU 代际：`B100`
- 节点数：`1`
- 每节点设备数：`8`
- 处理器类型：`GPU`
- 数据：`synthesis`
- 输入长度：`1024`
- 输出长度：`10`
- 精度：`1 byte`
- 运行模式：`decode_mode: on`
- `use_ramulator: off`
- `compressed_kv: on`
- `use_absorb: on`
- `use_flash_mla: on`
- `use_flash_attention: on`

所以它更像是在模拟：

- 一个 1 节点 8 卡 B100 类系统
- 上面跑 `deepseekV3`
- 关注 decode 阶段的推理成本

## 8. 你最常改的配置有哪些

虽然 `config.yaml` 字段很多，但真正最常改的通常是下面这些。

### 8.1 改模型

```yaml
model:
  model_name: deepseekV3
```

可以切换为源码当前支持的模型，例如：

- `mixtral`
- `openMoE`
- `llama7bMoE`
- `llama3_405B`
- `grok1`
- `deepseekV3`
- `llama4_scout`
- `llama4_maverick`

### 8.2 改硬件规模

```yaml
system:
  gpu_gen: B100
  num_node: 1
  num_device: 8
```

这里通常用来实验：

- 不同 GPU 代际
- 不同卡数
- 不同节点数

### 8.3 改运行阶段

```yaml
system:
  optimization:
    prefill_mode: off
    decode_mode: on
```

注意这里有约束：

- `prefill_mode` 和 `decode_mode` 不能同时为 `on`

如果两个都开，程序会报错。

### 8.4 改输入输出长度

```yaml
simulation:
  input_len: 1024
  output_len: 10
```

这两个是最常见的实验变量。

注意：

- 如果 `decode_mode: on`，那么 `output_len` 必须大于 `1`

否则程序会直接失败。

### 8.5 改 batch 限制

```yaml
serving:
  max_batch_size: 32
  max_process_token: 0
```

这里会影响调度器一次能处理多少请求和 token。

### 8.6 改是否启用 Ramulator

```yaml
system:
  optimization:
    use_ramulator: off
```

如果改成：

```yaml
use_ramulator: on
```

程序会启用更细粒度的内存建模。

这通常意味着：

- 模拟更细
- 运行可能更慢
- 对 DRAM 配置文件更敏感

## 9. 运行后结果会输出到哪里

### 9.1 CSV 结果

默认配置下：

- 你从 `build/` 运行
- `output_directory: ../data/`

那么输出 CSV 会进入：

- 仓库根目录下的 `data/`

输出文件名会比较长，因为它会编码很多实验参数，例如：

- 模型名
- 输入输出长度
- 处理器类型
- 节点数
- 设备数
- TP/DP
- batch 限制
- 是否 Ramulator
- 是否 decode/prefill

这是正常的，不是错误。

### 9.2 终端日志

如果你没有重定向日志，程序会直接在终端打印运行信息。

如果你这样运行：

```bash
./run config.yaml > test.log
```

那么终端输出会进入：

- `build/test.log`

### 9.3 Gantt 输出

如果你把：

```yaml
export_gantt: on
```

打开，那么程序还会导出 gantt 数据到：

```yaml
gantt_directory: ./gantt
```

如果你从 `build/` 运行，这对应：

- `build/gantt/`

## 10. 一个推荐的最小运行流程

如果你只是想先确认“它能跑”，推荐按下面这几步来。

### 第一步：构建

```bash
mkdir -p build
cd build
cmake ..
make -j
```

### 第二步：回到仓库根目录创建输出目录

```bash
cd ..
mkdir -p data
```

### 第三步：运行默认配置

```bash
cd build
./run config.yaml
```

### 第四步：查看结果

运行完成后，去仓库根目录下的 `data/` 看 CSV 文件。

## 11. 一个更实用的实验工作流

如果你准备连续做多组实验，更推荐下面这种方式。

### 11.1 保留一份默认配置

不要直接反复乱改仓库根目录的 `config.yaml`。更推荐：

- 让根目录 `config.yaml` 保持“模板状态”
- 在 `build/` 下复制多份实验配置

例如：

```bash
cd build
cp config.yaml config_decode_1k.yaml
cp config.yaml config_decode_4k.yaml
cp config.yaml config_prefill.yaml
```

然后分别修改这些文件。

### 11.2 每次运行显式指定配置名

例如：

```bash
cd build
./run config_decode_1k.yaml
./run config_decode_4k.yaml
./run config_prefill.yaml
```

这样后续整理实验结果会更清楚。

## 12. 常见问题

### 12.1 报错：找不到输出目录

现象通常是：

- 程序运行了，但没有生成 CSV
- 或者输出路径不对

优先检查：

- 你是不是从 `build/` 目录运行的
- `log.output_directory` 对应的目录是否真实存在

最简单的修复方式：

```bash
cd /path/to/LLMSimulator
mkdir -p data
cd build
./run config.yaml
```

### 12.2 报错：`prefill_mode` 和 `decode_mode` 冲突

这两个模式不能同时开。

正确做法是二选一：

```yaml
prefill_mode: on
decode_mode: off
```

或者：

```yaml
prefill_mode: off
decode_mode: on
```

### 12.3 报错：`decode_mode` 下 `output_len` 不合法

如果你打开了：

```yaml
decode_mode: on
```

那么：

- `output_len` 必须大于 `1`

### 12.4 报错：显存或内存不足

程序内部会检查容量，并可能提示 `Out of Memory`。

可以尝试：

- 减小 `max_batch_size`
- 减小 `input_len`
- 减小 `output_len`
- 调整并行度
- 打开 `mem_cap_limit`
- 或者改用更大的硬件模板，比如从 `H100` 改到 `B100`

### 12.5 打开 `use_ramulator` 后更慢

这是正常现象。

因为：

- `use_ramulator: off` 更像理想化或简化内存建模
- `use_ramulator: on` 会执行更细粒度的 DRAM 模拟

因此后者通常更真实，但也更慢。

### 12.6 改了配置却没生效

最常见原因是：

- 你改的是仓库根目录的 `config.yaml`
- 但实际运行时用的是 `build/config.yaml`

所以一定要确认你运行命令里传的是哪一份配置。

最稳妥的方法是直接显式写：

```bash
cd build
./run config.yaml
```

或者：

```bash
cd build
./run my_config.yaml
```

## 13. 建议的目录与命令习惯

为了减少路径问题，推荐养成下面这个习惯。

### 13.1 构建时

```bash
cd /path/to/LLMSimulator
mkdir -p build
cd build
cmake ..
make -j
```

### 13.2 运行时

```bash
cd /path/to/LLMSimulator
mkdir -p data
cd build
./run config.yaml
```

### 13.3 做实验时

```bash
cd /path/to/LLMSimulator/build
cp config.yaml exp1.yaml
cp config.yaml exp2.yaml
./run exp1.yaml
./run exp2.yaml
```

## 14. 一句话总结

这个工程最推荐的运行方式就是：先在仓库根目录完成构建，再进入 `build/` 目录，用 `./run config.yaml` 执行模拟；第一次运行前记得在仓库根目录创建 `data/` 目录，这样默认 CSV 结果就能正常落盘。
