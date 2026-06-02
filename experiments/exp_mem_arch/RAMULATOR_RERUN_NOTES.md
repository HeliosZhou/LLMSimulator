# Ramulator 重跑问题说明

## 背景

本目录 `experiments/exp_mem_arch` 的实验目标是比较同一模型、同一 batch/sequence、同一内存架构下，启用和不启用 Ramulator 时的性能差异。实验矩阵为：

- 内存架构：HBM3E、GDDR6、DDR5
- Reordering：on、off
- Ramulator：on、off
- Batch/GPU：32、64、128、256
- Seq length：2048、4096、8192

总计 3 x 2 x 2 x 4 x 3 = 144 个实验。

## 原先 Ramulator 层级仿真的问题

原先不能稳定完成 GDDR6/DDR5 的 Ramulator 层级仿真，主要问题有以下几个。

1. DDR5 的地址空间配置不足。

   旧 DDR5 配置可寻址空间和实验中设定的 192GB 容量不匹配。实验会按 192GB 级别设置模型权重、KV cache 和激活占用，但 Ramulator 侧 DDR5 组织只有约 2GB 级别，导致地址映射越界或仿真不可信。

2. GDDR6 复用了不匹配的地址层级。

   HBM3E、DDR5、GDDR6 的 Ramulator 地址向量层级不同。旧逻辑把 GDDR6 当成 DDR5/HBM 风格的多层地址来生成，导致 `AddrVec` 和 Ramulator GDDR6 模型期望不一致。

3. DDR5/GDDR6 的请求筛选逻辑不适用。

   Read/Write 请求生成原先依赖 `addr_vec[0] == 0` 来筛选有效通道。DDR5 只有一个 channel 时，这个条件会导致采样/筛选语义不正确；GDDR6/DDR5 请求量又很大，直接全量送入 Ramulator 会导致运行时间过长。

4. 短运行可能只输出 CSV 表头。

   `runIterationMixed` 末尾没有保证把最后的 `stat_list` flush 到 CSV。某些短实验虽然完成了，但 CSV 可能只有表头，没有数据行。

5. Ramulator 详细计数没有完整写入 CSV。

   Ramulator 的 read/write/act/ref 等计数已经累积到设备 `StatusBoard`，但原先没有完整拷贝到导出的 `Stat` 中，导致启用 Ramulator 后 CSV 里看不到应有的层级仿真统计。

## 本次修改内容

为使 144 个实验可以正常运行并且结果可比较，本次做了以下修改。

1. 增加 Ramulator 采样参数。

   在 `eval/test.cpp`、`SystemConfig`、`PIMHWConfig`、`DRAMInterface` 链路中加入 `system.ramulator_sample_stride`。当前实验设置为：

   - HBM3E：`ramulator_sample_stride=1`
   - GDDR6：`ramulator_sample_stride=4096`
   - DDR5：`ramulator_sample_stride=4096`

   GDDR6/DDR5 每 4096 个有效 DRAM bundle 取 1 个代表请求送入 Ramulator，再按实际采样比例放大时延和命令计数。这样保留内存组织和访问模式，同时避免全量请求使仿真长时间无法完成。

2. 修正 Read/Write 请求采样。

   `src/dram/pimkernel/Read.cpp` 和 `src/dram/pimkernel/Write.cpp` 现在会统计有效命令数、采样命令数，并设置 `PIMRequest.sample_scale`。`src/dram/dram_interface.cpp` 根据该比例放大 `memory_duration` 和 Ramulator command counters。

3. 增加 DDR5/GDDR6 的内存组织配置。

   `src/dram/memory_config.h` 中加入：

   - `ddr5_192GB`
   - `gddr6_192GB`

   对应 build 目录下的 Ramulator YAML：

   - `build/dram_config_HBM3E_192GB.yaml`
   - `build/dram_config_GDDR6.yaml`
   - `build/dram_config_DDR5.yaml`

4. 修正不同内存架构的地址向量层级。

   `src/dram/mmap_controller.cpp` 根据 `ramulator_addr_levels` 生成不同长度的 Ramulator 地址向量：

   - GDDR6：5 层，`channel, bankgroup, bank, row, column`
   - DDR5：6 层，`channel, rank, bankgroup, bank, row, column`
   - HBM/HBM3E：7 层，`channel, pseudochannel, rank, bankgroup, bank, row, column`

5. 按内存带宽选择对应 Ramulator 配置。

   `src/hardware/device.cpp` 根据 `memory_bandwidth` 选择 DRAM 配置：

   - 大于等于 4 TB/s：HBM3E
   - 大于等于 256 GB/s：GDDR6
   - 低于 256 GB/s：DDR5

6. 修正 CSV 输出和 Ramulator 统计导出。

   `src/hardware/cluster.cpp` 已保证最后一批 `stat_list` 会导出，并把设备 `StatusBoard` 中的 Ramulator counters 写入 CSV 的 `Stat` 字段。

7. 修正实验脚本输出隔离。

   `run_experiments.sh` 每个实验现在写入独立临时目录，再把本次产生的 CSV 重命名为标准结果名，避免误把上一轮 CSV 当成本轮输出。

   新增 `run_missing_parallel.sh` 用于并发补跑缺失项。每个 worker 使用独立工作目录和独立 `config.yaml`，避免并发时共享 `build/config.yaml` 互相覆盖。

## 本次重跑结果

当前目录下已完成全量 144 个标准结果：

- 结果目录：`experiments/exp_mem_arch/data`
- HBM3E：48 个
- GDDR6：48 个
- DDR5：48 个
- 缺失组合：0
- 空 CSV 或只有表头的 CSV：0
- `[FAIL]` / `[WARN]` 日志：无

旧数据已备份到：

- `experiments/exp_mem_arch/data_backup_before_rerun_20260601_181116`

本次运行日志：

- `experiments/exp_mem_arch/rerun_144_20260601_181116.log`
- `experiments/exp_mem_arch/rerun_missing_parallel_20260601_183205.log`

## Ramulator ON/OFF 的控制变量检查

当前实验脚本中，Ramulator ON 和 Ramulator OFF 使用相同的硬件与负载配置。对每个固定的 `memory_type + reordering + batch + seq_len` 组合，ON/OFF 配置只改变：

- `system.optimization.use_ramulator`

保持一致的关键配置包括：

- `model.model_name = deepseekV3`
- `system.gpu_gen = B200`
- `system.num_node = 4`
- `system.num_device = 8`
- `system.processor_type = GPU`
- `system.memory_bandwidth`
- `system.memory_capacity`
- `system.ramulator_sample_stride`
- `system.distribution.expert_tensor_degree = 1`
- `system.distribution.none_expert_tensor_degree = 1`
- `system.optimization.use_absorb`
- `system.optimization.compressed_kv = true`
- `system.optimization.use_flash_mla = true`
- `system.optimization.use_flash_attention = true`
- `system.optimization.reuse_kv_cache = true`
- `system.optimization.parallel_execution = false`
- `system.optimization.hetero_subbatch = false`
- `system.optimization.disagg_system = false`
- `system.optimization.prefill_mode = false`
- `system.optimization.decode_mode = true`
- `serving.max_batch_size = batch_per_gpu x 4 x 8`
- `simulation.input_len`
- `simulation.output_len = 2`
- `simulation.precision_byte = 1`
- `simulation.iter = 3`

我额外做了两层检查：

1. 按脚本枚举 72 对 ON/OFF 配置，除 `system.optimization.use_ramulator` 外没有其他字段差异。
2. 按实际 CSV 结果检查 72 对 ON/OFF 输出，`memory_capacity`、`activation_size`、`weight_size`、`kv_cache_size`、`total_memory_used`、`memory_utilization`、`batchsize`、`seqlen` 都一致。

因此，在当前 144 个实验中，启用 Ramulator 与不启用 Ramulator 的对比满足“同一硬件、同一负载、唯一变量为 Ramulator 是否启动”的要求。
