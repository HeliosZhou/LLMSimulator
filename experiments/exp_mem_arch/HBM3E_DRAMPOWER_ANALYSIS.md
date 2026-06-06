# HBM3E DRAMPower 实验分析

本文总结 `data_drampower/` 目录下 48 组启用 DRAMPower 的 HBM3E 实验结果。

## 实验范围

- Reorder：on/off
- Sequence length：2048/4096/8192
- Batch per GPU：32/64/128/256
- Ramulator 层级仿真：on/off
- `data/summary_hbm3e.csv` 中 `drampower=on` 的行数：48

## 能耗模型实现口径

当前实验中的 `DRAMPower` 不是直接调用 DRAMPower CLI 读取完整 command
trace，也不是使用官方 HBM3E memspec。实现路径是：

```text
LLMSimulator/Ramulator 命令计数
  -> HBM3EAdapter
  -> drampower_act/read/write/ref/background/total_energy
```

也就是说，它更准确地说是一个 **HBM3E DRAMPower-style adapter**。代码路径：

- `src/module/module_graph.cpp`：当 `use_drampower=true` 时，把每个算子的
  `ExecStatus` 转成 `CommandCounters` 并调用 `HBM3EAdapter::calculate()`。
- `src/dram/ramulator2/src/drampower/hbm3e_adapter.{h,cpp}`：保存 HBM3E
  电流/时序假设，并按 DRAMPower/IDD 风格公式计算能耗。
- `src/hardware/cluster.cpp`：把 `drampower_*` 字段写入 CSV。

输入到 adapter 的命令计数包括：

```text
act_count, read_count, write_count,
all_act_count, all_read_count, all_write_count,
ref_count
```

这些计数的来源取决于是否启用 Ramulator：

- `Ramulator=on`：来自 Ramulator 实际发出的 DRAM command 统计。
- `Ramulator=off`：来自 ideal memory model，根据 tensor size、memory
  granularity、cube/channel/column 组织估算 ACT/READ/WRITE count。

### Adapter 参数

`HBM3EAdapter` 的默认参数如下：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `vdd` | 1.2 V | 供电电压 |
| `idd0` | 56 mA | ACT/precharge 相关工作电流近似 |
| `idd2n` | 33 mA | standby/background 电流 |
| `idd4r` | 157 mA | READ 工作电流近似 |
| `idd4w` | 135 mA | WRITE 工作电流近似 |
| `idd5` | 118 mA | REF 工作电流近似 |
| `tck_ns` | 0.5 ns | 时钟周期 |
| `trcd_cycles` | 28 | RCD 周期数 |
| `tras_cycles` | 68 | RAS 周期数 |
| `trp_cycles` | 28 | RP 周期数 |
| `trfc_cycles` | 400 | RFC 周期数 |
| `burst_length_cycles` | 2 | burst 时间 |
| `command_parallelism` | 128 | 并行放大系数 |
| `fallback_act_nj` | 0.909 nJ | ACT 单命令能耗下限 |
| `fallback_read_nj` | 0.891 nJ | READ 单命令能耗下限 |
| `fallback_write_nj` | 0.891 nJ | WRITE 单命令能耗下限 |
| `fallback_ref_nj` | 0.909 nJ | REF 单命令能耗下限 |

### 计算公式

Adapter 对每类命令先计算单命令能耗，再乘以 `command_parallelism` 和命令计数。
单位为 nJ。

ACT：

```text
act_time = (tRCD + tRAS + tRP) * tCK
per_act = VDD * (IDD0 - IDD2N) * act_time
act_energy = max(per_act, fallback_act_nJ)
             * command_parallelism
             * act_count
```

READ：

```text
burst_time = burst_length_cycles * tCK
per_read = VDD * (IDD4R - IDD2N) * burst_time
read_energy = max(per_read, fallback_read_nJ)
              * command_parallelism
              * read_count
```

WRITE：

```text
per_write = VDD * (IDD4W - IDD2N) * burst_time
write_energy = max(per_write, fallback_write_nJ)
               * command_parallelism
               * write_count
```

REF：

```text
ref_time = tRFC * tCK
per_ref = VDD * (IDD5 - IDD2N) * ref_time
ref_energy = max(per_ref, fallback_ref_nJ)
             * command_parallelism
             * ref_count
```

Background：

```text
background_energy = VDD * IDD2N * background_time
                    * command_parallelism
```

这里的 `background_time` 在当前实现中来自每个执行段的 `total_duration`，而不是
`memory_duration`。因此它表示 DRAM 被计入 standby/background 的执行时间基准，
不能直接解释为 DRAM service time。

总 DRAMPower 能耗为：

```text
drampower_total_energy =
  drampower_act_energy
  + drampower_read_energy
  + drampower_write_energy
  + drampower_all_act_energy
  + drampower_all_read_energy
  + drampower_all_write_energy
  + drampower_ref_energy
  + drampower_background_energy
```

旧版 `data_drampower/` 48 组结果中 `all_*` 和 `ref_count` 为 0，因此主要组成是
ACT、READ、WRITE 和 Background。`HBM3EAdapter` 本身有 REF 计算公式，但只有在
上游 `ExecStatus.ref_count` 非 0 时才会产生 `drampower_ref_energy`。这批旧结果
里的 `Ramulator=on` 数据不适合分析 REF，因为当时写入 CSV 的计数没有完整使用
controller 级别的 issued command counter。

### 与原始能耗字段的区别

CSV 中同时有两套 DRAM 能耗字段：

- `act_energy/read_energy/write_energy/ref_energy/background_energy`：旧的
  fixed event-energy model，主要继承 FGDRAM 风格常数。
- `drampower_act_energy/drampower_read_energy/...`：当前 HBM3EAdapter 计算
  出来的 DRAMPower-style 能耗。

`mac_energy` 是计算能耗，按 `FLOPs * 0.46 pJ/MAC` 的固定系数估算，不属于
`drampower_total_energy`。因此分析 DRAM 能耗时应使用 `drampower_*`；分析
总系统能耗时，需要明确是否额外把 `mac_energy` 加进去。

### 当前模型的局限

这个 adapter 是轻量估算模型，适合做同一实验矩阵内的相对比较，但不应等同于
完整 DRAMPower HBM3E trace-level 仿真。它没有建模：

- bank state / row-buffer hit 和 miss 的精细状态转移；
- precharge、activate、read、write 的完整命令时序互相约束；
- 数据翻转率、DBI/ECC、I/O toggle 相关差异；
- 官方 HBM3E memspec 中的完整 IDD/timing 项；
- 温度、电压频率 scaling、rank/channel 级 idle state 切换。

因此本文中的 `DRAMPower` 数值应理解为：

```text
基于 LLMSimulator/Ramulator 命令计数的 HBM3E IDD-style DRAM 能耗估算
```

而不是实际硬件测量值。

## Step 与 Token 口径

每一行 `t2t` 表示一次 batch decode step。也就是说，这是当前 active
batch 中每条序列各生成 1 个 token 的整批 step 能耗，不是单条序列的单
token 能耗。平均单 token 能耗按如下方式计算：

```text
drampower_total_energy / numtoken
```

## Ramulator-on REF 重跑结果

为了统一使用 Ramulator 中实际发出的 DRAM command 计数，已重新运行 24 个
`Ramulator=on, DRAMPower=on` 实验。新结果不再混用上层 LLMSimulator ideal
memory model 的 ACT/READ/WRITE 估算；ACT、READ、WRITE、REF 都来自
Ramulator controller 的 issued command counter。

新的数据路径：

- 原始逐 step CSV：`data_drampower_ramulcmd_ref/result_hbm3e_b{B}_l{L}_reorder_{on|off}_ramul_on.csv`
- 聚合明细 CSV：`data/energy_breakdown_ramulator_on_drampower_ref.csv`
- Markdown 明细：`RAMULATOR_ON_DRAMPOWER_REF_ENERGY.md`

对应代码路径：

- `src/dram/ramulator2/src/dram_controller/controller.h`：
  `record_issued_command()` 把 Ramulator 实际发出的 ACT/RD/WR/refreshing
  command 累计到 `kACT/kREAD/kWRITE/kREF`。
- `src/dram/ramulator2/src/dram_controller/impl/refresh/all_bank_refresh.cpp`：
  `AllBankRefresh` 每到 `nREFI` 对每个 rank 插入 all-bank-refresh request。
- `src/dram/dram_interface.cpp`：`updateStatus()` 读取
  `memory_system->get_issued_dram_cmd()` 的增量，写入
  `ExecStatus.{act,read,write,ref}_count`。
- `src/module/module_graph.cpp`：把 `ExecStatus` 转成 `CommandCounters` 后调用
  `HBM3EAdapter::calculate()`。

24 组新实验中 `ref_count` 全部非 0，平均 `ref_count` 范围是
72,401 到 40,805,223 / step。分组平均如下：

| Reorder | Avg DRAM J/step | Avg J/token | ACT | READ | WRITE | REF | Background |
|---|---:|---:|---:|---:|---:|---:|---:|
| on | 62.35 | 0.0236 | 11.3% | 82.4% | 2.8% | 0.6% | 2.8% |
| off | 2625.18 | 0.6910 | 7.8% | 50.3% | 37.6% | 1.1% | 3.2% |

两个端点示例：

| Reorder | Seq | B/GPU | REF count | REF J/step | DRAM J/step | REF 占比 |
|---|---:|---:|---:|---:|---:|---:|
| on | 2048 | 32 | 72,401 | 0.19 | 40.91 | 0.46% |
| off | 8192 | 256 | 40,805,223 | 106.55 | 9495.81 | 1.12% |

因此，对开启 Ramulator 后的 REF 能耗分析，应使用这 24 个重跑结果，而不是旧的
`data_drampower/` 中 `ref_count=0` 的 Ramulator-on CSV。

## 总体范围

| 指标 | 最小值 | 平均值 | 最大值 |
|---|---:|---:|---:|
| 延迟 (ms) | 8.10 | 254.10 | 1883.74 |
| DRAMPower 总能耗 (J/step) | 38.06 | 1309.56 | 9389.26 |
| DRAMPower 背景/静态能耗 (J/step) | 0.78 | 39.29 | 301.47 |
| 平均单 token DRAMPower 能耗 (J/token) | 0.0080 | 0.3481 | 1.1779 |
| memory_duration (ms) | 311.81 | 265850.12 | 2702974.28 |
| background_time (ms) | 154.74 | 7751.27 | 59476.13 |

## 能耗构成

各项在 `drampower_total_energy` 中的占比：

| 组成项 | 最小占比 | 平均占比 | 最大占比 |
|---|---:|---:|---:|
| ACT | 5.50% | 7.57% | 12.65% |
| READ | 49.90% | 68.96% | 91.22% |
| WRITE | 1.09% | 20.75% | 40.31% |
| REF | 0.00% | 0.00% | 0.00% |
| 背景/静态 | 1.92% | 2.72% | 3.21% |

这里 REF 为 0 不是因为 adapter 缺少 REF 能耗模型，而是因为旧版 48 组实验的
`ref_count` 全部为 0；因此 `drampower_ref_energy_nJ` 也全部为 0。开启
Ramulator 且需要 REF 能耗时，应使用上一节的 24 组重跑结果。

开启 reorder 时，READ 能耗占主导。关闭 reorder 后，WRITE 命令数量大幅上升，
WRITE 能耗占比也显著提高。

完整逐实验明细见 `ENERGY_BREAKDOWN_ALL_EXPERIMENTS.md` 和
`data/energy_breakdown_all_experiments.csv`。其中 CSV 保留每个实验的
ACT、READ、WRITE、all-ACT、all-READ、all-WRITE、REF、Background 和 MAC
的 J/step、J/token 以及占比；Markdown 简表分成两个口径：

- `DRAM 内部构成`：分母是 `drampower_total_energy`，不包含 MAC。
- `包含 MAC 后的构成`：分母是 `drampower_total_energy + mac_energy`。

## 分组平均值

| Reorder | Ramulator | 延迟 ms | 能耗 J/step | J/token | 背景占比 | READ 占比 | WRITE 占比 | ACT 占比 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| on | on | 22.86 | 61.93 | 0.0235 | 2.86% | 82.93% | 2.85% | 11.36% |
| on | off | 19.31 | 60.40 | 0.0224 | 2.02% | 89.54% | 2.91% | 5.54% |
| off | on | 524.99 | 2595.93 | 0.6834 | 3.20% | 50.88% | 38.03% | 7.89% |
| off | off | 449.25 | 2519.99 | 0.6629 | 2.80% | 52.48% | 39.22% | 5.50% |

## Ramulator 层级仿真的影响

在相同 reorder、sequence、batch 设置下，开启 Ramulator 层级仿真后的变化：

| 指标 | 最小倍率 | 平均倍率 | 最大倍率 |
|---|---:|---:|---:|
| 延迟，Ramulator on/off | 1.11 | 1.19 | 1.31 |
| DRAMPower 能耗，Ramulator on/off | 0.98 | 1.03 | 1.07 |
| memory_duration，Ramulator on/off | 1.01 | 1.07 | 1.27 |

Ramulator 层级仿真主要拉高延迟和 memory service duration。DRAMPower 总能耗
变化相对温和，因为它主要由命令计数驱动，而 Ramulator on/off 下命令计数差异
没有 reorder on/off 那么大；层级仿真更多体现为调度、排队和等待成本。

## Reorder 的影响

在相同 Ramulator、sequence、batch 设置下，关闭 reorder 的代价非常明显：

| 指标 | 最小倍率 | 平均倍率 | 最大倍率 |
|---|---:|---:|---:|
| 延迟，reorder off/on | 6.47 | 20.10 | 40.75 |
| DRAMPower 能耗，reorder off/on | 8.11 | 34.93 | 78.32 |
| READ count，reorder off/on | 5.11 | 20.77 | 46.34 |
| WRITE count，reorder off/on | 221.26 | 521.64 | 898.83 |

主要原因是关闭 reorder 后命令计数爆炸，尤其是 WRITE 命令。它是这 48 组实验
中能耗增长最主要的来源。

## Sequence 和 Batch 维度的扩展趋势

下面结果对 reorder 和 Ramulator 两个维度取平均：

| Seq | Batch/GPU | 延迟 ms | 能耗 J/step | J/token | READ count | WRITE count | 背景占比 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 32 | 35.86 | 181.88 | 0.1776 | 926814787 | 507123417 | 2.70% |
| 2048 | 64 | 66.23 | 328.33 | 0.1603 | 1579197006 | 1014179167 | 2.72% |
| 2048 | 128 | 126.69 | 620.85 | 0.1516 | 2883961996 | 2028291481 | 2.71% |
| 2048 | 256 | 247.72 | 1206.18 | 0.1472 | 5493492456 | 4056517870 | 2.69% |
| 4096 | 32 | 62.85 | 326.91 | 0.3192 | 1572672067 | 1010439897 | 2.72% |
| 4096 | 64 | 120.21 | 618.43 | 0.3020 | 2870911566 | 2020812127 | 2.73% |
| 4096 | 128 | 234.66 | 1201.07 | 0.2932 | 5467391117 | 4041557401 | 2.72% |
| 4096 | 256 | 463.68 | 2366.60 | 0.2889 | 10660350696 | 8083049710 | 2.71% |
| 8192 | 32 | 116.82 | 616.99 | 0.6025 | 2864386627 | 2017072857 | 2.73% |
| 8192 | 64 | 228.20 | 1198.67 | 0.5853 | 5454340686 | 4034078047 | 2.74% |
| 8192 | 128 | 450.66 | 2361.47 | 0.5765 | 10634249357 | 8068089241 | 2.73% |
| 8192 | 256 | 895.61 | 4687.36 | 0.5722 | 20994067176 | 16136113390 | 2.72% |

Sequence length 增大会增加 attention 相关的访存流量，因此延迟和能耗都会上升。
Batch 增大会提高整个 step 的总能耗，但在相同 sequence length 下，平均单 token
能耗通常会下降，因为固定的 step 开销被更多 token 摊薄。

## 关键结论

1. DRAMPower 总能耗通常由 READ 主导；但关闭 reorder 后，WRITE 会成为非常重要的能耗组成。
2. 背景/静态能耗稳定存在，但占比不高，平均约为 2.72%。
3. Ramulator 层级仿真平均使延迟增加约 19%，但 DRAMPower 能耗平均只增加约 3%。
4. Reorder 是决定能耗的关键因素。关闭 reorder 后，DRAMPower 能耗平均约为开启 reorder 的 34.93 倍。
5. 若需要平均单生成 token 的 DRAM 能耗，应使用 `drampower_total_energy / numtoken`。CSV 中的能耗字段本身是整批 decode step 的能耗。
6. 当前 DRAMPower 字段只覆盖 DRAM 侧能耗；`mac_energy` 是额外的计算能耗估算，不包含在 `drampower_total_energy` 中。
