# 功耗计算口径与实验结果分析

本文分析 `experiments/exp_mem_arch/` 中 HBM3E 内存架构实验的功耗相关字段和结果。当前文档只保留 Ramulator controller 真实命令计数口径下的 24 组结果：

- `data_drampower_ramulcmd_ref/`：24 组重新运行的 `Ramulator=on, DRAMPower=on` 结果，ACT/READ/WRITE/REF 均来自 Ramulator controller 实际发出的 DRAM command counter。
- `data/energy_breakdown_ramulator_on_drampower_ref.csv`：24 组 Ramulator-on 指令口径结果的能耗拆分表。

注意：当前代码已经把 `Background` 的时间基准改成优先使用 Ramulator memory elapsed time，但本文分析的 CSV 是这次改动之前生成的。因此这些结果中的 `background_time` 仍使用生成 CSV 时的口径；若需要新 `Background` 口径的数值，必须重新跑实验。

## 计算链路和指令来源

### 总链路

已有 DRAMPower-style 能耗不是直接调用官方 DRAMPower CLI，也不是读完整 command trace，而是走下面这条链：

```text
LLMSimulator 执行一个 layer / operator
  -> 生成 ExecStatus
  -> ExecStatus 里保存 ACT/READ/WRITE/REF 等命令计数和 background_time
  -> module_graph.cpp 调用 HBM3EAdapter::calculate()
  -> 输出 drampower_act/read/write/ref/background/total_energy
  -> cluster.cpp 写入 CSV
```

关键代码路径：

- `src/module/module_graph.cpp`：`to_drampower_counters()` 把 `ExecStatus` 转成 `CommandCounters`；`set_pop_status()` 在 `use_drampower=true` 时调用 `HBM3EAdapter::calculate()`。
- `src/dram/ramulator2/src/drampower/hbm3e_adapter.{h,cpp}`：保存 HBM3E 电流、时序和 fallback 参数，并按 IDD/DRAMPower-style 公式计算能耗。
- `src/hardware/cluster.cpp`：把 `drampower_*` 字段写入 CSV。
- `src/module/status.h`：定义 `act_count/read_count/write_count/ref_count/background_time/drampower_*` 字段。

### Ramulator=on 时指令从哪里来

`Ramulator=on` 时，ACT/READ/WRITE/REF 来自 Ramulator controller 实际发出的命令，不是上层 tensor size 的估算：

```text
Ramulator scheduler 选中请求
  -> m_dram->issue_command(req_it->command, req_it->addr_vec)
  -> record_issued_command(req_it->command)
  -> controller 内 m_issued_dram_cmd 累加
  -> memory_system->get_issued_dram_cmd() 汇总所有 controller
  -> DRAMInterface::updateStatus() 取增量
  -> ExecStatus.{act,read,write,ref}_count
```

具体对应关系在 `src/dram/ramulator2/src/dram_controller/controller.h`：

- `ACT`、`ACT-1`、`ACT-2` 计入 `kACT`。
- `RD`、`RDA`、`CASRD`、`RD16`、`RD16A` 计入 `kREAD`。
- `WR`、`WRA`、`CASWR`、`WR16`、`WR16A` 计入 `kWRITE`。
- `m_dram->m_command_meta(command).is_refreshing=true` 的命令计入 `kREF`。
- `ALL-ACT/ALL-RD/ALL-WR` 计入对应 `all_*` 字段；本批 HBM3E 结果中这些字段为 0。

命令真正发出的位置：

- `src/dram/ramulator2/src/dram_controller/impl/generic_dram_controller.cpp`
- `src/dram/ramulator2/src/dram_controller/impl/PIM_dram_controller.cpp`

多 controller 汇总位置：

- `src/dram/ramulator2/src/memory_system/impl/PIM_DRAM_system.cpp`

LLMSimulator 取数位置：

- `src/dram/dram_interface.cpp`：读取 `memory_system->get_issued_dram_cmd()`，减去上次累计值，只把本次执行段新增命令写入 `ExecStatus`。

REF 的来源：

- `src/dram/ramulator2/src/dram_controller/impl/refresh/all_bank_refresh.cpp` 中 `AllBankRefresh` 按 `nREFI` 插入 all-bank-refresh request；controller 发出 refresh command 后由 `is_refreshing` 归类到 `ref_count`。

## DRAMPower-style 能耗公式

`HBM3EAdapter` 默认参数：

| 参数 | 值 |
|---|---:|
| `VDD` | 1.2 V |
| `IDD0` | 56 mA |
| `IDD2N` | 33 mA |
| `IDD4R` | 157 mA |
| `IDD4W` | 135 mA |
| `IDD5` | 118 mA |
| `tCK` | 0.5 ns |
| `tRCD/tRAS/tRP` | 28/68/28 cycles |
| `tRFC` | 400 cycles |
| `burst_length` | 2 cycles |
| `command_parallelism` | 128 |
| fallback ACT/READ/WRITE/REF | 0.909/0.891/0.891/0.909 nJ |

单 count 能耗：

| 项 | 当前 nJ/count | 公式 |
|---|---:|---|
| ACT | 219.0336 | `max(VDD*(IDD0-IDD2N)*(tRCD+tRAS+tRP)*tCK, fallback_act) * 128` |
| READ | 114.0480 | `max(VDD*(IDD4R-IDD2N)*burst_time, fallback_read) * 128` |
| WRITE | 114.0480 | `max(VDD*(IDD4W-IDD2N)*burst_time, fallback_write) * 128` |
| REF | 2611.2000 | `max(VDD*(IDD5-IDD2N)*tRFC*tCK, fallback_ref) * 128` |
| Background | 5.0688 nJ/ns | `VDD * IDD2N * background_time * 128` |

总 DRAMPower-style DRAM 能耗：

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

`mac_energy` 不包含在 `drampower_total_energy` 中。若要看“DRAM + 计算”总量，本文使用：

```text
total_plus_mac = drampower_total_energy + mac_energy
```

本文功耗分析使用 `drampower_act_energy/.../drampower_total_energy` 这些 DRAMPower-style 字段。

## Background 口径说明

这 24 组结果中的 `Background` 已经计入 DRAMPower-style 总能耗，公式固定为：

```text
drampower_background_energy = background_time * 5.0688 nJ/ns
```

但 CSV 生成时的 `background_time` 不是严格的 Ramulator bank/channel idle/standby 状态统计。其后代码已经改为：

```text
Ramulator request elapsed memory time -> ExecStatus.background_time
  -> HBM3EAdapter::background_energy_nj()
```

也就是说，当前代码的新口径是“Ramulator elapsed memory time 送入 DRAMPower 计算 background energy”，仍不是按 bank 状态拆分 idle、standby、active standby、refresh standby 的完整状态机统计。

对这 24 组结果做一个估算：如果把 `background_time` 从 CSV 生成时的执行时间基准替换成 `memory_duration`，则平均 background 能耗会明显上升：

| Reorder | CSV Background J/step | 用 memory_duration 估算 J/step | 增长倍数 | CSV DRAM J/step | 重估 DRAM J/step |
|---|---:|---:|---:|---:|---:|
| on | 1.77 | 40.26 | 22.74x | 62.35 | 100.84 |
| off | 83.22 | 2680.58 | 32.21x | 2625.18 | 5222.54 |

这张表不是正式重跑结果，只说明 `Background` 对时间基准非常敏感。正式结论必须用当前代码重新生成 CSV。

## 24 组 Ramulator-on 指令口径结果

24 组来自 `data/energy_breakdown_ramulator_on_drampower_ref.csv`，全部为：

```text
Ramulator=on, DRAMPower=on
```

这批结果是当前最适合分析“Ramulator 指令送入 DRAMPower”的结果，因为 ACT/READ/WRITE/REF 都来自 Ramulator controller issued command counter。

这 24 组的实验矩阵是：

- Sequence length：2048/4096/8192
- Batch per GPU：32/64/128/256
- Reorder：on/off
- Ramulator hierarchy：on
- DRAMPower：on

也就是说，这里的“Ramulator 层级结果”不是再比较 `ramulator=on/off`，而是在 Ramulator controller 真实命令计数口径下，比较 `sequence length`、`batch per GPU` 和 `reorder on/off` 对延时与能耗的影响。

总体均值：

| Reorder | Avg latency ms | Avg memory_duration ms | Avg DRAM J/step | Avg DRAM J/token | Total+MAC J/step | ACT | READ | WRITE | REF | Background | MAC in Total+MAC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| on | 22.86 | 7943.42 | 62.35 | 0.0236 | 194.92 | 11.3% | 82.4% | 2.8% | 0.6% | 2.8% | 61.2% |
| off | 524.99 | 528839.63 | 2625.18 | 0.6910 | 11204.98 | 7.8% | 50.3% | 37.6% | 1.1% | 3.2% | 76.3% |

能耗拆分的平均绝对值：

| Reorder | ACT J/step | READ J/step | WRITE J/step | REF J/step | Background J/step | DRAM J/step |
|---|---:|---:|---:|---:|---:|---:|
| on | 6.83 | 51.40 | 1.92 | 0.42 | 1.77 | 62.35 |
| off | 202.34 | 1304.54 | 1005.83 | 29.25 | 83.22 | 2625.18 |

核心结论：

- Reorder on 的 DRAM 能耗平均只有 62.35 J/step；reorder off 平均为 2625.18 J/step，是 reorder on 的约 42.1 倍。
- 按 token 归一化后，reorder on 平均 0.0236 J/token；reorder off 平均 0.6910 J/token，是 reorder on 的约 29.3 倍。
- Reorder off 的平均延时是 reorder on 的约 23.0 倍，平均 `memory_duration` 是约 66.6 倍。延时放大很明显，但能耗放大更能暴露访问结构问题。
- Reorder on 中 READ 主导，平均 82.4%；WRITE 只有 2.8%。这说明重排后主要瓶颈是读，而不是大量写回。
- Reorder off 中 READ 和 WRITE 都很大，READ 平均 50.3%，WRITE 平均 37.6%。这说明不重排会显著增加写相关 DRAM 能耗。
- 从绝对值看，reorder off 相比 reorder on 的 READ 能耗约 25.4 倍，WRITE 能耗约 523.8 倍。WRITE 的爆炸是 reorder off 总 DRAM 能耗失控的最直接原因。
- REF 在全部 24 组中非零，但不是主导项。Reorder on 平均 0.6%，reorder off 平均 1.1%。
- Background 在当前 CSV 中占 2.8%-3.2%。但如前所述，这里的 Background 依赖 CSV 生成时的 `background_time` 口径，不能作为新 Ramulator elapsed-time 口径的最终结论。

逐组结果如下。`DRAM J/step` 是整批 active batch decode step 的 DRAMPower-style DRAM 能耗，`DRAM J/token` 按当前 step 的 token 数归一化。

| Reorder | Seq | B/GPU | Lat ms | Mem ms | DRAM J/step | DRAM J/token | Total+MAC J/token | ACT | READ | WRITE | REF | BG |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| on | 2048 | 32 | 10.48 | 395.66 | 40.91 | 0.0400 | 0.0640 | 12.4% | 83.0% | 1.3% | 0.5% | 2.8% |
| on | 2048 | 64 | 14.52 | 913.41 | 44.84 | 0.0219 | 0.0459 | 12.6% | 81.7% | 2.3% | 0.6% | 2.9% |
| on | 2048 | 128 | 21.86 | 2906.29 | 51.83 | 0.0127 | 0.0367 | 11.6% | 81.0% | 4.0% | 0.6% | 2.9% |
| on | 2048 | 256 | 36.62 | 10785.57 | 66.21 | 0.0081 | 0.0321 | 10.8% | 79.6% | 6.2% | 0.7% | 2.8% |
| on | 4096 | 32 | 11.02 | 560.05 | 43.29 | 0.0423 | 0.0742 | 12.3% | 83.1% | 1.2% | 0.5% | 2.8% |
| on | 4096 | 64 | 15.20 | 1525.33 | 49.32 | 0.0241 | 0.0560 | 12.0% | 82.5% | 2.1% | 0.6% | 2.9% |
| on | 4096 | 128 | 23.35 | 5292.20 | 60.90 | 0.0149 | 0.0468 | 11.0% | 82.2% | 3.4% | 0.6% | 2.8% |
| on | 4096 | 256 | 39.81 | 20200.89 | 84.41 | 0.0103 | 0.0422 | 10.0% | 81.5% | 4.8% | 0.8% | 2.8% |
| on | 8192 | 32 | 11.68 | 882.01 | 47.72 | 0.0466 | 0.0942 | 11.7% | 83.9% | 1.1% | 0.6% | 2.8% |
| on | 8192 | 64 | 16.84 | 2755.50 | 58.54 | 0.0286 | 0.0762 | 11.5% | 83.3% | 1.8% | 0.7% | 2.9% |
| on | 8192 | 128 | 26.67 | 10071.46 | 79.26 | 0.0193 | 0.0670 | 10.3% | 83.5% | 2.6% | 0.8% | 2.8% |
| on | 8192 | 256 | 46.23 | 39032.69 | 120.94 | 0.0148 | 0.0624 | 9.3% | 83.6% | 3.4% | 0.9% | 2.8% |
| off | 2048 | 32 | 67.77 | 12510.47 | 333.67 | 0.3259 | 1.2927 | 8.2% | 53.1% | 34.5% | 1.0% | 3.1% |
| off | 2048 | 64 | 128.93 | 45447.86 | 630.16 | 0.3077 | 1.2745 | 8.0% | 51.2% | 36.5% | 1.1% | 3.2% |
| off | 2048 | 128 | 250.92 | 173221.56 | 1222.38 | 0.2984 | 1.2652 | 7.8% | 50.3% | 37.7% | 1.1% | 3.2% |
| off | 2048 | 256 | 495.45 | 676414.24 | 2407.55 | 0.2939 | 1.2607 | 7.7% | 49.7% | 38.3% | 1.1% | 3.2% |
| off | 4096 | 32 | 125.46 | 24792.41 | 628.85 | 0.6141 | 2.5316 | 7.9% | 51.3% | 36.6% | 1.1% | 3.2% |
| off | 4096 | 64 | 244.65 | 90633.49 | 1220.90 | 0.5961 | 2.5136 | 7.8% | 50.2% | 37.7% | 1.1% | 3.2% |
| off | 4096 | 128 | 482.35 | 346056.18 | 2403.87 | 0.5869 | 2.5043 | 7.7% | 49.7% | 38.3% | 1.1% | 3.2% |
| off | 4096 | 256 | 958.23 | 1351936.26 | 4770.47 | 0.5823 | 2.4998 | 7.7% | 49.5% | 38.6% | 1.1% | 3.2% |
| off | 8192 | 32 | 241.19 | 49362.20 | 1219.60 | 1.1910 | 5.0097 | 7.8% | 50.3% | 37.7% | 1.1% | 3.2% |
| off | 8192 | 64 | 476.03 | 181004.65 | 2402.36 | 1.1730 | 4.9918 | 7.7% | 49.7% | 38.3% | 1.1% | 3.2% |
| off | 8192 | 128 | 945.13 | 691721.95 | 4766.55 | 1.1637 | 4.9824 | 7.7% | 49.5% | 38.6% | 1.1% | 3.2% |
| off | 8192 | 256 | 1883.74 | 2702974.28 | 9495.81 | 1.1592 | 4.9779 | 7.6% | 49.3% | 38.7% | 1.1% | 3.2% |

极值：

| 指标 | 最小值 | 对应实验 | 最大值 | 对应实验 |
|---|---:|---|---:|---|
| DRAM J/step | 40.91 | reorder=on, seq=2048, B/GPU=32 | 9495.81 | reorder=off, seq=8192, B/GPU=256 |
| DRAM J/token | 0.0081 | reorder=on, seq=2048, B/GPU=256 | 1.1910 | reorder=off, seq=8192, B/GPU=32 |
| Total+MAC J/token | 0.0321 | reorder=on, seq=2048, B/GPU=256 | 5.0097 | reorder=off, seq=8192, B/GPU=32 |
| REF J/step | 0.1891 | reorder=on, seq=2048, B/GPU=32 | 106.55 | reorder=off, seq=8192, B/GPU=256 |
| Background J/step | 1.1580 | reorder=on, seq=2048, B/GPU=32 | 301.47 | reorder=off, seq=8192, B/GPU=256 |

## Reorder 对延时和功耗的影响

逐组比较 `reorder=off / reorder=on`：

| Seq | B/GPU | Lat on ms | Lat off ms | Lat off/on | DRAM on J/step | DRAM off J/step | DRAM off/on | DRAM on J/token | DRAM off J/token | WRITE 占比差 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 32 | 10.48 | 67.77 | 6.47x | 40.91 | 333.67 | 8.16x | 0.0400 | 0.3259 | +33.2 pp |
| 2048 | 64 | 14.52 | 128.93 | 8.88x | 44.84 | 630.16 | 14.05x | 0.0219 | 0.3077 | +34.3 pp |
| 2048 | 128 | 21.86 | 250.92 | 11.48x | 51.83 | 1222.38 | 23.58x | 0.0127 | 0.2984 | +33.7 pp |
| 2048 | 256 | 36.62 | 495.45 | 13.53x | 66.21 | 2407.55 | 36.36x | 0.0081 | 0.2939 | +32.1 pp |
| 4096 | 32 | 11.02 | 125.46 | 11.39x | 43.29 | 628.85 | 14.53x | 0.0423 | 0.6141 | +35.4 pp |
| 4096 | 64 | 15.20 | 244.65 | 16.10x | 49.32 | 1220.90 | 24.76x | 0.0241 | 0.5961 | +35.6 pp |
| 4096 | 128 | 23.35 | 482.35 | 20.65x | 60.90 | 2403.87 | 39.47x | 0.0149 | 0.5869 | +34.9 pp |
| 4096 | 256 | 39.81 | 958.23 | 24.07x | 84.41 | 4770.47 | 56.52x | 0.0103 | 0.5823 | +33.7 pp |
| 8192 | 32 | 11.68 | 241.19 | 20.66x | 47.72 | 1219.60 | 25.56x | 0.0466 | 1.1910 | +36.6 pp |
| 8192 | 64 | 16.84 | 476.03 | 28.26x | 58.54 | 2402.36 | 41.04x | 0.0286 | 1.1730 | +36.5 pp |
| 8192 | 128 | 26.67 | 945.13 | 35.43x | 79.26 | 4766.55 | 60.14x | 0.0193 | 1.1637 | +36.0 pp |
| 8192 | 256 | 46.23 | 1883.74 | 40.75x | 120.94 | 9495.81 | 78.52x | 0.0148 | 1.1592 | +35.3 pp |

结论：

- Reorder 的收益随 sequence length 和 batch size 增大而变大。
- 在 `seq=8192, B/GPU=256` 时，不重排的延时是重排的 40.75 倍，DRAM 能耗是 78.52 倍。
- 能耗倍率通常高于延时倍率，说明 reorder 不只是缩短执行时间，还改变了 DRAM command 结构，尤其是大幅压低 WRITE。
- 每一组成对比较中，reorder off 的 WRITE 占比都比 reorder on 高 32.1-36.6 个百分点。这个差异比 ACT、REF、Background 更能解释总能耗差距。

## Sequence length 对延时和功耗的影响

按 sequence length 看平均值：

| Reorder | Seq | Avg latency ms | Avg memory_duration ms | Avg DRAM J/step | Avg DRAM J/token | READ | WRITE | REF | Background |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| on | 2048 | 20.87 | 3750.23 | 50.95 | 0.0206 | 81.3% | 3.4% | 0.6% | 2.8% |
| on | 4096 | 22.35 | 6894.62 | 59.48 | 0.0229 | 82.3% | 2.9% | 0.6% | 2.8% |
| on | 8192 | 25.36 | 13185.42 | 76.61 | 0.0273 | 83.6% | 2.2% | 0.7% | 2.8% |
| off | 2048 | 235.77 | 226898.54 | 1148.44 | 0.3065 | 51.1% | 36.7% | 1.1% | 3.2% |
| off | 4096 | 452.67 | 453354.58 | 2256.02 | 0.5949 | 50.2% | 37.8% | 1.1% | 3.2% |
| off | 8192 | 886.52 | 906265.77 | 4471.08 | 1.1717 | 49.7% | 38.3% | 1.1% | 3.2% |

随着 seq 增大：

- Reorder on 下，seq 从 2048 增到 8192，平均延时只从 20.87 ms 增到 25.36 ms，DRAM J/step 从 50.95 增到 76.61，J/token 从 0.0206 增到 0.0273。长上下文会增加能耗，但增长较缓。
- Reorder off 下，seq 从 2048 增到 8192，平均延时从 235.77 ms 增到 886.52 ms，DRAM J/step 从 1148.44 增到 4471.08，J/token 从 0.3065 增到 1.1717。能耗接近随 sequence length 成比例放大。
- READ/WRITE 结构随 seq 增大也有变化：reorder on 的 READ 占比从 81.3% 升到 83.6%，WRITE 占比从 3.4% 降到 2.2%；reorder off 的 WRITE 占比从 36.7% 升到 38.3%。这说明长序列下，不重排的写回/搬运压力更重。

## Batch size 对延时和功耗的影响

按 batch size 看平均值：

| Reorder | B/GPU | Avg latency ms | Avg memory_duration ms | Avg DRAM J/step | Avg DRAM J/token | Avg Total+MAC J/token | MAC in Total+MAC |
|---|---:|---:|---:|---:|---:|---:|---:|
| on | 32 | 11.06 | 612.58 | 43.97 | 0.0429 | 0.0775 | 43.7% |
| on | 64 | 15.52 | 1731.41 | 50.90 | 0.0249 | 0.0594 | 57.3% |
| on | 128 | 23.96 | 6089.98 | 64.00 | 0.0156 | 0.0501 | 68.3% |
| on | 256 | 40.89 | 23339.72 | 90.52 | 0.0110 | 0.0456 | 75.6% |
| off | 32 | 144.81 | 28888.36 | 727.37 | 0.7103 | 2.9446 | 75.6% |
| off | 64 | 283.20 | 105695.33 | 1417.80 | 0.6923 | 2.9266 | 76.2% |
| off | 128 | 559.47 | 403666.56 | 2797.60 | 0.6830 | 2.9173 | 76.5% |
| off | 256 | 1112.47 | 1577108.26 | 5557.94 | 0.6785 | 2.9128 | 76.7% |

随着 batch 增大：

- Reorder on 下，batch 从 32 增到 256，平均 DRAM J/step 从 43.97 增到 90.52，只有约 2.06 倍；但 token 数增加 8 倍，所以 DRAM J/token 从 0.0429 降到 0.0110，下降约 74.3%。这说明重排后 batch 放大能有效摊薄整批 step 的固定/共享内存开销。
- Reorder off 下，batch 从 32 增到 256，平均 DRAM J/step 从 727.37 增到 5557.94，约 7.64 倍，接近 batch 的 8 倍增长；DRAM J/token 只从 0.7103 降到 0.6785，下降约 4.5%。这说明不重排时每个 token 仍承担大量内存搬运成本，batch 放大几乎不能摊薄。
- 从延时看，reorder on 的平均延时从 11.06 ms 增到 40.89 ms，reorder off 从 144.81 ms 增到 1112.47 ms。延时都会随 batch 增加，但 reorder off 的绝对延时和 `memory_duration` 都高得多。
- 含 MAC 后，reorder on 的 `Total+MAC J/token` 从 0.0775 降到 0.0456；reorder off 仍维持在约 2.91-2.94 J/token。也就是说，reorder on 才能把 DRAM 能耗压到计算能耗可比较甚至更小的范围。

## REF 的影响

24 组 Ramulator-on 结果中 REF 全部非零：

| Reorder | Avg REF J/step | REF 占 DRAM | Max REF J/step |
|---|---:|---:|---:|
| on | 0.42 | 0.6% | 1.06 |
| off | 29.25 | 1.1% | 106.55 |
| 全部 | 14.84 | 0.9% | 106.55 |

结论：

- REF 不是总 DRAM 能耗的主导项，平均只占 0.9%。
- REF 绝对值随长 sequence 和大 batch 增大明显，最大达到 106.55 J/step，对应 `reorder=off, seq=8192, B/GPU=256`。
- 如果分析 refresh 成本或长时间运行的静态/周期性开销，应使用 `data_drampower_ramulcmd_ref/` 这批 Ramulator issued-command 口径结果。
## 内存占用

| Reorder | Batch/GPU | Seq Len | Activation (GB) | Weight (GB) | KV Cache (GB) | Total (GB) | Utilization | OOM |
|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| on | 32 | 2048 | 0.05 | 69.05 | 4.22 | 73.33 | 38.19% | NO |
| on | 32 | 4096 | 0.08 | 69.05 | 8.44 | 77.58 | 40.40% | NO |
| on | 32 | 8192 | 0.14 | 69.05 | 16.88 | 86.08 | 44.83% | NO |
| on | 64 | 2048 | 0.10 | 69.05 | 8.45 | 77.60 | 40.42% | NO |
| on | 64 | 4096 | 0.16 | 69.05 | 16.88 | 86.10 | 44.84% | NO |
| on | 64 | 8192 | 0.29 | 69.05 | 33.76 | 103.10 | 53.70% | NO |
| on | 128 | 2048 | 0.20 | 69.05 | 16.89 | 86.15 | 44.87% | NO |
| on | 128 | 4096 | 0.33 | 69.05 | 33.77 | 103.15 | 53.72% | NO |
| on | 128 | 8192 | 0.58 | 69.05 | 67.52 | 137.15 | 71.43% | NO |
| on | 256 | 2048 | 0.41 | 69.05 | 33.78 | 103.24 | 53.77% | NO |
| on | 256 | 4096 | 0.66 | 69.05 | 67.53 | 137.24 | 71.48% | NO |
| on | 256 | 8192 | 1.16 | 69.05 | 135.03 | 205.24 | 106.90% | YES |
| off | 32 | 2048 | 4.05 | 69.05 | 4.22 | 77.32 | 40.27% | NO |
| off | 32 | 4096 | 8.08 | 69.05 | 8.44 | 85.57 | 44.57% | NO |
| off | 32 | 8192 | 16.14 | 69.05 | 16.88 | 102.07 | 53.16% | NO |
| off | 64 | 2048 | 8.09 | 69.05 | 8.45 | 85.59 | 44.58% | NO |
| off | 64 | 4096 | 16.16 | 69.05 | 16.88 | 102.09 | 53.17% | NO |
| off | 64 | 8192 | 32.28 | 69.05 | 33.76 | 135.09 | 70.36% | NO |
| off | 128 | 2048 | 16.19 | 69.05 | 16.89 | 102.13 | 53.19% | NO |
| off | 128 | 4096 | 32.31 | 69.05 | 33.77 | 135.13 | 70.38% | NO |
| off | 128 | 8192 | 64.56 | 69.05 | 67.52 | 201.13 | 104.76% | YES |
| off | 256 | 2048 | 32.37 | 69.05 | 33.78 | 135.21 | 70.42% | NO |
| off | 256 | 4096 | 64.62 | 69.05 | 67.53 | 201.21 | 104.80% | YES |
| off | 256 | 8192 | 129.12 | 69.05 | 135.03 | 333.21 | 173.55% | YES |


## ACT、READ、WRITE、REF、Background 的角色

### ACT

Reorder on 下 ACT 平均 11.3%，reorder off 下 ACT 平均 7.8%。ACT 占比并不一定随总能耗同步增加，因为 reorder off 中 WRITE 大幅增加，把 ACT 比例稀释了。

### READ

READ 是 reorder on 的主导项，平均 82.4%。这说明重排把访问模式压到以读为主，写回和搬运被显著压低。

### WRITE

WRITE 是区分 reorder on/off 的关键项。Reorder on 平均 2.8%，reorder off 平均 37.6%。不重排时，大量写相关访问使 DRAM 能耗急剧上升。

### REF

REF 全部非零，但平均只占 0.6%-1.1%。它不是主导项，不过在长 sequence、大 batch 下绝对值不可忽略。

### Background

当前 CSV 中 Background 占 2.8%-3.2%，但该比例依赖 CSV 生成时的 `background_time` 口径。当前代码改动后，Background 将按 Ramulator elapsed memory time 送入 DRAMPower；预计数值会比当前 CSV 大很多，需要重跑确认。

## 最终结论

1. 最可信的 Ramulator 指令口径结果是 `data_drampower_ramulcmd_ref/` 的 24 组：ACT/READ/WRITE/REF 全部来自 Ramulator controller issued command counter。
2. Reorder 是功耗差异的最大因素。Ramulator-on + DRAMPower-style 下，reorder off 平均 DRAM J/step 是 reorder on 的约 42 倍，最大逐组差异达到 78.5 倍。
3. Reorder on 的 DRAM 能耗主要由 READ 主导；reorder off 中 WRITE 占比大幅上升，是总能耗爆炸的关键原因。
4. 24 组 Ramulator-on 结果的 REF 全部非零，但 REF 平均只占 0.6%-1.1%，不是主导项。
5. Background 已被计入 DRAMPower-style 总能耗，但当前 CSV 仍是生成时的 background_time 口径；当前代码已经改为 Ramulator elapsed memory time，需要重新跑实验才能给出新 Background 结果。
6. `mac_energy` 不属于 `drampower_total_energy`。分析 DRAM 本身时看 `drampower_total_energy`；分析 DRAM+计算时看 `drampower_total_energy + mac_energy`。
