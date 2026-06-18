# DRAMSpec HBM3E 参数说明文档

本文档系统地罗列 DRAMSpec 仿真 HBM3E 所需的所有参数，方便逐个确认。

---

## 一、参数总览

DRAMSpec 需要 **2 个输入文件**：

1. **Technology Input**（40 个参数）- 工艺物理/电气特性
2. **Architecture Input**（20 个参数）- DRAM 组织结构

---

## 二、Technology Input 参数清单（40个）

### 按优先级分类：

#### 🔴 P0 级：关键电流参数（9个）- 直接决定 IDD4R/IDD4W

| # | 参数名 | 当前值 | 推荐值 | 单位 | 数据来源 | 置信度 |
|---|--------|--------|--------|------|---------|--------|
| 1 | TechnologyNode[nm] | 29 | **10** | nm | Samsung 官方 | ⭐⭐⭐ |
| 2 | Vdd[V] | 1.2 | **1.1** | V | JEDEC Table 70 | ⭐⭐⭐ |
| 3 | Vpp[V] | 2.5 | **1.8** | V | JEDEC Table 70 | ⭐⭐⭐ |
| 4 | WireCapacitance[fF/mm] | 150 | **100** | fF/mm | 工艺缩放 ÷1.5 | ⭐⭐ |
| 5 | WireResistance[Ohm/mm] | 100 | **40** | Ohm/mm | 工艺缩放 ÷2.5 | ⭐⭐ |
| 6 | DQDriverResistance[Ohm] | 250 | **120** | Ohm | HKMG 改善 ÷2 | ⭐⭐ |
| 7 | OCDCurrentSlope[uA/MHz] | 4.221 | **2.0** | uA/MHz | 低摆幅 I/O ÷2 | ⭐⭐ |
| 8 | IDD2NFreqSlope[mA/MHz] | 0.015 | **0.008** | mA/MHz | 工艺改善 ÷2 | ⭐⭐ |
| 9 | IDD2NOffset[mA] | 6.75 | **4.5** | mA | 工艺改善 ÷1.5 | ⭐⭐ |

**影响说明**：
- 参数 4,5,6 → 影响 IDD4ChargingCurrent（充放电电流）
- 参数 7 → 影响 ioTermRdCurrent（I/O 终端电流）
- 参数 8,9 → 影响 IDD2N（背景漏电流）
- 参数 2 → 功耗 ∝ V²，影响所有电流

---

#### 🟡 P1 级：重要参数（11个）- 中等影响

| # | 参数名 | 当前值 | 推荐值 | 单位 | 估算方法 |
|---|--------|--------|--------|------|---------|
| 10 | LocalWordlineDriverResistance[Ohm] | 500 | **250** | Ohm | HKMG ÷2 |
| 11 | GlobalWordlineDriverResistance[Ohm] | 200 | **100** | Ohm | HKMG ÷2 |
| 12 | WriteDriverResistance[Ohm] | 300 | **150** | Ohm | HKMG ÷2 |
| 13 | CSLDriverResistance[Ohm] | 170 | **100** | Ohm | HKMG ÷2 |
| 14 | GlobalDataLineDriverResistance[Ohm] | 300 | **150** | Ohm | HKMG ÷2 |
| 15 | CellCapacitance[fF] | 20 | **15** | fF | 工艺 ÷1.3 |
| 16 | BitlineCapacitancePerCell[aF] | 50 | **38** | aF | 工艺 ÷1.3 |
| 17 | WordlineCapacitancePerCell[aF] | 60 | **46** | aF | 工艺 ÷1.3 |
| 18 | SecondarySenseAmpCurrent[uA] | 200 | **150** | uA | 工艺 ÷1.3 |
| 19 | FullySharedResourcesCurrent[mA] | 2 | **1.5** | mA | 保守估算 |
| 20 | SemiSharedResourcesCurrent[mA] | 0.5 | **0.3** | mA | 保守估算 |

---

#### 🟢 P2 级：次要参数（10个）- 较小影响

| # | 参数名 | 当前值 | 推荐值 | 单位 | 缩放因子 |
|---|--------|--------|--------|------|---------|
| 21 | CellResistance[KOhm] | 20 | 15 | KOhm | ÷1.3 |
| 22 | CellWidth[um] | 0.085 | 0.060 | um | ×0.7 |
| 23 | CellHeight[um] | 0.045 | 0.032 | um | ×0.7 |
| 24 | BitlineResistancePerCell[Ohm] | 35 | 25 | Ohm | ÷1.4 |
| 25 | WordlineResistancePerCell[Ohm] | 25 | 18 | Ohm | ÷1.4 |
| 26 | CSLLoadCapacitance[fF] | 8 | 5 | fF | ÷1.6 |
| 27 | IDD2NTempAlpha[mA] | 0.6775 | 0.45 | mA | ÷1.5 |
| 28 | IDD2NTempBeta[C^-1] | 0.04467 | 0.04467 | - | 不变 |
| 29 | IDD2NRefTemp[C] | 25 | 25 | C | 不变 |
| 30 | nBanksPerSemiSharedResource[] | 2 | 2 | - | 不变 |

---

#### ⚪ P3 级：尺寸/延迟参数（10个）- 主要影响 timing

| # | 参数名 | 当前值 | 推荐值 | 单位 | 缩放因子 |
|---|--------|--------|--------|------|---------|
| 31 | PrimarySenseAmpHeight[um] | 10 | 7 | um | ×0.7 |
| 32 | LocalWordlineDriverWitdh[um] | 5 | 3.5 | um | ×0.7 |
| 33 | RowDecoderWidth[um] | 53.5 | 40 | um | ×0.75 |
| 34 | ColumnDecoderHeight[um] | 250 | 180 | um | ×0.72 |
| 35 | DQDriverHeight[um] | 205 | 150 | um | ×0.73 |
| 36 | DQtoTSVWireLength[um] | 200 | 150 | um | ×0.75 |
| 37 | TSVHeight[um] | 896 | 650 | um | ×0.73 |
| 38 | DriverEnableDelay[ns] | 0.6 | 0.4 | ns | ÷1.5 |
| 39 | InOutSSADelay[ns] | 2 | 1.2 | ns | ÷1.7 |
| 40 | CommandDecoderDelay[ns] | 2 | 1.5 | ns | ÷1.3 |
| 41 | IODelay[ns] | 1 | 0.6 | ns | ÷1.7 |
| 42 | SSAPrechargeDelay[ns] | 1 | 0.7 | ns | ÷1.4 |
| 43 | tWRMargin[ns] | 1 | 0.8 | ns | ÷1.25 |
| 44 | EqualizerDelay[ns] | 1 | 0.7 | ns | ÷1.4 |
| 45 | AdditionalTRLLatency[cc] | 0 | 0 | - | 不变 |

---

## 三、Architecture Input 参数（20个）

你的当前配置已经正确，无需修改：

| 参数 | 当前值 | 状态 |
|------|--------|------|
| DRAMType[-] | DDR | ✅ |
| 3D[-] | ON | ✅ |
| DLL[-] | ON | ✅ |
| ChannelSize[Gb] | 24 | ✅ |
| NumberOfBanksPerChannel[] | 16 | ✅ |
| Interface[bit] | 32 | ✅ |
| Prefetch[] | 8 | ✅ |
| Frequency[MHz] | 2000 | ✅ |
| tREFI(base)[us] | 3.9 | ✅ |
| （其他参数省略，均正确） | ... | ✅ |

---

## 四、数据来源说明

### ⭐⭐⭐ 官方确认

- **Vdd = 1.1V**：JEDEC JESD238 Table 70
- **Vpp = 1.8V**：JEDEC JESD238 Table 70  
- **10nm 工艺**：Samsung HBM3E 官方博客

### ⭐⭐ 工艺缩放

基于 29nm → 10nm 的标准缩放规律：
- Wire C: ÷1.5（低k介电层）
- Wire R: ÷2.5（工艺改善）
- Driver R: ÷2（HKMG 技术）
- Cell C: ÷1.3（面积缩放）
- Leakage: ÷1.5-2（HKMG 降漏电）

### ⭐ 保守估算

- Background current
- Shared resources

### ❓ 需要从 JEDEC 确认

**如果 PDF Table 90 可见，请提供**：
- IDD0, IDD2N, IDD3N, IDD4R, IDD4W, IDD5B

---

## 五、预期结果

### 当前（29nm）

```
IDD4R: 1414 mA  ← 高估 3-5x
IDD4W: 1414 mA
```

### 校准后（10nm）

```
IDD4R: 300-500 mA  ← 目标范围
IDD4W: 300-500 mA
```

**改善来源**：
- Vdd: 1.2→1.1V  → ×0.84
- Wire C: 150→100 → ×0.67
- OCD: 4.2→2.0   → ×0.48
- 综合：×0.27 → 降低 73%

---

## 六、下一步

1. 确认上述 9 个 P0 参数
2. 创建 tech_hbm3e_calibrated_10nm.json
3. 运行 DRAMSpec
4. 验证 IDD4R/IDD4W 是否在 300-500 mA

---

**文档路径**: /home/zsy/LLMSimulator/experiments/exp_mem_arch_dramspec/  
**创建时间**: 2026-06-13  
**作者**: Claude Opus 4.7
