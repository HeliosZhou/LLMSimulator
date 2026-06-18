# DRAMSpec vs HBM3EAdapter Reorder-On Comparison

This note compares the current DRAMSpec-calibrated HBM3E-like simulation against
the HBM3EAdapter data used by the original DRAM command energy/share heatmaps.

## Data Scope

- Baseline data: `../exp_mem_arch/data/energy_breakdown_ramulator_on_drampower_ref.csv`
- DRAMSpec data: `data/summary_dramspec_hbm3e_like.csv`
- Filter: `reorder=on`, `ramulator=on`
- Sweep: batch/GPU `32, 64, 128, 256` and sequence length `2048, 4096, 8192`
- Components: `ACT`, `READ`, `WRITE`, `REF`, `BG`, `MAC`

Generated comparison artifacts:

- `data/dramspec_vs_hbm3e_adapter_reorder_on.csv`
- `plots/figure_dramspec_vs_hbm3e_adapter_reorder_on_dram_only_energy.png`
- `plots/figure_dramspec_vs_hbm3e_adapter_reorder_on_dram_only_share.png`
- `plots/figure_dramspec_vs_hbm3e_adapter_reorder_on_energy.png`
- `plots/figure_dramspec_vs_hbm3e_adapter_reorder_on_share.png`

## Figure Meaning

The absolute-energy figure reports `J/step`. It should be used to compare how
much energy each model assigns to each command component.

The share figure reports each component as a percentage of `DRAM + MAC`. It
should be used to compare the energy composition. Because MAC is unchanged
between the two experiments, its percentage rises when DRAM energy decreases.

The DRAM-only figures mirror
`../exp_mem_arch/plots/figure_dram_only_command_energy_heatmaps.png` and
`../exp_mem_arch/plots/figure_dram_only_command_share_heatmaps.png`: they omit
MAC, and the share figure uses DRAM total as the denominator.

## Key Result

Across the reorder-on, Ramulator-on sweep, DRAMSpec-calibrated DRAM-only energy
is about `0.59x-0.61x` of the HBM3EAdapter baseline.

Representative points:

| Batch/GPU | Seq | HBM3EAdapter DRAM J/step | DRAMSpec DRAM J/step | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 2048 | 40.9099 | 24.2730 | 0.5933 |
| 256 | 8192 | 120.9356 | 73.9925 | 0.6118 |

Component-level energy ratios, DRAMSpec over HBM3EAdapter:

| Component | Ratio range | Interpretation |
| --- | ---: | --- |
| ACT | `0.247x-0.257x` | Lower because DRAMSpec timing/current makes ACT unit energy smaller. |
| READ | `0.638x` | Lower because generated `IDD4R` minus background current gives lower per-read event energy. |
| WRITE | `0.638x` | Same ratio as READ because current DRAMSpec config has `IDD4W = IDD4R`. |
| REF | `1.84x-2.26x` | Higher because generated refresh current/timing leads to larger REF event energy. |
| BG | `0.517x-0.536x` | Lower because DRAMSpec uses lower `VDD` and `IDD2N`. |
| MAC | `1.000x` | Same workload and compute-energy model. |

## What Changed Between Models

Both paths still use the local `HBM3EAdapter` event-level formula:

```text
event_energy = max(IDD-derived event energy, fallback) * command_parallelism * command_count
background_energy = VDD * IDD2N * background_time * command_parallelism
```

The difference is the parameter source.

### HBM3EAdapter Baseline

The baseline adapter defaults live in
`src/dram/ramulator2/src/drampower/hbm3e_adapter.h`.

Important parameters:

| Parameter | HBM3EAdapter baseline |
| --- | ---: |
| `VDD` | `1.2 V` |
| `IDD0` | `56 mA` |
| `IDD2N` | `33 mA` |
| `IDD4R` | `157 mA` |
| `IDD4W` | `135 mA` |
| `IDD5` | `118 mA` |
| `tRCD/tRAS/tRP` | `28/68/28 cycles` |
| `tRFC` | `400 cycles` |
| fallback ACT/RD/WR/REF | nonzero |

This should be treated as an HBM3E DRAMPower-style adapter, not a complete
vendor HBM3E memspec. Its current values are inherited from an HBM2-like
DRAMPower resource path and then used with the B200/HBM3E system target.

### DRAMSpec-Calibrated HBM3E-Like

The DRAMSpec path uses:

- `inputs/tech_hbm3e_calibrated_10nm.json`
- `inputs/arch_hbm3e_like_b200_24gb_8gbps.json`
- `generated/dramspec_hbm3e_like_power.yaml`
- `generated/dram_config_HBM3E_DRAMSpec.yaml`

Important generated power parameters:

| Parameter | DRAMSpec-calibrated |
| --- | ---: |
| `VDD` | `1.1 V` |
| `IDD0` | `41.222 mA` |
| `IDD2N` | `20.542 mA` |
| `IDD4R` | `537.527 mA` |
| `IDD4W` | `537.527 mA` |
| `IDD5` | `243.545 mA` |
| `tRCD/tRAS/tRP` | `8/23/7 cycles` |
| `tRFC` | `402 cycles` |
| fallback ACT/RD/WR/REF | `0` |

This path also overrides the Ramulator timing config through
`generated/dram_config_HBM3E_DRAMSpec.yaml`, so latency and command counts may
shift slightly relative to the baseline. The energy formula is still the same
adapter-style formula, but the timing/current inputs come from DRAMSpec.

## Interpretation

The main visible effect is that ACT, READ, WRITE, and BG energies drop under
the calibrated DRAMSpec input, while REF increases. The total DRAM energy still
drops because READ dominates reorder-on energy.

The result should be reported as **DRAMSpec-calibrated HBM3E-like**, not as a
datasheet-level HBM3E model. A stronger absolute-energy claim would require
JEDEC or vendor current tables for HBM3E/HBM3E-like devices.
