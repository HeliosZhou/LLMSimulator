# DRAMSpec-Calibrated HBM3E-Like Experiment

This experiment keeps the B200/HBM3E system target but replaces the memory timing and DRAMPower-style current parameters with a DRAMSpec-calibrated HBM3E-like configuration.

## Parameter Source

- DRAMSpec technology input: `inputs/tech_hbm3e_calibrated_10nm.json`
- DRAMSpec architecture input: `inputs/arch_hbm3e_like_b200_24gb_8gbps.json`
- Generated Ramulator config: `generated/dram_config_HBM3E_DRAMSpec.yaml`
- Generated DRAMPower-style config: `generated/dramspec_hbm3e_like_power.yaml`
- Included modes: reorder=on, ramulator=on,off.
- Scope: calibrated HBM3E-like model, not vendor datasheet-level HBM3E.

## DRAMSpec Output Snapshot

| Parameter | Value |
|---|---:|
| trcd ns | 3.7524 |
| tcl ns | 8.4413 |
| tras ns | 11.1501 |
| trp ns | 3.4032 |
| trc ns | 14.5533 |
| twr ns | 3.9881 |
| trfc ns | 200.7924 |
| trefI ns | 3900.0000 |
| IDD0 mA | 41.2220 |
| IDD2n mA | 20.5421 |
| IDD3n mA | 26.0421 |
| IDD4R mA | 537.5265 |
| IDD4W mA | 537.5265 |
| IDD5B mA | 243.5453 |

## Results

| Reorder | Ramulator | Seq | Batch/GPU | Latency ms | DRAMPower J/step | ACT | READ | WRITE | REF |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| on | off | 2048 | 32 | 8.0994 | 23.425701 | 9648824 | 303774176 | 4563567 | 0 |
| on | off | 2048 | 64 | 11.6835 | 25.980450 | 10659334 | 333058724 | 9040039 | 0 |
| on | off | 2048 | 128 | 18.8011 | 31.093633 | 12741747 | 391629901 | 17994878 | 0 |
| on | off | 2048 | 256 | 32.9555 | 41.322930 | 16959500 | 508772141 | 35904684 | 0 |
| on | off | 4096 | 32 | 8.3864 | 24.770519 | 10201784 | 321468896 | 4563567 | 0 |
| on | off | 4096 | 64 | 12.2537 | 28.670087 | 11765254 | 368448164 | 9040039 | 0 |
| on | off | 4096 | 128 | 19.9374 | 36.472907 | 14953587 | 462408781 | 17994878 | 0 |
| on | off | 4096 | 256 | 35.2243 | 52.081478 | 21383180 | 650329901 | 35904684 | 0 |
| on | off | 8192 | 32 | 8.9605 | 27.460156 | 11307704 | 356858336 | 4563567 | 0 |
| on | off | 8192 | 64 | 13.3940 | 34.049361 | 13977094 | 439227044 | 9040039 | 0 |
| on | off | 8192 | 128 | 22.2102 | 47.231456 | 19377267 | 603966541 | 17994878 | 0 |
| on | off | 8192 | 256 | 39.7621 | 73.598575 | 30230540 | 933445421 | 35904684 | 0 |
| on | on | 2048 | 32 | 10.0271 | 24.273045 | 23090343 | 297811878 | 4525826 | 58252 |
| on | on | 2048 | 64 | 13.8966 | 26.592622 | 25475881 | 321244409 | 9002455 | 72958 |
| on | on | 2048 | 128 | 21.1584 | 31.092263 | 27864539 | 368109452 | 17955444 | 105598 |
| on | on | 2048 | 256 | 35.7528 | 40.021017 | 32848181 | 461842531 | 35864815 | 156324 |
| on | on | 4096 | 32 | 10.3986 | 25.712315 | 23993799 | 315506598 | 4525826 | 68908 |
| on | on | 4096 | 64 | 14.6124 | 29.433406 | 27099049 | 356633849 | 9002455 | 90238 |
| on | on | 4096 | 128 | 22.5735 | 36.759154 | 30866459 | 438888332 | 17955444 | 140158 |
| on | on | 4096 | 256 | 38.5832 | 51.362240 | 39002695 | 603400291 | 35864815 | 225200 |
| on | on | 8192 | 32 | 11.1181 | 28.554071 | 25634823 | 350896038 | 4525826 | 86188 |
| on | on | 8192 | 64 | 16.0382 | 35.107328 | 30216649 | 427412729 | 9002455 | 124798 |
| on | on | 8192 | 128 | 25.3893 | 48.074405 | 36559259 | 580446092 | 17955444 | 209278 |
| on | on | 8192 | 256 | 44.2070 | 73.992532 | 50384455 | 886515811 | 35864815 | 363440 |

## Baseline Comparison

Baseline is `experiments/exp_mem_arch/data/summary_hbm3e.csv` filtered to `drampower=on`, i.e. the previous HBM3 timing + HBM2-derived current adapter.

| Reorder | Ramulator | Seq | Batch/GPU | Latency Ratio | DRAMPower Ratio |
|---|---|---:|---:|---:|---:|
| on | off | 2048 | 32 | 1.000x | 0.615x |
| on | on | 2048 | 32 | 0.957x | 0.596x |
| on | on | 8192 | 256 | 0.956x | 0.617x |

## Notes

- The 10nm calibrated input lowers `IDD4R/IDD4W` substantially versus the earlier 29nm HBM-derived input; treat the result as HBM3E-like until calibrated against JEDEC/vendor current tables.
- Current implementation overrides the Ramulator config through `system.dram_config_path` and the DRAMPower-style adapter through `system.optimization.dram_power_config_path`.
