# Exp Mem Arch: HBM3E Ramulator Hierarchy

This experiment keeps only HBM3E and compares the simulator with Ramulator
hierarchy simulation enabled and disabled.

## Matrix

| Dimension | Values | Count |
|---|---|---:|
| Memory | HBM3E | 1 |
| Reordering | on, off | 2 |
| Sequence length | 2048, 4096, 8192 | 3 |
| Batch per GPU | 32, 64, 128, 256 | 4 |
| Ramulator hierarchy | on, off | 2 |
| Total | | 48 |

The HBM3E configuration uses 8 TB/s bandwidth, 192 GiB capacity, and
`system.ramulator_sample_stride = 1`.

DRAMPower is an independent energy accounting mode for the same 48-run HBM3E
matrix. It keeps latency and command-count collection unchanged and writes
separate `drampower_*` energy fields.

## Run

```bash
bash experiments/exp_mem_arch/run_experiments.sh
python3 experiments/exp_mem_arch/analyze_hbm3e.py --all
```

Run the DRAMPower matrix separately:

```bash
DRAMPOWER=on bash experiments/exp_mem_arch/run_experiments.sh
python3 experiments/exp_mem_arch/analyze_hbm3e.py --all
```

## Outputs

- `data/result_hbm3e_b{B}_l{L}_reorder_{on|off}_ramul_{on|off}.csv`: raw simulator CSV.
- `data_drampower/result_hbm3e_b{B}_l{L}_reorder_{on|off}_ramul_{on|off}.csv`: DRAMPower-enabled raw simulator CSV.
- `data/summary_hbm3e.csv`: compact summary with latency, command counts, energy fields, and time fields.
- `configs/result_hbm3e_*.yaml`: per-run configs.
- `configs_drampower/result_hbm3e_*.yaml`: per-run DRAMPower configs.
- `logs/result_hbm3e_*.log`: per-run stdout/stderr logs.
- `logs_drampower/result_hbm3e_*.log`: per-run DRAMPower stdout/stderr logs.
- `plots/hbm3e_ramulator_*.png`: generated plots.
- `HBM3E_ANALYSIS_REPORT.md`: generated markdown report.

## Command Counts And Time Fields

The raw CSV and `summary_hbm3e.csv` include:

- `act_count`, `read_count`, `write_count`, `ref_count`
- `all_act_count`, `all_read_count`, `all_write_count`
- `memory_duration`
- `background_time`
- `drampower_act_energy`, `drampower_read_energy`, `drampower_write_energy`
- `drampower_ref_energy`, `drampower_background_energy`, `drampower_total_energy`

`memory_duration` is the memory service duration accumulated by the simulator.
It is not the same as standby/background time.

`background_time` is the time base used for DRAM background/standby energy
accounting. In the current C++ implementation it is accumulated from each
execution segment's `total_duration`; it can include compute-dominated time and
must not be substituted for `memory_duration`.
