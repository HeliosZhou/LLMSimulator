# Exp Mem Arch LPDDR: Ramulator-On With LPDDR5 DRAMPower Parameters

This experiment is derived from `experiments/exp_mem_arch`.

It keeps the same B200/HBM3E system-level target and Ramulator command-count
path, but switches DRAMPower energy accounting from the HBM3E adapter parameters
to LPDDR5 parameters.

## Matrix

| Dimension | Values | Count |
|---|---|---:|
| System memory target | HBM3E | 1 |
| DRAMPower parameter model | LPDDR5 | 1 |
| Reordering | on, off | 2 |
| Sequence length | 2048, 4096, 8192 | 3 |
| Batch per GPU | 32, 64, 128, 256 | 4 |
| Ramulator hierarchy | on | 1 |
| Total | | 24 |

The simulator still uses 8 TB/s bandwidth, 192 GiB capacity, and
`system.ramulator_sample_stride = 1`. Only DRAMPower energy parameters are
switched by setting:

```yaml
system:
  optimization:
    use_ramulator: true
    use_drampower: true
    dram_power_model: lpddr5
```

## Run

```bash
bash experiments/exp_mem_arch_lpddr/run_experiments.sh
python3 experiments/exp_mem_arch_lpddr/analyze_lpddr.py --all
```

## Outputs

- `data/result_hbm3e_lpddr_b{B}_l{L}_reorder_{on|off}_ramul_on.csv`: raw simulator CSV.
- `data/summary_lpddr.csv`: compact summary with latency, command counts, DRAMPower energy fields, and time fields.
- `configs/result_hbm3e_lpddr_*.yaml`: per-run configs.
- `logs/result_hbm3e_lpddr_*.log`: per-run stdout/stderr logs.
- `plots/lpddr5_*.png`: generated plots.
- `LPDDR5_DRAMPOWER_ANALYSIS.md`: generated markdown report.

`analyze_lpddr.py` summarizes only `type=t2t` rows. The raw CSV also contains
many scheduler-generated `type=e2e` latency rows whose DRAMPower fields are zero
and whose `dram_energy_model` remains `fgdram`; they are not used for the
energy tables.

## Interpretation

This is not a full LPDDR5 memory-system simulation. Ramulator still receives the
same HBM3E-targeted requests and command-count path as `exp_mem_arch`; the new
variable is the DRAMPower energy parameter set used for the aggregate
ACT/READ/WRITE/REF/background energy calculation.
