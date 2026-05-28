# Exp Mem: Memory Type Impact on LLM Inference

## Motivation

B200 default uses HBM3E (8 TB/s, 192 GB). This experiment compares the impact
of replacing the memory subsystem with DDR5 and GDDR6 while keeping all other
hardware parameters (compute, interconnect, GPU count) unchanged.

## Memory Configurations

| Config | Bandwidth (per GPU) | Capacity (per GPU) | Rationale |
|--------|--------------------|--------------------|-----------|
| HBM3E (baseline) | 8.0 TB/s | 192 GB | B200 default |
| GDDR6 | 512 GB/s | 192 GB | ~16x lower BW, same cap |
| DDR5 | 64 GB/s | 192 GB | ~125x lower BW, same cap |

All configs use capacity = 192 GB to isolate bandwidth as the variable.

## Sweep Grid

| Parameter | Values |
|-----------|--------|
| Memory type | HBM3E, GDDR6, DDR5 |
| Batch per GPU | 32, 64, 128, 256 |
| Sequence length | 2048, 4096, 8192 |
| Absorb mode | on, off |

Model: deepseekV3, GPU: B200, 4 nodes x 8 devices = 32 GPUs, decode mode.

## Usage

Run all experiments:

```bash
bash experiments/exp_mem/run_experiments.sh
```

Run and plot:

```bash
python3 experiments/exp_mem/run_memory_comparison.py --all
```

Plot only (from existing CSV data):

```bash
python3 experiments/exp_mem/run_memory_comparison.py --plot
```

## Output

- `data/result_{mem}_b{B}_l{L}_absorb_{on|off}.csv`
- `data/summary_memory_comparison.csv`
- `plots/figure_memory_comparison.png`
