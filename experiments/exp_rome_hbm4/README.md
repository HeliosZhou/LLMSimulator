# HBM4 Baseline Simulation (RoMe-style)

Reproduces the **HBM4 baseline** side of RoMe Figure 12 using LLMSimulator.

## Setup

- **Memory system**: HBM4 baseline (64 pseudo channels, 1KB row, 32B access, 2TB/s per stack)
- **Accelerators**: 8 × B200-class (560 TFLOPS BF16, 256 GB, 16 TB/s)
- **Models**: DeepSeek-V3 (MLA+MoE), Grok 1 (GQA+MoE), Llama 3-405B (GQA+dense)
- **Stage**: Decode only, sequence length = 8K
- **Batch sizes**: 8, 16, 32, 64, 128, 256, 512, 1024 (constrained by memory capacity)

## Parallel strategies (same as RoMe)

| Model | Attention TP | Expert TP | DP |
|-------|-------------|-----------|-----|
| DeepSeek-V3 | 1 | 1 | 8 |
| Grok 1 | 8 | 1 | 1 |
| Llama 3-405B | 8 | 1 | 1 |

## Usage

```bash
cd ~/LLMSimulator
cmake --build build -j
python3 experiments/exp_rome_hbm4/run_hbm4_simulation.py
```

## Output

- `data/hbm4_baseline_results.csv` — TPOT per model per batch size
- `data/*_trace.csv` — per-run tensor access traces

## Reference

- RoMe: arXiv:2512.01541, Figure 12
- Folded Banks: ISCA 2025, Table 1
- HBM4 parameters: `dram_config_HBM4_baseline.yaml`
