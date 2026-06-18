# RoMe Figure 1 Trace Reproduction

This experiment reproduces the Figure 1 access-size distribution from LLMSimulator runtime tensor traces.

The trace run uses the paper-style 8-accelerator context:

- DeepSeek-V3: non-expert TP = 1, expert TP = 1, DP = 8.
- Grok 1: non-expert TP = 8, expert TP = 1.
- Llama 3-405B: non-expert TP = 8.

The exported CSVs are rank-0 per-accelerator traces under that
8-accelerator context. They are not merged traces from all 8 devices.

Run:

```bash
cmake --build build -j
python3 experiments/exp_rome0_background/run_trace_figure1.py
```

Regenerate only from existing traces:

```bash
python3 experiments/exp_rome0_background/reproduce_figure1.py --from-trace experiments/exp_rome0_background/data/*_trace.csv
```

Outputs:

- `data/*_trace.csv`
- `data/figure1_access_samples.csv`
- `data/figure1_access_summary.csv`
- `plots/rome_figure1_access_distribution.png`
- `plots/rome_figure1_access_distribution.pdf`
- `FIGURE1_REPRODUCTION.md`

The previous analytical model-parameter fallback has been removed; this directory is trace-only.
