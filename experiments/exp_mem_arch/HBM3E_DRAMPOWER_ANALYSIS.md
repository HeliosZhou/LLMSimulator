# HBM3E DRAMPower Analysis

This report summarizes the 48 DRAMPower-enabled HBM3E runs in
`data_drampower/`.

## Scope

- Reordering: on/off
- Sequence length: 2048/4096/8192
- Batch per GPU: 32/64/128/256
- Ramulator hierarchy simulation: on/off
- DRAMPower rows in `data/summary_hbm3e.csv`: 48

Each `t2t` row is one decode step for the whole active batch. The energy fields
are whole-step energy values. Average per-token energy is computed as
`drampower_total_energy / numtoken`.

## Overall Ranges

| Metric | Min | Avg | Max |
|---|---:|---:|---:|
| Latency (ms) | 8.10 | 254.10 | 1883.74 |
| DRAMPower total energy (J/step) | 38.06 | 1309.56 | 9389.26 |
| DRAMPower background energy (J/step) | 0.78 | 39.29 | 301.47 |
| Per-token DRAMPower energy (J/token) | 0.0080 | 0.3481 | 1.1779 |
| Memory duration (ms) | 311.81 | 265850.12 | 2702974.28 |
| Background time (ms) | 154.74 | 7751.27 | 59476.13 |

## Energy Breakdown

Average contribution to `drampower_total_energy`:

| Component | Min % | Avg % | Max % |
|---|---:|---:|---:|
| ACT | 5.50 | 7.57 | 12.65 |
| READ | 49.90 | 68.96 | 91.22 |
| WRITE | 1.09 | 20.75 | 40.31 |
| REF | 0.00 | 0.00 | 0.00 |
| Background/static | 1.92 | 2.72 | 3.21 |

READ dominates when reordering is enabled. With reordering disabled, WRITE
energy becomes much larger because write command counts rise sharply.

## Group Averages

| Reorder | Ramulator | Latency ms | Energy J/step | J/token | Background % | READ % | WRITE % | ACT % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| on | on | 22.86 | 61.93 | 0.0235 | 2.86 | 82.93 | 2.85 | 11.36 |
| on | off | 19.31 | 60.40 | 0.0224 | 2.02 | 89.54 | 2.91 | 5.54 |
| off | on | 524.99 | 2595.93 | 0.6834 | 3.20 | 50.88 | 38.03 | 7.89 |
| off | off | 449.25 | 2519.99 | 0.6629 | 2.80 | 52.48 | 39.22 | 5.50 |

## Ramulator Effect

For the same reorder/sequence/batch setting, enabling Ramulator hierarchy
simulation changes the metrics as follows:

| Metric | Min ratio | Avg ratio | Max ratio |
|---|---:|---:|---:|
| Latency, Ramulator on/off | 1.11 | 1.19 | 1.31 |
| DRAMPower energy, Ramulator on/off | 0.98 | 1.03 | 1.07 |
| Memory duration, Ramulator on/off | 1.01 | 1.07 | 1.27 |

Ramulator mainly increases latency and memory service duration. DRAMPower energy
changes only moderately because command counts are close to the ideal-memory
case; the hierarchy model primarily exposes scheduling/waiting cost.

## Reordering Effect

For the same Ramulator/sequence/batch setting, disabling reordering is much more
expensive:

| Metric | Min ratio | Avg ratio | Max ratio |
|---|---:|---:|---:|
| Latency, reorder off/on | 6.47 | 20.10 | 40.75 |
| DRAMPower energy, reorder off/on | 8.11 | 34.93 | 78.32 |
| READ count, reorder off/on | 5.11 | 20.77 | 46.34 |
| WRITE count, reorder off/on | 221.26 | 521.64 | 898.83 |

The dominant effect is command-count explosion when reordering is disabled,
especially WRITE commands. This is the largest driver of energy growth in the
48-run matrix.

## Scaling By Sequence And Batch

Average across reorder and Ramulator settings:

| Seq | Batch/GPU | Latency ms | Energy J/step | J/token | READ count | WRITE count | Background % |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 32 | 35.86 | 181.88 | 0.1776 | 926814787 | 507123417 | 2.70 |
| 2048 | 64 | 66.23 | 328.33 | 0.1603 | 1579197006 | 1014179167 | 2.72 |
| 2048 | 128 | 126.69 | 620.85 | 0.1516 | 2883961996 | 2028291481 | 2.71 |
| 2048 | 256 | 247.72 | 1206.18 | 0.1472 | 5493492456 | 4056517870 | 2.69 |
| 4096 | 32 | 62.85 | 326.91 | 0.3192 | 1572672067 | 1010439897 | 2.72 |
| 4096 | 64 | 120.21 | 618.43 | 0.3020 | 2870911566 | 2020812127 | 2.73 |
| 4096 | 128 | 234.66 | 1201.07 | 0.2932 | 5467391117 | 4041557401 | 2.72 |
| 4096 | 256 | 463.68 | 2366.60 | 0.2889 | 10660350696 | 8083049710 | 2.71 |
| 8192 | 32 | 116.82 | 616.99 | 0.6025 | 2864386627 | 2017072857 | 2.73 |
| 8192 | 64 | 228.20 | 1198.67 | 0.5853 | 5454340686 | 4034078047 | 2.74 |
| 8192 | 128 | 450.66 | 2361.47 | 0.5765 | 10634249357 | 8068089241 | 2.73 |
| 8192 | 256 | 895.61 | 4687.36 | 0.5722 | 20994067176 | 16136113390 | 2.72 |

Longer sequence lengths increase attention-related memory traffic and therefore
raise both latency and energy. Larger batches raise total step energy, while
average energy per generated token tends to decrease within the same sequence
length because fixed per-step work is amortized across more tokens.

## Key Takeaways

1. DRAMPower total energy is dominated by READ traffic, except when reordering
   is disabled, where WRITE traffic becomes a major component.
2. Background/static energy is consistently visible but small, about 2.7% on
   average.
3. Ramulator hierarchy simulation increases latency by about 19% on average but
   increases DRAMPower energy by only about 3% on average.
4. Reordering is the main determinant of energy. Disabling it increases
   DRAMPower energy by about 35x on average in this matrix.
5. Use `drampower_total_energy / numtoken` for average generated-token DRAM
   energy. The CSV energy fields themselves are whole-batch decode-step energy.
