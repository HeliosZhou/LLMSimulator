# Reorder-on ACT Sensitivity What-if

## Setup

- Input: `data/energy_breakdown_ramulator_on_drampower_ref.csv`
- Scope: `reorder=on`, `ramulator=on`, `type=t2t` rows already aggregated in the input CSV.
- Target: `READ+WRITE/ACT = 5`.
- Adjustment mode: `per-config`.
- Timing model: `row-episode`.
- Method: increase ACT count/energy to hit the target; keep READ, WRITE, and MAC energy fixed; rescale memory duration, REF, and background energy using the row-episode timing model.
- Interpretation: post-process timing-aware approximation, not a full Ramulator re-simulation.
- Row episode cycles: `max(nRAS, nRCD + (k-1)*nCCD + post_col_to_pre) + nRP`, with nRCD=28, nRAS=68, nRP=28, nCCD=4.

## Summary

- Baseline weighted `READ+WRITE/ACT`: **14.993**.
- Weighted memory-duration scale: **2.483x**.
- REF+background: **26.292 J/step** -> **62.701 J/step**.
- DRAM total: **748.153 J/step** -> **948.385 J/step** (**26.8%**).
- Total+MAC: **2338.988 J/step** -> **2539.221 J/step** (**8.6%**).

## Generated Figures

- DRAM-only absolute heatmap: `plots/figure_rowbuf_act5_dram_only_command_energy_heatmaps.png`
- DRAM-only relative heatmap: `plots/figure_rowbuf_act5_dram_only_command_share_heatmaps.png`
- Total+MAC absolute heatmap: `plots/figure_rowbuf_act5_dram_command_energy_heatmaps.png`
- Total+MAC relative heatmap: `plots/figure_rowbuf_act5_dram_command_share_heatmaps.png`

## Per Configuration

| Seq | Batch/GPU | Old accesses/ACT | New accesses/ACT | ACT scale | Time scale | Old DRAM J | New DRAM J | DRAM delta | Old Total+MAC J | New Total+MAC J | Total+MAC delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 32 | 13.031 | 5.000 | 2.606 | 2.221 | 40.910 | 50.718 | 24.0% | 65.514 | 75.322 | 15.0% |
| 2048 | 64 | 12.828 | 5.000 | 2.566 | 2.194 | 44.844 | 55.512 | 23.8% | 94.052 | 104.721 | 11.3% |
| 2048 | 128 | 14.089 | 5.000 | 2.818 | 2.294 | 51.832 | 65.071 | 25.5% | 150.249 | 163.488 | 8.8% |
| 2048 | 256 | 15.256 | 5.000 | 3.051 | 2.372 | 66.211 | 84.028 | 26.9% | 263.044 | 280.861 | 6.8% |
| 4096 | 32 | 13.163 | 5.000 | 2.633 | 2.234 | 43.289 | 53.790 | 24.3% | 75.954 | 86.454 | 13.8% |
| 4096 | 64 | 13.503 | 5.000 | 2.701 | 2.257 | 49.315 | 61.519 | 24.7% | 114.645 | 126.849 | 10.6% |
| 4096 | 128 | 14.983 | 5.000 | 2.997 | 2.372 | 60.899 | 77.139 | 26.7% | 191.559 | 207.799 | 8.5% |
| 4096 | 256 | 16.522 | 5.000 | 3.304 | 2.478 | 84.408 | 108.410 | 28.4% | 345.728 | 369.730 | 6.9% |
| 8192 | 32 | 13.982 | 5.000 | 2.796 | 2.307 | 47.717 | 59.830 | 25.4% | 96.504 | 108.616 | 12.6% |
| 8192 | 64 | 14.248 | 5.000 | 2.850 | 2.324 | 58.536 | 73.666 | 25.8% | 156.109 | 171.239 | 9.7% |
| 8192 | 128 | 16.091 | 5.000 | 3.218 | 2.464 | 79.257 | 101.519 | 28.1% | 274.403 | 296.666 | 8.1% |
| 8192 | 256 | 17.935 | 5.000 | 3.587 | 2.588 | 120.936 | 157.184 | 30.0% | 511.228 | 547.476 | 7.1% |

## Notes

- Lowering accesses/ACT from the baseline requires more ACT commands for the same READ/WRITE traffic.
- The row-episode model accounts for the fact that lower accesses/ACT also shortens each opened-row column-service episode; it does not simply multiply time by ACT scale.
- REF and background energy are scaled by the estimated memory-duration change; READ, WRITE, and MAC energy are unchanged.
- The model does not capture real queueing, bank-level overlap, row conflicts, refresh scheduling, or read/write turnaround. A full experiment would need Ramulator to emit or enforce a different access trace/address mapping and rerun timing.
