# Exp Mem Arch DRAMSpec: B200/HBM3E-Like Parameters

This experiment keeps the original `exp_mem_arch` B200/HBM3E system target but
uses DRAMSpec to generate an HBM3E-like timing/current parameter set.

The model is intentionally labeled **DRAMSpec-calibrated HBM3E-like**. It is not
a vendor datasheet-level HBM3E memspec.

## What Changes

- Ramulator config is overridden through `system.dram_config_path`.
- DRAMPower-style current/timing parameters are overridden through
  `system.optimization.dram_power_config_path`.
- LLMSimulator top-level target remains B200, 192 GiB HBM3E, 8 TB/s.

## Important Files

- `tools/DRAMSpec/`: local clone of `https://github.com/tukl-msd/DRAMSpec.git`
  used as the parameter generator. This directory is git-ignored.
- `inputs/tech_hbm3e_calibrated_10nm.json`: DRAMSpec technology input,
  using the 10nm HBM3E-like calibration documented in
  `DRAMSPEC_PARAMETERS.md`.
- `inputs/arch_hbm3e_like_b200_24gb_8gbps.json`: B200/HBM3E-like architecture
  input aligned with the current LLMSimulator/Ramulator HBM3 organization.
- `generated/dram_config_HBM3E_DRAMSpec.yaml`: generated Ramulator config.
- `generated/dramspec_hbm3e_like_power.yaml`: generated HBM3EAdapter parameter
  config.
- `data/summary_dramspec_hbm3e_like.csv`: reorder-on 24-run summary.
- `data/dramspec_vs_hbm3e_adapter_reorder_on.csv`: reorder-on component-level
  comparison against the HBM3EAdapter heatmap data.
- `plots/figure_dramspec_vs_hbm3e_adapter_reorder_on_dram_only_energy.png`:
  DRAM-only absolute energy comparison, matching the original DRAM-only
  heatmap layout.
- `plots/figure_dramspec_vs_hbm3e_adapter_reorder_on_dram_only_share.png`:
  DRAM-only energy-share comparison, matching the original DRAM-only heatmap
  layout.
- `DRAMSPEC_HBM3E_LIKE_ANALYSIS.md`: generated report.
- `DRAMSPEC_VS_HBM3E_ADAPTER_REORDER_ON.md`: comparison note explaining the
  DRAMSpec and HBM3EAdapter parameter/model differences.

## Build DRAMSpec

```bash
bash experiments/exp_mem_arch_dramspec/build_dramspec_local.sh
```

The script downloads and unpacks `libboost1.83-dev` into
`tools/deps/boost_root` instead of installing system packages, then compiles
DRAMSpec with `g++`.

## Generate Parameters

```bash
python3 experiments/exp_mem_arch_dramspec/generate_dramspec_configs.py
```

This writes raw DRAMSpec outputs to `generated/dramspec_raw/` and converted
LLMSimulator configs to `generated/`.

## Run

Smoke run:

```bash
bash experiments/exp_mem_arch_dramspec/run_experiments.sh
```

Reorder-on 24-run matrix:

```bash
BATCH_SIZES='32 64 128 256' \
SEQ_LENGTHS='2048 4096 8192' \
REORDERING_MODES='on' \
RAMULATOR_MODES='on off' \
bash experiments/exp_mem_arch_dramspec/run_experiments.sh
```

Use `FORCE_RERUN=1` when regenerating existing CSVs after a parameter change.

Analyze:

```bash
python3 experiments/exp_mem_arch_dramspec/analyze_dramspec.py --all
python3 experiments/exp_mem_arch_dramspec/analyze_dramspec.py --missing
```

Compare against the original HBM3EAdapter reorder-on heatmap data:

```bash
python3 experiments/exp_mem_arch_dramspec/plot_dramspec_vs_hbm3e_adapter.py
```

## Current Result

The reorder-on 24-run matrix is complete. The generated report compares against
`experiments/exp_mem_arch/data/summary_hbm3e.csv` filtered to `drampower=on`.

The calibrated 10nm parameter set lowers `IDD4R/IDD4W` from the earlier
29nm/HBM-derived `1413.88 mA` estimate to `537.53 mA`. In reorder-on runs,
DRAMPower totals are now roughly `0.60x` to `0.62x` of the old HBM3 timing +
HBM2-derived current adapter baseline for the highlighted Ramulator cases.
