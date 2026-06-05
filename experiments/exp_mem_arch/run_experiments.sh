#!/usr/bin/env bash
# HBM3E Ramulator hierarchy experiment.
#
# Sweep:
#   reordering(on/off) x seq_len(2048/4096/8192) x batch_per_gpu(32/64/128/256)
#   x ramulator(on/off)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
EXP_DIR="$PROJECT_DIR/experiments/exp_mem_arch"
DATA_DIR="$EXP_DIR/data"
CONFIG_DIR="$EXP_DIR/configs"
LOG_DIR="$EXP_DIR/logs"
PLOT_DIR="$EXP_DIR/plots"

mkdir -p "$DATA_DIR" "$CONFIG_DIR" "$LOG_DIR" "$PLOT_DIR"

BATCH_SIZES=(32 64 128 256)
SEQ_LENGTHS=(2048 4096 8192)
REORDERING_MODES=(on off)
RAMULATOR_MODES=(on off)

OUTPUT_LEN=2
PRECISION_BYTE=1
ITER=3

GPU_GEN="B200"
NUM_NODE=4
NUM_DEVICE=8
MEMORY_TYPE="hbm3e"
MEMORY_BW=8000000000000
MEMORY_CAP=206158430208
RAMULATOR_SAMPLE_STRIDE=1

TOTAL=$(( ${#REORDERING_MODES[@]} * ${#SEQ_LENGTHS[@]} * ${#BATCH_SIZES[@]} * ${#RAMULATOR_MODES[@]} ))

echo "==========================================="
echo "HBM3E Ramulator hierarchy experiment"
echo "GPU: $GPU_GEN, nodes: $NUM_NODE, devices/node: $NUM_DEVICE"
echo "Memory: HBM3E, bandwidth: $MEMORY_BW B/s, capacity: $MEMORY_CAP bytes"
echo "Reordering modes: ${REORDERING_MODES[*]}"
echo "Seq lengths: ${SEQ_LENGTHS[*]}"
echo "Batch/GPU: ${BATCH_SIZES[*]}"
echo "Ramulator modes: ${RAMULATOR_MODES[*]}"
echo "Total combinations: $TOTAL"
echo "==========================================="

for REORDER in "${REORDERING_MODES[@]}"; do
  for SEQ_LEN in "${SEQ_LENGTHS[@]}"; do
    for BATCH in "${BATCH_SIZES[@]}"; do
      for RAMUL in "${RAMULATOR_MODES[@]}"; do
        RESULT_NAME="result_${MEMORY_TYPE}_b${BATCH}_l${SEQ_LEN}_reorder_${REORDER}_ramul_${RAMUL}.csv"
        CONFIG_NAME="${RESULT_NAME%.csv}.yaml"
        LOG_NAME="${RESULT_NAME%.csv}.log"

        if [[ -f "$DATA_DIR/$RESULT_NAME" ]]; then
          echo "[SKIP] $RESULT_NAME already exists"
          continue
        fi

        echo ""
        echo ">>> HBM3E reorder=${REORDER} ramulator=${RAMUL} batch/GPU=${BATCH} seq=${SEQ_LEN}"

        TMP_RUN_DIR="$(mktemp -d "$DATA_DIR/.tmp_${RESULT_NAME%.csv}.XXXXXX")"
        CONFIG_PATH="$CONFIG_DIR/$CONFIG_NAME"

        python3 - "$BUILD_DIR/config.yaml" "$CONFIG_PATH" "$TMP_RUN_DIR" "$REORDER" "$RAMUL" "$BATCH" "$SEQ_LEN" <<'PY'
import sys
from pathlib import Path

import yaml

base_config = Path(sys.argv[1])
config_path = Path(sys.argv[2])
output_dir = Path(sys.argv[3])
reorder = sys.argv[4]
ramulator = sys.argv[5]
batch_per_gpu = int(sys.argv[6])
seq_len = int(sys.argv[7])

num_node = 4
num_device = 8

with base_config.open("r") as f:
    cfg = yaml.safe_load(f)

cfg["model"]["model_name"] = "deepseekV3"
cfg["system"]["gpu_gen"] = "B200"
cfg["system"]["num_node"] = num_node
cfg["system"]["num_device"] = num_device
cfg["system"]["distribution"]["expert_tensor_degree"] = 1
cfg["system"]["distribution"]["none_expert_tensor_degree"] = 1

opt = cfg["system"]["optimization"]
opt["use_absorb"] = reorder == "on"
opt["compressed_kv"] = True
opt["use_flash_mla"] = True
opt["use_flash_attention"] = True
opt["reuse_kv_cache"] = True
opt["kv_cache_reuse_rate"] = 0.0
opt["parallel_execution"] = False
opt["hetero_subbatch"] = False
opt["disagg_system"] = False
opt["use_low_unit_moe_only"] = False
opt["use_ramulator"] = ramulator == "on"
opt["prefill_mode"] = False
opt["decode_mode"] = True

cfg["system"]["processor_type"] = "GPU"
cfg["system"]["memory_bandwidth"] = 8000000000000
cfg["system"]["memory_capacity"] = 206158430208
cfg["system"]["ramulator_sample_stride"] = 1

cfg["serving"]["max_batch_size"] = batch_per_gpu * num_node * num_device
cfg["serving"]["max_process_token"] = 0

sim = cfg["simulation"]
sim["data"] = "synthesis"
sim["input_len"] = seq_len
sim["output_len"] = 2
sim["precision_byte"] = 1
sim["skewness"] = 0.0
sim["iter"] = 3
sim["injection_rate"] = 0
sim["exit_out_of_memory"] = False
sim["mem_cap_limit"] = False

cfg["log"]["print_log"] = False
cfg["log"]["export_gantt"] = False
cfg["log"]["output_directory"] = str(output_dir.resolve())

config_path.parent.mkdir(parents=True, exist_ok=True)
with config_path.open("w") as f:
    yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
PY

        TIMEOUT=900
        if [[ "$RAMUL" == "on" ]]; then
          TIMEOUT=3600
        fi

        pushd "$BUILD_DIR" >/dev/null
        if ! timeout "$TIMEOUT" ./run "$CONFIG_PATH" >"$LOG_DIR/$LOG_NAME" 2>&1; then
          echo "  [FAIL] see $LOG_DIR/$LOG_NAME"
          popd >/dev/null
          rm -rf "$TMP_RUN_DIR"
          continue
        fi
        popd >/dev/null

        LATEST_CSV="$(find "$TMP_RUN_DIR" -maxdepth 1 -type f -name '*.csv' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
        if [[ -n "$LATEST_CSV" ]]; then
          mv "$LATEST_CSV" "$DATA_DIR/$RESULT_NAME"
          echo "  [OK] saved $RESULT_NAME"
        else
          echo "  [WARN] no CSV output found"
        fi

        rm -rf "$TMP_RUN_DIR"
      done
    done
  done
done

echo ""
echo "==========================================="
echo "Experiments completed."
echo "Results: $DATA_DIR"
echo "Analyze: python3 experiments/exp_mem_arch/analyze_hbm3e.py --all"
echo "==========================================="
