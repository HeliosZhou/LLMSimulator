#!/usr/bin/env bash
# B200/HBM3E experiment using DRAMSpec-calibrated HBM3E-like parameters.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
EXP_DIR="$PROJECT_DIR/experiments/exp_mem_arch_dramspec"
DATA_DIR="$EXP_DIR/data"
CONFIG_DIR="$EXP_DIR/configs"
LOG_DIR="$EXP_DIR/logs"

mkdir -p "$DATA_DIR" "$CONFIG_DIR" "$LOG_DIR"

python3 "$EXP_DIR/generate_dramspec_configs.py"

BATCH_SIZES=(${BATCH_SIZES:-32})
SEQ_LENGTHS=(${SEQ_LENGTHS:-2048})
REORDERING_MODES=(${REORDERING_MODES:-on})
RAMULATOR_MODES=(${RAMULATOR_MODES:-on})

OUTPUT_LEN="${OUTPUT_LEN:-2}"
PRECISION_BYTE="${PRECISION_BYTE:-1}"
ITER="${ITER:-3}"

GPU_GEN="B200"
NUM_NODE=4
NUM_DEVICE=8
MEMORY_TYPE="hbm3e_dramspec"
MEMORY_BW=8000000000000
MEMORY_CAP=206158430208
RAMULATOR_SAMPLE_STRIDE="${RAMULATOR_SAMPLE_STRIDE:-1}"
DRAM_CONFIG_PATH="$EXP_DIR/generated/dram_config_HBM3E_DRAMSpec.yaml"
DRAM_POWER_CONFIG_PATH="$EXP_DIR/generated/dramspec_hbm3e_like_power.yaml"
FORCE_RERUN="${FORCE_RERUN:-0}"

TOTAL=$(( ${#REORDERING_MODES[@]} * ${#SEQ_LENGTHS[@]} * ${#BATCH_SIZES[@]} * ${#RAMULATOR_MODES[@]} ))

echo "==========================================="
echo "DRAMSpec-calibrated HBM3E-like experiment"
echo "GPU: $GPU_GEN, nodes: $NUM_NODE, devices/node: $NUM_DEVICE"
echo "Memory target: HBM3E, bandwidth: $MEMORY_BW B/s, capacity: $MEMORY_CAP bytes"
echo "Batches/GPU: ${BATCH_SIZES[*]}"
echo "Seq lengths: ${SEQ_LENGTHS[*]}"
echo "Reordering modes: ${REORDERING_MODES[*]}"
echo "Ramulator modes: ${RAMULATOR_MODES[*]}"
echo "Total combinations: $TOTAL"
echo "Force rerun: $FORCE_RERUN"
echo "==========================================="

for REORDER in "${REORDERING_MODES[@]}"; do
  for SEQ_LEN in "${SEQ_LENGTHS[@]}"; do
    for BATCH in "${BATCH_SIZES[@]}"; do
      for RAMUL in "${RAMULATOR_MODES[@]}"; do
        RESULT_NAME="result_${MEMORY_TYPE}_b${BATCH}_l${SEQ_LEN}_reorder_${REORDER}_ramul_${RAMUL}.csv"
        CONFIG_NAME="${RESULT_NAME%.csv}.yaml"
        LOG_NAME="${RESULT_NAME%.csv}.log"

        if [[ -f "$DATA_DIR/$RESULT_NAME" && "$FORCE_RERUN" != "1" ]]; then
          echo "[SKIP] $RESULT_NAME already exists"
          continue
        fi
        if [[ -f "$DATA_DIR/$RESULT_NAME" && "$FORCE_RERUN" == "1" ]]; then
          echo "[RERUN] Removing previous $RESULT_NAME"
          rm -f "$DATA_DIR/$RESULT_NAME"
        fi

        echo ""
        echo ">>> DRAMSpec HBM3E-like reorder=${REORDER} ramulator=${RAMUL} batch/GPU=${BATCH} seq=${SEQ_LEN}"

        TMP_RUN_DIR="$(mktemp -d "$DATA_DIR/.tmp_${RESULT_NAME%.csv}.XXXXXX")"
        CONFIG_PATH="$CONFIG_DIR/$CONFIG_NAME"

        python3 - "$BUILD_DIR/config.yaml" "$CONFIG_PATH" "$TMP_RUN_DIR" \
          "$REORDER" "$RAMUL" "$BATCH" "$SEQ_LEN" "$OUTPUT_LEN" "$PRECISION_BYTE" "$ITER" \
          "$MEMORY_BW" "$MEMORY_CAP" "$RAMULATOR_SAMPLE_STRIDE" "$DRAM_CONFIG_PATH" "$DRAM_POWER_CONFIG_PATH" <<'PY'
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
output_len = int(sys.argv[8])
precision_byte = int(sys.argv[9])
iters = int(sys.argv[10])
memory_bw = float(sys.argv[11])
memory_cap = float(sys.argv[12])
sample_stride = int(sys.argv[13])
dram_config_path = sys.argv[14]
dram_power_config_path = sys.argv[15]

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
cfg["system"]["memory_bandwidth"] = memory_bw
cfg["system"]["memory_capacity"] = memory_cap
cfg["system"]["ramulator_sample_stride"] = sample_stride
cfg["system"]["dram_config_path"] = dram_config_path
cfg["system"]["memory_scale_factor"] = 0.5

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
opt["use_drampower"] = True
opt["dram_power_model"] = "dramspec_hbm3e_like"
opt["dram_power_config_path"] = dram_power_config_path
opt["prefill_mode"] = False
opt["decode_mode"] = True

cfg["system"]["processor_type"] = "GPU"
cfg["serving"]["max_batch_size"] = batch_per_gpu * num_node * num_device
cfg["serving"]["max_process_token"] = 0

sim = cfg["simulation"]
sim["data"] = "synthesis"
sim["input_len"] = seq_len
sim["output_len"] = output_len
sim["precision_byte"] = precision_byte
sim["skewness"] = 0.0
sim["iter"] = iters
sim["injection_rate"] = 0
sim["exit_out_of_memory"] = False
sim["mem_cap_limit"] = False

cfg["log"]["print_log"] = False
cfg["log"]["export_gantt"] = False
cfg["log"]["output_directory"] = str(output_dir.resolve())

config_path.parent.mkdir(parents=True, exist_ok=True)
with config_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

        set +e
        (cd "$BUILD_DIR" && ./run "$CONFIG_PATH") >"$LOG_DIR/$LOG_NAME" 2>&1
        rc=$?
        set -e
        if [[ $rc -ne 0 ]]; then
          echo "[FAIL] $RESULT_NAME (see $LOG_DIR/$LOG_NAME)" >&2
          rm -rf "$TMP_RUN_DIR"
          exit $rc
        fi

        CSV_COUNT="$(find "$TMP_RUN_DIR" -maxdepth 1 -name '*.csv' | wc -l)"
        if [[ "$CSV_COUNT" -ne 1 ]]; then
          echo "[FAIL] Expected exactly one CSV in $TMP_RUN_DIR, found $CSV_COUNT" >&2
          find "$TMP_RUN_DIR" -maxdepth 1 -type f -print >&2
          rm -rf "$TMP_RUN_DIR"
          exit 1
        fi
        mv "$(find "$TMP_RUN_DIR" -maxdepth 1 -name '*.csv' -print -quit)" "$DATA_DIR/$RESULT_NAME"
        rm -rf "$TMP_RUN_DIR"
        echo "[OK] $RESULT_NAME"
      done
    done
  done
done

python3 "$EXP_DIR/analyze_dramspec.py" --all
