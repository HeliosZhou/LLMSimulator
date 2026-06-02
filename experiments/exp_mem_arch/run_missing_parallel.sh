#!/bin/bash
# Run only missing exp_mem_arch results in parallel.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
DATA_DIR="$PROJECT_DIR/experiments/exp_mem_arch/data"
LOG_DIR="$PROJECT_DIR/experiments/exp_mem_arch/logs"
WORK_DIR="$PROJECT_DIR/experiments/exp_mem_arch/parallel_work"

mkdir -p "$DATA_DIR" "$LOG_DIR" "$WORK_DIR"

BATCH_SIZES=(32 64 128 256)
SEQ_LENGTHS=(2048 4096 8192)
REORDERING_MODES=("on" "off")
RAMULATOR_MODES=("on" "off")
OUTPUT_LEN=2
PRECISION_BYTE=1
ITER=3

GPU_GEN="B200"
NUM_NODE=4
NUM_DEVICE=8

declare -A MEM_BW
declare -A MEM_CAP
declare -A RAMULATOR_SAMPLE_STRIDE
MEM_BW[hbm3e]="8000000000000"
MEM_CAP[hbm3e]="206158430208"
RAMULATOR_SAMPLE_STRIDE[hbm3e]="1"
MEM_BW[gddr6]="512000000000"
MEM_CAP[gddr6]="206158430208"
RAMULATOR_SAMPLE_STRIDE[gddr6]="4096"
MEM_BW[ddr5]="64000000000"
MEM_CAP[ddr5]="206158430208"
RAMULATOR_SAMPLE_STRIDE[ddr5]="4096"

MEM_TYPES=("hbm3e" "gddr6" "ddr5")

run_one() {
    local mem="$1"
    local reorder="$2"
    local ramul="$3"
    local batch="$4"
    local seq_len="$5"

    local result_name="result_${mem}_b${batch}_l${seq_len}_reorder_${reorder}_ramul_${ramul}.csv"
    local result_path="$DATA_DIR/$result_name"
    if [ -f "$result_path" ]; then
        echo "[SKIP] $result_name"
        return 0
    fi

    local safe_name="${result_name%.csv}"
    local task_work_dir="$WORK_DIR/$safe_name"
    local tmp_run_dir
    tmp_run_dir=$(mktemp -d "$DATA_DIR/.tmp_${safe_name}.XXXXXX")

    rm -rf "$task_work_dir"
    mkdir -p "$task_work_dir"
    cp "$BUILD_DIR/config.yaml" "$task_work_dir/config.yaml"
    cp "$BUILD_DIR"/dram_config*.yaml "$task_work_dir/"
    ln -s "$BUILD_DIR/run" "$task_work_dir/run"

    python3 - "$task_work_dir/config.yaml" "$tmp_run_dir" \
        "$mem" "$reorder" "$ramul" "$batch" "$seq_len" <<'PY'
import sys
import yaml

config_path, output_dir, mem, reorder, ramul, batch, seq_len = sys.argv[1:8]
batch = int(batch)
seq_len = int(seq_len)

mem_bw = {
    "hbm3e": 8000000000000,
    "gddr6": 512000000000,
    "ddr5": 64000000000,
}
mem_cap = {
    "hbm3e": 206158430208,
    "gddr6": 206158430208,
    "ddr5": 206158430208,
}
sample_stride = {
    "hbm3e": 1,
    "gddr6": 4096,
    "ddr5": 4096,
}

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

cfg["model"]["model_name"] = "deepseekV3"
cfg["system"]["gpu_gen"] = "B200"
cfg["system"]["num_node"] = 4
cfg["system"]["num_device"] = 8
cfg["system"]["distribution"]["expert_tensor_degree"] = 1
cfg["system"]["distribution"]["none_expert_tensor_degree"] = 1
cfg["system"]["optimization"]["use_absorb"] = reorder == "on"
cfg["system"]["optimization"]["compressed_kv"] = True
cfg["system"]["optimization"]["use_flash_mla"] = True
cfg["system"]["optimization"]["use_flash_attention"] = True
cfg["system"]["optimization"]["reuse_kv_cache"] = True
cfg["system"]["optimization"]["kv_cache_reuse_rate"] = 0.0
cfg["system"]["optimization"]["parallel_execution"] = False
cfg["system"]["optimization"]["hetero_subbatch"] = False
cfg["system"]["optimization"]["disagg_system"] = False
cfg["system"]["optimization"]["use_low_unit_moe_only"] = False
cfg["system"]["optimization"]["use_ramulator"] = ramul == "on"
cfg["system"]["optimization"]["prefill_mode"] = False
cfg["system"]["optimization"]["decode_mode"] = True
cfg["system"]["processor_type"] = "GPU"
cfg["system"]["memory_bandwidth"] = mem_bw[mem]
cfg["system"]["memory_capacity"] = mem_cap[mem]
cfg["system"]["ramulator_sample_stride"] = sample_stride[mem]
cfg["serving"]["max_batch_size"] = batch * 4 * 8
cfg["serving"]["max_process_token"] = 0
cfg["simulation"]["data"] = "synthesis"
cfg["simulation"]["input_len"] = seq_len
cfg["simulation"]["output_len"] = 2
cfg["simulation"]["precision_byte"] = 1
cfg["simulation"]["skewness"] = 0.0
cfg["simulation"]["iter"] = 3
cfg["simulation"]["injection_rate"] = 0
cfg["simulation"]["exit_out_of_memory"] = False
cfg["simulation"]["mem_cap_limit"] = False
cfg["log"]["print_log"] = False
cfg["log"]["export_gantt"] = False
cfg["log"]["output_directory"] = output_dir

with open(config_path, "w") as f:
    yaml.safe_dump(cfg, f)
PY

    local timeout_sec=600
    if [ "$ramul" = "on" ]; then
        timeout_sec=3600
        if [ "$mem" != "hbm3e" ]; then
            timeout_sec=7200
        fi
    fi

    local log_file="$LOG_DIR/${safe_name}.log"
    echo "[RUN] $result_name"
    if ! (cd "$task_work_dir" && timeout "$timeout_sec" ./run >"$log_file" 2>&1); then
        echo "[FAIL] $result_name"
        rm -rf "$tmp_run_dir" "$task_work_dir"
        return 1
    fi

    local latest_csv
    latest_csv=$(find "$tmp_run_dir" -maxdepth 1 -type f -name '*.csv' -printf '%T@ %p\n' |
        sort -nr | head -1 | cut -d' ' -f2-)
    if [ -z "$latest_csv" ]; then
        echo "[WARN] no CSV output: $result_name"
        rm -rf "$tmp_run_dir" "$task_work_dir"
        return 1
    fi

    mv "$latest_csv" "$result_path"
    rm -rf "$tmp_run_dir" "$task_work_dir"
    echo "[OK] $result_name"
}

if [ "${1:-}" = "--run-one" ]; then
    shift
    run_one "$@"
    exit $?
fi

JOBS="${JOBS:-6}"
TASK_FILE="$WORK_DIR/missing_tasks.txt"
: >"$TASK_FILE"

for mem in "${MEM_TYPES[@]}"; do
    for reorder in "${REORDERING_MODES[@]}"; do
        for ramul in "${RAMULATOR_MODES[@]}"; do
            for seq_len in "${SEQ_LENGTHS[@]}"; do
                for batch in "${BATCH_SIZES[@]}"; do
                    result_name="result_${mem}_b${batch}_l${seq_len}_reorder_${reorder}_ramul_${ramul}.csv"
                    if [ ! -f "$DATA_DIR/$result_name" ]; then
                        printf '%s %s %s %s %s\n' "$mem" "$reorder" "$ramul" "$batch" "$seq_len" >>"$TASK_FILE"
                    fi
                done
            done
        done
    done
done

total_tasks=$(wc -l <"$TASK_FILE")
echo "Missing tasks: $total_tasks"
echo "Parallel jobs: $JOBS"
if [ "$total_tasks" -eq 0 ]; then
    exit 0
fi

xargs -r -n 5 -P "$JOBS" "$0" --run-one <"$TASK_FILE"
