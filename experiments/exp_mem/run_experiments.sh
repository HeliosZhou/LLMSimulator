#!/bin/bash
# Memory Type Impact Experiment
# Compares HBM3E vs GDDR6 vs DDR5 on B200 with DeepSeek-V3
#
# Sweep: memory_type x batch_size x seq_len x absorb_mode

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
DATA_DIR="$PROJECT_DIR/experiments/exp_mem/data"
PLOT_DIR="$PROJECT_DIR/experiments/exp_mem/plots"

mkdir -p "$DATA_DIR" "$PLOT_DIR"

BATCH_SIZES=(32 64 128 256)
SEQ_LENGTHS=(2048 4096 8192)
ABSORB_MODES=("on" "off")
OUTPUT_LEN=2
PRECISION_BYTE=1
ITER=3

GPU_GEN="B200"
NUM_NODE=1
NUM_DEVICE=8

# Memory configs: name bandwidth_B/s capacity_bytes
declare -A MEM_BW
declare -A MEM_CAP
MEM_BW[hbm3e]="8000000000000"
MEM_CAP[hbm3e]="206158430208"
MEM_BW[gddr6]="512000000000"
MEM_CAP[gddr6]="206158430208"
MEM_BW[ddr5]="64000000000"
MEM_CAP[ddr5]="206158430208"

MEM_TYPES=("hbm3e" "gddr6" "ddr5")

echo "==========================================="
echo "Memory Type Impact Experiment"
echo "GPU: $GPU_GEN, Iterations: $ITER"
echo "Memory types: ${MEM_TYPES[*]}"
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "Seq lengths: ${SEQ_LENGTHS[*]}"
echo "Absorb modes: ${ABSORB_MODES[*]}"
echo "==========================================="

for MEM in "${MEM_TYPES[@]}"; do
    echo ""
    echo "======================================="
    echo "  Memory type: $MEM"
    echo "  Bandwidth: ${MEM_BW[$MEM]} B/s"
    echo "  Capacity:  ${MEM_CAP[$MEM]} bytes"
    echo "======================================="

    for ABSORB in "${ABSORB_MODES[@]}"; do
        for SEQ_LEN in "${SEQ_LENGTHS[@]}"; do
            for BATCH in "${BATCH_SIZES[@]}"; do
                RESULT_NAME="result_${MEM}_b${BATCH}_l${SEQ_LEN}_absorb_${ABSORB}.csv"

                if [ -f "$DATA_DIR/$RESULT_NAME" ]; then
                    echo "  [SKIP] $RESULT_NAME already exists"
                    continue
                fi

                echo ""
                echo ">>> [$MEM] absorb=${ABSORB} B=${BATCH} L=${SEQ_LEN}"

                python3 -c "
import yaml
with open('${BUILD_DIR}/config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['model']['model_name'] = 'deepseekV3'
cfg['system']['gpu_gen'] = '${GPU_GEN}'
cfg['system']['num_node'] = ${NUM_NODE}
cfg['system']['num_device'] = ${NUM_DEVICE}
cfg['system']['distribution']['expert_tensor_degree'] = 1
cfg['system']['distribution']['none_expert_tensor_degree'] = 1
cfg['system']['optimization']['use_absorb'] = $([ \"$ABSORB\" = \"on\" ] && echo \"True\" || echo \"False\")
cfg['system']['optimization']['compressed_kv'] = True
cfg['system']['optimization']['use_flash_mla'] = True
cfg['system']['optimization']['use_flash_attention'] = True
cfg['system']['optimization']['reuse_kv_cache'] = True
cfg['system']['optimization']['kv_cache_reuse_rate'] = 0.0
cfg['system']['optimization']['parallel_execution'] = False
cfg['system']['optimization']['hetero_subbatch'] = False
cfg['system']['optimization']['disagg_system'] = False
cfg['system']['optimization']['use_low_unit_moe_only'] = False
cfg['system']['optimization']['use_ramulator'] = False
cfg['system']['optimization']['prefill_mode'] = False
cfg['system']['optimization']['decode_mode'] = True
cfg['system']['processor_type'] = 'GPU'
cfg['system']['memory_bandwidth'] = ${MEM_BW[$MEM]}
cfg['system']['memory_capacity'] = ${MEM_CAP[$MEM]}
cfg['serving']['max_batch_size'] = $((BATCH * ${NUM_NODE} * ${NUM_DEVICE}))
cfg['serving']['max_process_token'] = 0
cfg['simulation']['data'] = 'synthesis'
cfg['simulation']['input_len'] = ${SEQ_LEN}
cfg['simulation']['output_len'] = ${OUTPUT_LEN}
cfg['simulation']['precision_byte'] = ${PRECISION_BYTE}
cfg['simulation']['skewness'] = 0.0
cfg['simulation']['iter'] = ${ITER}
cfg['simulation']['injection_rate'] = 0
cfg['simulation']['exit_out_of_memory'] = False
cfg['simulation']['mem_cap_limit'] = False
cfg['log']['print_log'] = False
cfg['log']['export_gantt'] = False
cfg['log']['output_directory'] = '${DATA_DIR}'
with open('${BUILD_DIR}/config.yaml', 'w') as f:
    yaml.dump(cfg, f)
"

                pushd "$BUILD_DIR" > /dev/null
                timeout 600 ./run 2>&1 || {
                    echo "  [FAIL/OOM] Simulation failed"
                    popd > /dev/null
                    continue
                }
                popd > /dev/null

                LATEST_CSV=$(ls -t "$DATA_DIR"/*.csv 2>/dev/null | head -1)
                if [ -n "$LATEST_CSV" ] && [ "$(basename "$LATEST_CSV")" != "$RESULT_NAME" ]; then
                    mv "$LATEST_CSV" "$DATA_DIR/$RESULT_NAME"
                    echo "  [OK] Saved -> $RESULT_NAME"
                elif [ -f "$DATA_DIR/$RESULT_NAME" ]; then
                    echo "  [OK] Already exists -> $RESULT_NAME"
                else
                    echo "  [WARN] No CSV output found"
                fi
            done
        done
    done
done

echo ""
echo "==========================================="
echo "All experiments completed."
echo "Results in: $DATA_DIR"
echo "Run: python3 experiments/exp_mem/run_memory_comparison.py --plot"
echo "==========================================="
