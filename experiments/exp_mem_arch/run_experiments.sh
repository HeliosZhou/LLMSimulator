#!/bin/bash
# Memory Architecture Comparison Experiment with Ramulator
# Compares HBM3E vs GDDR6 vs DDR5 with/without Ramulator and reordering
#
# Sweep: memory_type x reordering x ramulator x batch_size x seq_len

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
DATA_DIR="$PROJECT_DIR/experiments/exp_mem_arch/data"
PLOT_DIR="$PROJECT_DIR/experiments/exp_mem_arch/plots"

mkdir -p "$DATA_DIR" "$PLOT_DIR"

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

# Memory configs: name bandwidth_B/s capacity_bytes
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

echo "==========================================="
echo "Memory Architecture Comparison Experiment"
echo "GPU: $GPU_GEN, Iterations: $ITER"
echo "Memory types: ${MEM_TYPES[*]}"
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "Seq lengths: ${SEQ_LENGTHS[*]}"
echo "Reordering modes: ${REORDERING_MODES[*]}"
echo "Ramulator modes: ${RAMULATOR_MODES[*]}"
echo "Total combinations: $(( ${#MEM_TYPES[@]} * ${#REORDERING_MODES[@]} * ${#RAMULATOR_MODES[@]} * ${#BATCH_SIZES[@]} * ${#SEQ_LENGTHS[@]} ))"
echo "==========================================="

for MEM in "${MEM_TYPES[@]}"; do
    for REORDER in "${REORDERING_MODES[@]}"; do
        for RAMUL in "${RAMULATOR_MODES[@]}"; do
            echo ""
            echo "======================================="
            echo "  Memory type: $MEM"
            echo "  Bandwidth: ${MEM_BW[$MEM]} B/s"
            echo "  Capacity:  ${MEM_CAP[$MEM]} bytes"
            echo "  Reordering: $REORDER"
            echo "  Ramulator: $RAMUL"
            echo "======================================="

            for SEQ_LEN in "${SEQ_LENGTHS[@]}"; do
                for BATCH in "${BATCH_SIZES[@]}"; do
                    RESULT_NAME="result_${MEM}_b${BATCH}_l${SEQ_LEN}_reorder_${REORDER}_ramul_${RAMUL}.csv"

                    if [ -f "$DATA_DIR/$RESULT_NAME" ]; then
                        echo "  [SKIP] $RESULT_NAME already exists"
                        continue
                    fi

                    echo ""
                    echo ">>> [$MEM] reorder=${REORDER} ramul=${RAMUL} B=${BATCH} L=${SEQ_LEN}"

                    TMP_RUN_DIR=$(mktemp -d "$DATA_DIR/.tmp_${RESULT_NAME%.csv}.XXXXXX")

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
cfg['system']['optimization']['use_absorb'] = $([ \"$REORDER\" = \"on\" ] && echo \"True\" || echo \"False\")
cfg['system']['optimization']['compressed_kv'] = True
cfg['system']['optimization']['use_flash_mla'] = True
cfg['system']['optimization']['use_flash_attention'] = True
cfg['system']['optimization']['reuse_kv_cache'] = True
cfg['system']['optimization']['kv_cache_reuse_rate'] = 0.0
cfg['system']['optimization']['parallel_execution'] = False
cfg['system']['optimization']['hetero_subbatch'] = False
cfg['system']['optimization']['disagg_system'] = False
cfg['system']['optimization']['use_low_unit_moe_only'] = False
cfg['system']['optimization']['use_ramulator'] = $([ \"$RAMUL\" = \"on\" ] && echo \"True\" || echo \"False\")
cfg['system']['optimization']['prefill_mode'] = False
cfg['system']['optimization']['decode_mode'] = True
cfg['system']['processor_type'] = 'GPU'
cfg['system']['memory_bandwidth'] = ${MEM_BW[$MEM]}
cfg['system']['memory_capacity'] = ${MEM_CAP[$MEM]}
cfg['system']['ramulator_sample_stride'] = ${RAMULATOR_SAMPLE_STRIDE[$MEM]}
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
cfg['log']['output_directory'] = '${TMP_RUN_DIR}'
with open('${BUILD_DIR}/config.yaml', 'w') as f:
    yaml.dump(cfg, f)
"

                    pushd "$BUILD_DIR" > /dev/null
                    TIMEOUT=600
                    if [ "$RAMUL" = "on" ]; then
                        TIMEOUT=3600
                        if [ "$MEM" != "hbm3e" ]; then
                            TIMEOUT=7200
                        fi
                    fi
                    timeout "$TIMEOUT" ./run 2>&1 || {
                        echo "  [FAIL/OOM] Simulation failed"
                        popd > /dev/null
                        rm -rf "$TMP_RUN_DIR"
                        continue
                    }
                    popd > /dev/null

                    LATEST_CSV=$(find "$TMP_RUN_DIR" -maxdepth 1 -type f -name '*.csv' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
                    if [ -n "$LATEST_CSV" ]; then
                        mv "$LATEST_CSV" "$DATA_DIR/$RESULT_NAME"
                        echo "  [OK] Saved -> $RESULT_NAME"
                    else
                        echo "  [WARN] No CSV output found"
                    fi
                    rm -rf "$TMP_RUN_DIR"
                done
            done
        done
    done
done

echo ""
echo "==========================================="
echo "All experiments completed."
echo "Results in: $DATA_DIR"
echo "Run: python3 experiments/exp_mem_arch/analyze_results.py --plot"
echo "==========================================="
