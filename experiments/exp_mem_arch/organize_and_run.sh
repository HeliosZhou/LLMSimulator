#!/bin/bash
# Organize existing data and run missing experiments
# Missing: GDDR6/DDR5 × reordering × Ramulator ON

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
DATA_DIR="$PROJECT_DIR/experiments/exp_mem_arch/data"

mkdir -p "$DATA_DIR"

echo "==========================================="
echo "Step 1: Copy existing data from exp_mem and exp1"
echo "==========================================="

# Copy from exp_mem (Ramulator OFF, all memory types)
echo "Copying from exp_mem (Ramulator OFF)..."
for f in "$PROJECT_DIR/experiments/exp_mem/data"/result_*.csv; do
    fname=$(basename "$f")
    # Extract parts: result_{mem}_b{B}_l{L}_absorb_{on|off}.csv
    # Convert to: result_{mem}_b{B}_l{L}_reorder_{on|off}_ramul_off.csv
    new_name=$(echo "$fname" | sed 's/_absorb_/_reorder_/' | sed 's/\.csv$/_ramul_off.csv/')
    if [ ! -f "$DATA_DIR/$new_name" ]; then
        cp "$f" "$DATA_DIR/$new_name"
        echo "  Copied: $fname -> $new_name"
    fi
done

# Copy from exp1 (HBM3E, Ramulator OFF)
echo "Copying from exp1/data (HBM3E, Ramulator OFF)..."
for f in "$PROJECT_DIR/experiments/exp1/data"/result_b*.csv; do
    fname=$(basename "$f")
    # Convert: result_b{B}_l{L}_absorb_{on|off}.csv -> result_hbm3e_b{B}_l{L}_reorder_{on|off}_ramul_off.csv
    new_name=$(echo "$fname" | sed 's/^result_/result_hbm3e_/' | sed 's/_absorb_/_reorder_/' | sed 's/\.csv$/_ramul_off.csv/')
    if [ ! -f "$DATA_DIR/$new_name" ]; then
        cp "$f" "$DATA_DIR/$new_name"
        echo "  Copied: $fname -> $new_name"
    fi
done

# Copy from exp1/data_ramulator (HBM3E, Ramulator ON)
echo "Copying from exp1/data_ramulator (HBM3E, Ramulator ON)..."
for f in "$PROJECT_DIR/experiments/exp1/data_ramulator"/result_b*.csv; do
    fname=$(basename "$f")
    # Convert: result_b{B}_l{L}_absorb_{on|off}.csv -> result_hbm3e_b{B}_l{L}_reorder_{on|off}_ramul_on.csv
    new_name=$(echo "$fname" | sed 's/^result_/result_hbm3e_/' | sed 's/_absorb_/_reorder_/' | sed 's/\.csv$/_ramul_on.csv/')
    if [ ! -f "$DATA_DIR/$new_name" ]; then
        cp "$f" "$DATA_DIR/$new_name"
        echo "  Copied: $fname -> $new_name"
    fi
done

echo ""
echo "==========================================="
echo "Step 2: Run missing experiments"
echo "Missing: GDDR6/DDR5 × reordering × Ramulator ON"
echo "==========================================="

BATCH_SIZES=(32 64 128 256)
SEQ_LENGTHS=(2048 4096 8192)
REORDERING_MODES=("on" "off")
OUTPUT_LEN=2
PRECISION_BYTE=1
ITER=3

GPU_GEN="B200"
NUM_NODE=1
NUM_DEVICE=8

declare -A MEM_BW
declare -A MEM_CAP
declare -A RAMULATOR_SAMPLE_STRIDE
MEM_BW[gddr6]="512000000000"
MEM_CAP[gddr6]="206158430208"
RAMULATOR_SAMPLE_STRIDE[gddr6]="4096"
MEM_BW[ddr5]="64000000000"
MEM_CAP[ddr5]="206158430208"
RAMULATOR_SAMPLE_STRIDE[ddr5]="4096"

MEM_TYPES=("gddr6" "ddr5")

for MEM in "${MEM_TYPES[@]}"; do
    for REORDER in "${REORDERING_MODES[@]}"; do
        echo ""
        echo "======================================="
        echo "  Memory type: $MEM (Ramulator ON)"
        echo "  Reordering: $REORDER"
        echo "======================================="

        for SEQ_LEN in "${SEQ_LENGTHS[@]}"; do
            for BATCH in "${BATCH_SIZES[@]}"; do
                RESULT_NAME="result_${MEM}_b${BATCH}_l${SEQ_LEN}_reorder_${REORDER}_ramul_on.csv"

                if [ -f "$DATA_DIR/$RESULT_NAME" ]; then
                    echo "  [SKIP] $RESULT_NAME already exists"
                    continue
                fi

                echo ""
                echo ">>> [$MEM] reorder=${REORDER} ramul=on B=${BATCH} L=${SEQ_LEN}"

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
cfg['system']['optimization']['use_ramulator'] = True
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
echo "Total files: $(ls -1 "$DATA_DIR"/result_*.csv 2>/dev/null | wc -l)"
echo "==========================================="
