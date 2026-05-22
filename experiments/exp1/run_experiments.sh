#!/bin/bash
# Reproduce Figure 6: Attention Block Latency Breakdown with/without MLA Reordering
# Paper: "Rethinking LLM Inference Bottlenecks"
#
# Sweeps: batch_size (32,64,128,256) x seq_len (2048,4096,8192) x use_absorb (on,off)
# Model: deepseekV3, GPU: B200

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
DATA_DIR="$PROJECT_DIR/experiments/exp1/data"
PLOT_DIR="$PROJECT_DIR/experiments/exp1/plots"

mkdir -p "$DATA_DIR"
mkdir -p "$PLOT_DIR"

BATCH_SIZES=(32 64 128 256)
SEQ_LENGTHS=(2048 4096 8192)
ABSORB_MODES=("on" "off")
OUTPUT_LEN=2
PRECISION_BYTE=1
ITER=3

GPU_GEN="B200"
NUM_NODE=1
NUM_DEVICE=8
NE_TP=1
EXPERT_TP=1

echo "==========================================="
echo "Attention Block Latency Breakdown Experiment"
echo "GPU: $GPU_GEN, Iterations: $ITER"
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "Seq lengths: ${SEQ_LENGTHS[*]}"
echo "Absorb modes: ${ABSORB_MODES[*]}"
echo "==========================================="

for ABSORB in "${ABSORB_MODES[@]}"; do
    for SEQ_LEN in "${SEQ_LENGTHS[@]}"; do
        for BATCH in "${BATCH_SIZES[@]}"; do
            CFG_NAME="cfg_b${BATCH}_l${SEQ_LEN}_absorb_${ABSORB}.yaml"
            RESULT_NAME="result_b${BATCH}_l${SEQ_LEN}_absorb_${ABSORB}.csv"

            echo ""
            echo ">>> [absorb=${ABSORB}] B=${BATCH} L=${SEQ_LEN}"

            # Generate config.yaml
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
cfg['serving']['max_batch_size'] = ${BATCH}
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

            # Run simulation
            pushd "$BUILD_DIR" > /dev/null
            timeout 300 ./run 2>&1 || {
                echo "  [FAIL/OOM] Simulation failed"
                popd > /dev/null
                continue
            }
            popd > /dev/null

            # Find and rename CSV
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

echo ""
echo "==========================================="
echo "All experiments completed."
echo "Results in: $DATA_DIR"
echo "Run: python3 experiments/exp1/run_attention_breakdown.py --plot"
echo "==========================================="
