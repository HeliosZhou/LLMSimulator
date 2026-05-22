#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/experiments/logs"
mkdir -p "$LOG_DIR"

cd "$ROOT_DIR"

echo "[$(date '+%F %T')] Building LLMSimulator"
cmake --build build -j 4

experiments=(
  "exp0:run_tpot_throughput.py"
  "exp2:run_tp_attention.py"
  "exp3:run_throughput_latency.py"
  "exp4:run_interconnect.py"
  "exp5:run_skew.py"
  "exp6:run_pim.py"
)

for item in "${experiments[@]}"; do
  exp="${item%%:*}"
  script="${item##*:}"
  log="$LOG_DIR/${exp}_$(date '+%Y%m%d_%H%M%S').log"
  echo "[$(date '+%F %T')] Starting $exp/$script; log=$log"
  python3 "experiments/$exp/$script" --run --plot --timeout 900 >"$log" 2>&1
  echo "[$(date '+%F %T')] Finished $exp/$script"
done

echo "[$(date '+%F %T')] All incomplete experiments finished"
