#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_ranking_npu_ascend910.sh [NUM_NPUS] [DATASET] [MODE]
# Example:
#   bash run_ranking_npu_ascend910.sh 8 debug-ranking-adapter train

NUM_NPUS=${1:-8}
DATASET=${2:-debug-ranking-adapter}
MODE=${3:-train}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Load Ascend runtime env if available.
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# Core distributed env
export WORLD_SIZE="${NUM_NPUS}"
export RANK_SIZE="${NUM_NPUS}"
export ACCELERATOR_TYPE="npu"

# Visibility and deterministic options
if [ -z "${ASCEND_VISIBLE_DEVICES:-}" ]; then
  export ASCEND_VISIBLE_DEVICES="$(seq -s, 0 $((NUM_NPUS - 1)))"
fi
export HCCL_WHITELIST_DISABLE=1

# Optional performance/stability knobs
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1

LOG_DIR="${SCRIPT_DIR}/logs/ranking_npu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

CMD=(
  python -m generative_recommenders.dlrm_v3.train.train_ranker
  --dataset "${DATASET}"
  --mode "${MODE}"
  --accelerator npu
)

echo "[INFO] Running command: ${CMD[*]}"
echo "[INFO] WORLD_SIZE=${WORLD_SIZE}, ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES}"
echo "[INFO] Logs: ${LOG_DIR}/train.log"

"${CMD[@]}" 2>&1 | tee "${LOG_DIR}/train.log"