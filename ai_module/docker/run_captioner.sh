#!/usr/bin/env bash
# Helper to caption crops inside iros2026_ai_module.
# Usage:
#   ./ai_module/docker/run_captioner.sh /data/workspace/crops /data/workspace/captions
# Paths are container paths (host ./data is mounted at /data/workspace).

set -euo pipefail

INPUT_DIR=${1:-/data/workspace/crops}
OUTPUT_DIR=${2:-/data/workspace/captions}
BATCH_SIZE=${BATCH_SIZE:-8}
QUANTIZATION=${QUANTIZATION:-int4}
MODEL=${CAPTIONING_MODEL:-qwen3vl}

docker exec -it iros2026_ai_module bash -lc "
  source /home/docker/ai_module/install/setup.bash &&
  export PATH=/home/docker/ai_module/install/captioner/lib/captioner:\$PATH &&
  caption_crops '${INPUT_DIR}' \
    --output_dir '${OUTPUT_DIR}' \
    --captioning_model '${MODEL}' \
    --quantization '${QUANTIZATION}' \
    --batch_size ${BATCH_SIZE}
"
