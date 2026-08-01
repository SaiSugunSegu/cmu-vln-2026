#!/usr/bin/env bash
# Ask Qwen-VL a question about an image inside iros2026_ai_module.
# Usage:
#   ./ai_module/docker/run_qwen_vqa.sh /data/workspace/img.png \
#       "How many pillows are on the bed?"
# Paths are container paths (host ./data is mounted at /data/workspace).

set -euo pipefail

IMAGE=${1:?image path required}
QUESTION=${2:?question required}
QUANTIZATION=${QUANTIZATION:-int4}
MODEL=${CAPTIONING_MODEL:-qwen3vl}

docker exec -it iros2026_ai_module bash -lc "
  source /home/docker/ai_module/install/setup.bash &&
  export PATH=/home/docker/ai_module/install/captioner/lib/captioner:\$PATH &&
  qwen_vqa '${IMAGE}' \
    --question '${QUESTION}' \
    --captioning_model '${MODEL}' \
    --quantization '${QUANTIZATION}'
"
