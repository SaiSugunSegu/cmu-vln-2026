#!/usr/bin/env bash
# Ask Qwen-VL a question about an image inside iros2026_odyssey.
# Usage:
#   ./ai_module/docker/run_qwen_vqa.sh /data/img.png \
#       "How many pillows are on the bed?"
# Paths are container paths (host ./data is mounted at /data).
#
# This reloads the model on every call (~60 s). `just vqa-ask` talks to the
# persistent server instead and is the preferred path.

set -euo pipefail

IMAGE=${1:?image path required}
QUESTION=${2:?question required}
QUANTIZATION=${QUANTIZATION:-int4}
MODEL=${CAPTIONING_MODEL:-qwen3vl}

# printf %q, not '...': a question containing an apostrophe would otherwise end
# the quoted string and hand the rest to the container's shell.
docker exec -it iros2026_odyssey bash -lc "
  source /home/docker/ai_module/install/setup.bash &&
  export PATH=/home/docker/ai_module/install/captioner/lib/captioner:\$PATH &&
  qwen_vqa $(printf '%q' "${IMAGE}") \
    --question $(printf '%q' "${QUESTION}") \
    --captioning_model $(printf '%q' "${MODEL}") \
    --quantization $(printf '%q' "${QUANTIZATION}")
"
