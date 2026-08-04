#!/usr/bin/env bash
# Helper to caption crops inside iros2026_ai_module.
# Usage:
#   ./ai_module/docker/run_captioner.sh /data/crops /data/captions
# Paths are container paths (host ./data is mounted at /data).
#
# Equivalent to `just caption crops captions`, for use without just installed.

set -euo pipefail

INPUT_DIR=${1:-/data/crops}
OUTPUT_DIR=${2:-/data/captions}
BATCH_SIZE=${BATCH_SIZE:-8}
QUANTIZATION=${QUANTIZATION:-int4}
MODEL=${CAPTIONING_MODEL:-qwen3vl}

# printf %q, not '...': a path containing a quote or space would otherwise end
# the quoted string and hand the rest to the container's shell.
docker exec -it iros2026_ai_module bash -lc "
  source /home/docker/ai_module/install/setup.bash &&
  export PATH=/home/docker/ai_module/install/captioner/lib/captioner:\$PATH &&
  mkdir -p $(printf '%q' "${OUTPUT_DIR}") &&
  caption_crops $(printf '%q' "${INPUT_DIR}") \
    --output_dir $(printf '%q' "${OUTPUT_DIR}") \
    --captioning_model $(printf '%q' "${MODEL}") \
    --quantization $(printf '%q' "${QUANTIZATION}") \
    --batch_size $(printf '%q' "${BATCH_SIZE}")
"
