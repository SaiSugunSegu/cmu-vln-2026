#!/usr/bin/env bash
# Run a captioner CLI inside the AI container, for use without `just` installed.
#
#   ./ai_module/docker/run_tool.sh caption [INPUT_DIR] [OUTPUT_DIR]
#   ./ai_module/docker/run_tool.sh vqa IMAGE "QUESTION"
#
# Paths are container paths (host ./data is mounted at /data).
#
# `just caption` and `just vqa-ask` are the preferred entry points. Note that vqa
# here reloads the model on every call (~60 s), whereas `just vqa-ask` talks to the
# resident server started by `just vqa-up`.
#
# Env overrides: QUANTIZATION (int4), CAPTIONING_MODEL (qwen3vl),
#                BATCH_SIZE (8, caption only)
set -euo pipefail

CONTAINER="${AI_CONTAINER:-iros2026_odyssey}"
QUANTIZATION="${QUANTIZATION:-int4}"
MODEL="${CAPTIONING_MODEL:-qwen3vl}"

usage() {
  sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
}

# bash -lc does not source ~/.bashrc non-interactively, so the captioner console
# scripts need their install dir put on PATH here.
in_container() {
  docker exec -it "$CONTAINER" bash -lc "
    source /home/docker/ai_module/install/setup.bash &&
    export PATH=/home/docker/ai_module/install/captioner/lib/captioner:\$PATH &&
    $*
  "
}

cmd="${1-}"
[[ $# -gt 0 ]] && shift

case "$cmd" in
  caption)
    input="${1:-/data/crops}"
    output="${2:-/data/captions}"
    batch_size="${BATCH_SIZE:-8}"
    # printf %q, not '...': a path containing a quote or space would otherwise end
    # the quoted string and hand the rest to the container's shell.
    in_container "mkdir -p $(printf '%q' "$output") &&
      caption_crops $(printf '%q' "$input") \
        --output_dir $(printf '%q' "$output") \
        --captioning_model $(printf '%q' "$MODEL") \
        --quantization $(printf '%q' "$QUANTIZATION") \
        --batch_size $(printf '%q' "$batch_size")"
    ;;
  vqa)
    image="${1:?image path required}"
    question="${2:?question required}"
    in_container "qwen_vqa $(printf '%q' "$image") \
      --question $(printf '%q' "$question") \
      --captioning_model $(printf '%q' "$MODEL") \
      --quantization $(printf '%q' "$QUANTIZATION")"
    ;;
  -h|--help)
    usage
    ;;
  *)
    [[ -n "$cmd" ]] && echo "unknown command: $cmd" >&2
    usage >&2
    exit 2
    ;;
esac
