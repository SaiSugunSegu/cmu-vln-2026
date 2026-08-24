#!/usr/bin/env bash
# Shared helpers for the submission-image scripts. Source it; do not execute.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
#   load_env
#   read_vqa
#   resolve_key "$VQA_PROVIDER"
#
# Nothing here prints a secret value.

# provider -> env var that constants.py reads (keep in sync with PROVIDERS there)
declare -A PROVIDER_KEY=(
  [gemini]=GEMINI_API_KEY
  [anthropic]=ANTHROPIC_API_KEY
  [dashscope]=DASHSCOPE_API_KEY
  [openrouter]=OPENROUTER_API_KEY
  [openai]=OPENAI_API_KEY
)

SUBMIT_LIB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VQA_YAML="$SUBMIT_LIB_ROOT/ai_module/src/captioner/config/vqa.yaml"

# Export repo-root .env into the environment. Fatal if it is missing: both callers
# need HF_TOKEN, and a silent skip would bake an image with no credentials.
load_env() {
  if [[ ! -f "$SUBMIT_LIB_ROOT/.env" ]]; then
    echo "repo-root .env is missing (need HF_TOKEN and the VLM provider key)" >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1091
  source "$SUBMIT_LIB_ROOT/.env"
  set +a
}

# Populate VQA_* from vqa.yaml. Values are raw: an absent key reads back empty
# rather than defaulted, so a caller can tell "unset" from "set to cloud".
read_vqa() {
  VQA_VLM_BACKEND=""
  VQA_TARGET_EXTRACT_BACKEND=""
  VQA_PROVIDER=""
  VQA_MODEL=""
  VQA_MODEL_LITE=""
  VQA_BASE_URL=""
  local line
  while IFS= read -r line; do
    case "$line" in
      vlm_backend=*) VQA_VLM_BACKEND="${line#vlm_backend=}" ;;
      target_extract_backend=*) VQA_TARGET_EXTRACT_BACKEND="${line#target_extract_backend=}" ;;
      provider=*) VQA_PROVIDER="${line#provider=}" ;;
      model=*) VQA_MODEL="${line#model=}" ;;
      model_lite=*) VQA_MODEL_LITE="${line#model_lite=}" ;;
      base_url=*) VQA_BASE_URL="${line#base_url=}" ;;
    esac
  done < <(python3 - "$VQA_YAML" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
for key in ("vlm_backend", "target_extract_backend", "provider", "model",
            "model_lite", "base_url"):
    value = cfg.get(key, "")
    print(f"{key}={'' if value is None else value}")
PY
  )
}

# Set KEY_ENV / KEY_VAL for a provider. An unknown provider, or a known one whose
# own variable is empty, falls back to the generic VLM_API_KEY. Call load_env first.
resolve_key() {
  local provider="${1-}"
  KEY_ENV="${PROVIDER_KEY[$provider]:-VLM_API_KEY}"
  KEY_VAL="${!KEY_ENV-}"
  if [[ -z "$KEY_VAL" && -n "${VLM_API_KEY-}" ]]; then
    KEY_ENV="VLM_API_KEY"
    KEY_VAL="$VLM_API_KEY"
  fi
}
