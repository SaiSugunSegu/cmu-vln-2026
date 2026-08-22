#!/usr/bin/env bash
# Bake HF_TOKEN and the vqa.yaml provider key into TAG as ENV (last layer).
# Called by `just up` so the running image is the one that gets pushed.
# Never prints secret values.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAG="iros2026_odyssey:submission"

# provider -> env var that constants.py reads (keep in sync with PROVIDERS there)
declare -A PROVIDER_KEY=(
  [gemini]=GEMINI_API_KEY
  [anthropic]=ANTHROPIC_API_KEY
  [dashscope]=DASHSCOPE_API_KEY
  [openrouter]=OPENROUTER_API_KEY
  [openai]=OPENAI_API_KEY
)

usage() {
  cat <<'EOF'
Bake HF_TOKEN and the VLM provider key into an image as ENV.

Usage:
  scripts/submit/wrap_image_keys.sh [--tag NAME]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! docker image inspect "$TAG" >/dev/null 2>&1; then
  echo "image $TAG does not exist — compose build it first" >&2
  exit 1
fi

if [[ ! -f "$ROOT/.env" ]]; then
  echo "repo-root .env is missing (need HF_TOKEN and the VLM provider key)" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
source "$ROOT/.env"
set +a

provider="$(python3 - "$ROOT/ai_module/src/captioner/config/vqa.yaml" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
print(cfg.get("provider") or "")
PY
)"
vlm_backend="$(python3 - "$ROOT/ai_module/src/captioner/config/vqa.yaml" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
print(cfg.get("vlm_backend") or "cloud")
print(cfg.get("target_extract_backend") or "cloud")
PY
)"
need_cloud=0
while IFS= read -r line; do
  if [[ "$line" == "cloud" ]]; then need_cloud=1; fi
done <<< "$vlm_backend"

key_env="${PROVIDER_KEY[$provider]:-VLM_API_KEY}"
key_val="${!key_env-}"
if [[ -z "$key_val" && -n "${VLM_API_KEY-}" ]]; then
  key_env="VLM_API_KEY"
  key_val="$VLM_API_KEY"
fi

if [[ -z "${HF_TOKEN-}" ]]; then
  echo "HF_TOKEN missing in .env — gated facebook/sam3 cannot be baked without it" >&2
  exit 1
fi
if (( need_cloud == 1 )) && [[ -z "$key_val" ]]; then
  echo "a cloud backend is set but neither $key_env nor VLM_API_KEY is in .env" >&2
  exit 1
fi

if (( need_cloud == 0 )) || [[ -z "$key_val" ]]; then
  echo "both backends are local — skipping provider-key wrap (HF_TOKEN already in the image)"
  exit 0
fi

echo "Baking HF_TOKEN + $key_env into $TAG (values not printed)."
echo "docker inspect / docker history on this tag will show the key — keep the Hub repo private."

wrap="$(mktemp -d)"
cat > "$wrap/Dockerfile" <<EOF
FROM ${TAG}
ARG HF_TOKEN
ARG ${key_env}
ENV HF_TOKEN=\${HF_TOKEN}
ENV ${key_env}=\${${key_env}}
EOF
DOCKER_BUILDKIT=1 docker build \
  --progress=plain \
  --build-arg HF_TOKEN="$HF_TOKEN" \
  --build-arg "${key_env}=${key_val}" \
  -t "$TAG" \
  "$wrap"
rm -rf "$wrap"
echo "runtime ENV baked: HF_TOKEN + $key_env → $TAG"
