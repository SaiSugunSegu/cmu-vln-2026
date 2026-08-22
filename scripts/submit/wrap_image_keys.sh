#!/usr/bin/env bash
# Bake HF_TOKEN and the vqa.yaml provider key into TAG as ENV (last layer).
# Called by `just up` so the running image is the one that gets pushed.
# Never prints secret values.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/submit/lib.sh
source "$ROOT/scripts/submit/lib.sh"

TAG="iros2026_odyssey:submission"

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

load_env
read_vqa
resolve_key "$VQA_PROVIDER"
key_env="$KEY_ENV"
key_val="$KEY_VAL"

# constants.py reads a missing or unknown backend as cloud, so an unset key here
# still means a key is required.
need_cloud=0
for backend in "${VQA_VLM_BACKEND:-cloud}" "${VQA_TARGET_EXTRACT_BACKEND:-cloud}"; do
  if [[ "$backend" != "local" ]]; then need_cloud=1; fi
done

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
