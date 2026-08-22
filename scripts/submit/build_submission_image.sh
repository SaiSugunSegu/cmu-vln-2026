#!/usr/bin/env bash
# Check the AI-module image the organizers will run the way eval will:
# no source mount, no /data mount, no host HF cache. Weights and API keys have to
# live in the image. `just up` is what bakes them.
#
#   just up
#   just build-submission-image
#   just trial-submission-image
#   just push-submission-image YOURHUBUSER/cmu-vln-ai:v1
#
# Official eval entry: ros2 launch dummy_vlm dummy_vlm.launch
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TAG="iros2026_odyssey:submission"
PLATFORM="linux/amd64"
DO_BUILD=1
DO_PUSH=0
DO_TRIAL=1
DRY_RUN=0

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
Check the submission AI image (weights and keys baked by `just up`).

Usage:
  just up
  just build-submission-image
  just trial-submission-image
  just push-submission-image YOURHUBUSER/cmu-vln-ai:v1

  scripts/submit/build_submission_image.sh [options]

Options:
  --tag NAME          Image tag (default: iros2026_odyssey:submission)
  --platform NAME     docker build --platform (default: linux/amd64)
  --skip-build        Check an image that already exists; do not rebuild
  --no-trial          Skip the live OpenRouter / provider trial call
  --push              docker push TAG after checks pass (TAG must be registry/name)
  --dry-run           Host-side checks only; do not build or touch Docker images
  -h, --help          This help

Reads repo-root .env for HF_TOKEN and the VLM key. Never prints secret values.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="${2:?}"; shift 2 ;;
    --platform) PLATFORM="${2:?}"; shift 2 ;;
    --skip-build) DO_BUILD=0; shift ;;
    --no-trial) DO_TRIAL=0; shift ;;
    --push) DO_PUSH=1; shift ;;
    --dry-run) DRY_RUN=1; DO_BUILD=0; DO_PUSH=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# -- colours / logging -------------------------------------------------------

if [[ -t 1 ]]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'
  YELLOW=$'\033[33m'; BLUE=$'\033[34m'; RESET=$'\033[0m'
else
  BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; RESET=""
fi

STEP=0
FAILS=0
WARNS=0
PASS_NOTES=()
FAIL_NOTES=()
WARN_NOTES=()

banner() {
  echo
  echo "${BOLD}${BLUE}════════════════════════════════════════════════════════════${RESET}"
  echo "${BOLD}${BLUE}  $*${RESET}"
  echo "${BOLD}${BLUE}════════════════════════════════════════════════════════════${RESET}"
}

step() {
  STEP=$((STEP + 1))
  echo
  echo "${BOLD}── [$STEP] $* ──${RESET}"
}

ok() {
  echo "  ${GREEN}PASS${RESET}  $*"
  PASS_NOTES+=("$*")
}

warn() {
  WARNS=$((WARNS + 1))
  echo "  ${YELLOW}WARN${RESET}  $*"
  WARN_NOTES+=("$*")
}

fail() {
  FAILS=$((FAILS + 1))
  echo "  ${RED}FAIL${RESET}  $*"
  FAIL_NOTES+=("$*")
}

info() { echo "  ${DIM}$*${RESET}"; }

die() {
  echo "${RED}ERROR:${RESET} $*" >&2
  exit 1
}

mask() {
  # prefix + length only — never the secret
  local value="${1-}"
  local n=${#value}
  if (( n == 0 )); then
    echo "(empty)"
  elif (( n <= 8 )); then
    echo "${value:0:2}… (len=$n)"
  else
    echo "${value:0:7}… (len=$n)"
  fi
}

# -- .env --------------------------------------------------------------------

load_env() {
  if [[ ! -f "$ROOT/.env" ]]; then
    die "repo-root .env is missing (need HF_TOKEN and the VLM provider key)"
  fi
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
}

# -- host-side parsers -------------------------------------------------------

parse_vqa() {
  python3 - "$ROOT/ai_module/src/captioner/config/vqa.yaml" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
for key in ("vlm_backend", "target_extract_backend", "provider", "model",
            "model_lite", "base_url"):
    value = cfg.get(key, "")
    print(f"{key}={'' if value is None else value}")
PY
}


# -- checks ------------------------------------------------------------------

preflight() {
  banner "Submission image — preflight"
  info "repo   $ROOT"
  info "tag    $TAG"
  info "plat   $PLATFORM"
  info "mode   build=$DO_BUILD push=$DO_PUSH trial=$DO_TRIAL dry_run=$DRY_RUN"

  step "Host tools"
  command -v docker >/dev/null && ok "docker $(docker version --format '{{.Client.Version}}' 2>/dev/null || echo present)" || fail "docker is not on PATH"
  command -v python3 >/dev/null && ok "python3 $(python3 --version | awk '{print $2}')" || fail "python3 is not on PATH"
  python3 -c "import yaml" 2>/dev/null && ok "PyYAML available" || fail "python3 -c 'import yaml' failed (need PyYAML for vqa.yaml)"

  step "vqa.yaml (what the image will ship)"
  local line vlm_backend="" target_extract_backend="" provider="" model="" model_lite="" base_url=""
  while IFS= read -r line; do
    case "$line" in
      vlm_backend=*) vlm_backend="${line#vlm_backend=}" ;;
      target_extract_backend=*) target_extract_backend="${line#target_extract_backend=}" ;;
      provider=*) provider="${line#provider=}" ;;
      model=*) model="${line#model=}" ;;
      model_lite=*) model_lite="${line#model_lite=}" ;;
      base_url=*) base_url="${line#base_url=}" ;;
    esac
  done < <(parse_vqa)
  info "vlm_backend=$vlm_backend  target_extract_backend=$target_extract_backend"
  info "provider=$provider  model=$model"
  info "model_lite=${model_lite:-<same as model>}  base_url=${base_url:-<provider preset>}"
  [[ "$vlm_backend" == "cloud" || "$vlm_backend" == "local" ]] \
    && ok "vlm_backend is $vlm_backend" \
    || fail "vlm_backend must be set to cloud or local, got '$vlm_backend'"
  [[ "$target_extract_backend" == "cloud" || "$target_extract_backend" == "local" ]] \
    && ok "target_extract_backend is $target_extract_backend" \
    || fail "target_extract_backend must be set to cloud or local, got '$target_extract_backend'"
  [[ -n "$provider" ]] && ok "provider is $provider" || fail "vqa.yaml provider is empty"
  [[ -n "$model" ]] && ok "model is $model" || fail "vqa.yaml model is empty (OpenRouter has no preset default)"

  step "Official entry (dummy_vlm.launch → smart_vlm.launch; backend from vqa.yaml)"
  [[ -f "$ROOT/ai_module/src/dummy_vlm/launch/dummy_vlm.launch" ]] \
    && ok "dummy_vlm.launch is present (official entry)" \
    || fail "dummy_vlm.launch missing"
  grep -q 'local_vqa_launch_flag' "$ROOT/ai_module/src/smart_vlm/launch/smart_vlm.launch" \
    && ok "smart_vlm.launch reads NEED_LOCAL_VQA from vqa.yaml" \
    || fail "smart_vlm.launch no longer derives local_vqa from vqa.yaml"

  step "Secrets in .env (values masked)"
  load_env
  local key_env="${PROVIDER_KEY[$provider]:-VLM_API_KEY}"
  local key_val="${!key_env-}"
  if [[ -z "$key_val" && -n "${VLM_API_KEY-}" ]]; then
    key_env="VLM_API_KEY"
    key_val="$VLM_API_KEY"
  fi
  if [[ -n "${HF_TOKEN-}" ]]; then
    ok "HF_TOKEN is set  $(mask "$HF_TOKEN")"
  else
    fail "HF_TOKEN missing — gated facebook/sam3 cannot be baked without it"
  fi
  if [[ "$vlm_backend" == "cloud" || "$target_extract_backend" == "cloud" ]]; then
    if [[ -n "$key_val" ]]; then
      ok "$key_env is set  $(mask "$key_val")"
    else
      fail "a cloud backend is set but neither $key_env nor VLM_API_KEY is in .env"
    fi
  else
    info "both backends are local — VLM API key not required (Qwen runs in-image)"
  fi
  # exported for later steps
  SUB_BACKEND="$vlm_backend"
  SUB_EXTRACT="$target_extract_backend"
  SUB_PROVIDER="$provider"
  SUB_MODEL="$model"
  SUB_KEY_ENV="$key_env"
  SUB_KEY_VAL="$key_val"

  step "Git hygiene — secrets must not be in the public tree"
  if git -C "$ROOT" check-ignore -q .env; then
    ok ".env is gitignored"
  else
    fail ".env is NOT gitignored — do not push this clone"
  fi
  if git -C "$ROOT" diff --cached --name-only | grep -qx '.env'; then
    fail ".env is staged for commit"
  else
    ok ".env is not staged"
  fi
  local hits
  hits="$(git -C "$ROOT" grep -nE 'sk-or-|sk-ant-|AIza[0-9A-Za-z_-]{20,}|hf_[A-Za-z0-9]{20,}' \
    -- ':!.env' ':!*.md' 2>/dev/null || true)"
  if [[ -n "$hits" ]]; then
    fail "tracked file looks like it contains a live token prefix"
    echo "$hits" | sed 's/^/         /'
  else
    ok "no live token prefixes in tracked source"
  fi

  if (( FAILS > 0 )); then
    echo
    echo "${RED}${BOLD}Preflight failed ($FAILS). Fix the FAIL lines before building.${RESET}"
    exit 1
  fi
}

build_image() {
  step "compose build + key wrap (same path as just up)"
  info "image   $TAG"
  info "this is the eval image: ai_module is COPY'd, not bind-mounted"
  echo
  local extra=()
  if [[ -f "$ROOT/.env" ]]; then extra+=(--env-file "$ROOT/.env"); fi
  (cd "$ROOT/docker" && docker compose "${extra[@]}" -f compose_gpu.yml build)
  "$ROOT/scripts/submit/wrap_image_keys.sh" --tag "$TAG"
  ok "bake finished → $TAG"
}

# Run a command in the submission image with no host mounts (eval-shaped).
in_image() {
  docker run --rm --network host --entrypoint bash "$TAG" -lc "$*"
}

check_image() {
  banner "Checks against $TAG  (no mounts — this is how eval sees it)"

  step "Image identity"
  if ! docker image inspect "$TAG" >/dev/null 2>&1; then
    fail "image $TAG does not exist (build it, or pass --tag)"
    return
  fi
  local size arch created
  size="$(docker image inspect "$TAG" --format '{{.Size}}')"
  arch="$(docker image inspect "$TAG" --format '{{.Architecture}}')"
  created="$(docker image inspect "$TAG" --format '{{.Created}}')"
  info "created  $created"
  info "size     $(numfmt --to=iec --suffix=B "$size" 2>/dev/null || echo "$size bytes")"
  info "arch     $arch"
  [[ "$arch" == "amd64" ]] && ok "architecture is amd64 (eval NUC)" || fail "architecture is $arch — eval NUC is x86_64/amd64"

  step "Baked Hugging Face weights"
  local hub_listing
  hub_listing="$(in_image 'ls /home/docker/.cache/huggingface/hub 2>/dev/null || true')"
  if [[ -z "$hub_listing" ]]; then
    fail "HF hub cache is empty — just up with HF_TOKEN in .env bakes weights into the image"
  else
    info "hub entries:"
    echo "$hub_listing" | sed 's/^/         /'
  fi
  echo "$hub_listing" | grep -q 'models--facebook--sam3' \
    && ok "facebook/sam3 is in the image" \
    || fail "facebook/sam3 missing — rebuild with --build-arg HF_TOKEN"
  if [[ "$SUB_BACKEND" == "local" || "$SUB_EXTRACT" == "local" ]]; then
    echo "$hub_listing" | grep -q 'models--Qwen--Qwen3-VL' \
      && ok "Qwen3-VL is in the image (a backend is local)" \
      || fail "Qwen3-VL missing and a vqa.yaml backend is local"
  else
    if echo "$hub_listing" | grep -q 'models--Qwen--Qwen3-VL'; then
      ok "Qwen3-VL is also baked (unused while both backends are cloud)"
    else
      info "Qwen3-VL not in image — fine while both backends are cloud"
    fi
  fi

  step "vqa.yaml + launch inside the image"
  local img_yaml
  img_yaml="$(in_image 'source /home/docker/ai_module/install/setup.bash && python3 -c "
from captioner.vlm_backends.config import load_vqa_config
c = load_vqa_config()
print(\"vlm_backend=\" + str(c.get(\"vlm_backend\",\"\")))
print(\"target_extract_backend=\" + str(c.get(\"target_extract_backend\",\"\")))
print(\"provider=\" + str(c.get(\"provider\",\"\")))
print(\"model=\" + str(c.get(\"model\",\"\")))
"')"
  echo "$img_yaml" | sed 's/^/         /'
  echo "$img_yaml" | grep -qx "vlm_backend=$SUB_BACKEND" && ok "image vlm_backend=$SUB_BACKEND" || fail "image vlm_backend does not match host ($SUB_BACKEND)"
  echo "$img_yaml" | grep -qx "target_extract_backend=$SUB_EXTRACT" && ok "image target_extract_backend=$SUB_EXTRACT" || fail "image target_extract_backend does not match host"
  echo "$img_yaml" | grep -qx "provider=$SUB_PROVIDER" && ok "image provider=$SUB_PROVIDER" || fail "image vqa.yaml provider does not match host"
  echo "$img_yaml" | grep -qx "model=$SUB_MODEL" && ok "image model=$SUB_MODEL" || fail "image vqa.yaml model does not match host"

  in_image 'source /home/docker/ai_module/install/setup.bash && python3 -c "
from captioner.vlm_backends.constants import VLM_BACKEND, TARGET_EXTRACT_BACKEND, NEED_LOCAL_VQA, local_vqa_launch_flag
print(\"VLM_BACKEND=\" + VLM_BACKEND)
print(\"TARGET_EXTRACT_BACKEND=\" + TARGET_EXTRACT_BACKEND)
print(\"NEED_LOCAL_VQA=\" + str(NEED_LOCAL_VQA))
print(\"local_vqa=\" + local_vqa_launch_flag())
"' | sed 's/^/         /'
  in_image 'grep -q local_vqa_launch_flag /home/docker/ai_module/src/smart_vlm/launch/smart_vlm.launch' \
    && ok "image launch derives local_vqa from vqa.yaml" \
    || fail "image launch is not reading NEED_LOCAL_VQA"

  step "Runtime ENV in the image (masked)"
  local env_dump
  env_dump="$(in_image 'python3 -c "
import os
for name in (\"HF_TOKEN\", \"VLM_API_KEY\", \"OPENROUTER_API_KEY\", \"GEMINI_API_KEY\",
             \"DASHSCOPE_API_KEY\", \"ANTHROPIC_API_KEY\", \"OPENAI_API_KEY\"):
    v = os.environ.get(name, \"\")
    print(f\"{name}={len(v)}\")
"')"
  while IFS= read -r line; do
    local name="${line%=*}" n="${line#*=}"
    if [[ "$n" != "0" ]]; then
      ok "$name is set in the image (len=$n)"
    else
      info "$name unset"
    fi
  done <<< "$env_dump"
  if [[ "$SUB_BACKEND" == "cloud" || "$SUB_EXTRACT" == "cloud" ]]; then
    echo "$env_dump" | grep -Eq "^(${SUB_KEY_ENV}|VLM_API_KEY)=[1-9]" \
      && ok "cloud key is present as ENV" \
      || fail "a cloud backend is set but $SUB_KEY_ENV is empty in the image — just up bakes it from .env"
  fi
  echo "$env_dump" | grep -Eq '^HF_TOKEN=[1-9]' \
    && ok "HF_TOKEN is present as ENV (gated SAM 3 at runtime)" \
    || warn "HF_TOKEN empty in image — fine if sam3 is already fully baked"

  step "ROS install + official launch file"
  in_image 'source /home/docker/ai_module/install/setup.bash && ros2 pkg prefix dummy_vlm >/dev/null && ros2 pkg prefix smart_vlm >/dev/null && ros2 pkg prefix sam_mapper >/dev/null && ros2 pkg prefix captioner >/dev/null' \
    && ok "dummy_vlm / smart_vlm / sam_mapper / captioner are installed" \
    || fail "colcon install is incomplete"
  in_image 'source /home/docker/ai_module/install/setup.bash && test -f "$(ros2 pkg prefix dummy_vlm)/share/dummy_vlm/launch/dummy_vlm.launch"' \
    && ok "ros2 can find dummy_vlm.launch" \
    || fail "dummy_vlm.launch is not in the install share"

  step "Import chain (same check the Dockerfile runs at build)"
  local imports
  if imports="$(in_image 'source /home/docker/ai_module/install/setup.bash && python3 -c "
from transformers import Sam3VideoModel
from sam_mapper.object_mapper import ObjMapper
from captioner.vlm_backends.openai_backend import OpenAIBackend
from captioner.vlm_backends.constants import VLM_BACKEND, VLM_PROVIDER, MODEL_NAME
from smart_vlm.numerical_utils import EXTRACT_SYSTEM
print(\"VLM_BACKEND=\" + VLM_BACKEND)
print(\"VLM_PROVIDER=\" + VLM_PROVIDER)
print(\"MODEL_NAME=\" + MODEL_NAME)
print(\"imports_ok\")
"')"; then
    echo "$imports" | sed 's/^/         /'
    echo "$imports" | grep -q imports_ok && ok "Python import chain is healthy" || fail "import check printed no imports_ok"
  else
    echo "$imports" | sed 's/^/         /' || true
    fail "import check failed inside the image"
  fi
}

trial_call() {
  [[ "$DO_TRIAL" == "1" ]] || { info "trial skipped (--no-trial)"; return 0; }
  [[ "$SUB_BACKEND" == "cloud" ]] || { info "trial skipped (vlm_backend=local)"; return 0; }

  step "Live trial: $SUB_PROVIDER / $SUB_MODEL from inside the image"
  info "question: A bed has two pillows and one blanket. How many pillows are on the bed?"
  local reply
  if reply="$(in_image 'source /home/docker/ai_module/install/setup.bash && python3 -c "
import json, os, urllib.request, urllib.error
from captioner.vlm_backends.constants import MODEL_NAME, VLM_API_KEY, VLM_BASE_URL

if not VLM_API_KEY:
    raise SystemExit(\"no VLM key in image environment\")
body = {
    \"model\": MODEL_NAME,
    \"messages\": [{\"role\": \"user\", \"content\":
        \"A bed has two pillows and one blanket. How many pillows are on the bed? Reply with one integer only.\"}],
    \"max_tokens\": 1024,
}
req = urllib.request.Request(
    VLM_BASE_URL.rstrip(\"/\") + \"/chat/completions\",
    data=json.dumps(body).encode(),
    headers={
        \"Authorization\": \"Bearer \" + VLM_API_KEY,
        \"Content-Type\": \"application/json\",
        \"HTTP-Referer\": \"https://github.com/SaiSugunSegu/cmu-vln-2026\",
        \"X-Title\": \"cmu-vln-submission-check\",
    },
    method=\"POST\",
)
try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode())
        status = resp.status
except urllib.error.HTTPError as exc:
    print(\"http_status=\" + str(exc.code))
    print(exc.read().decode()[:500])
    raise SystemExit(1)
choice = (payload.get(\"choices\") or [{}])[0]
content = ((choice.get(\"message\") or {}).get(\"content\") or \"\").strip()
print(\"http_status=\" + str(status))
print(\"model=\" + str(payload.get(\"model\")))
print(\"finish=\" + str(choice.get(\"finish_reason\")))
print(\"content=\" + content)
"')"; then
    echo "$reply" | sed 's/^/         /'
    echo "$reply" | grep -q 'http_status=200' && ok "provider accepted the baked key (HTTP 200)" || fail "trial HTTP was not 200"
    echo "$reply" | grep -Eq 'content=.*2' && ok "model answered the trial count (2)" || warn "trial reply did not contain 2 — read content above"
  else
    echo "$reply" | sed 's/^/         /' || true
    fail "trial call from inside the image failed"
  fi
}

push_image() {
  # `return` with no status inherits the failed `[[`, which trips `set -e`.
  [[ "$DO_PUSH" == "1" ]] || return 0
  step "docker push $TAG"
  if [[ "$TAG" != */* ]]; then
    fail "refusing to push '$TAG' — it has no registry/user prefix (pass --tag YOURHUBUSER/cmu-vln-ai:submission)"
    return 0
  fi
  docker push "$TAG"
  ok "pushed $TAG"
}

summary() {
  banner "Summary"
  echo "  tag                     $TAG"
  echo "  vlm_backend             $SUB_BACKEND"
  echo "  target_extract_backend  $SUB_EXTRACT"
  echo "  provider  $SUB_PROVIDER"
  echo "  model     $SUB_MODEL"
  echo "  key env   $SUB_KEY_ENV"
  echo
  echo "  ${GREEN}passed:${RESET} ${#PASS_NOTES[@]}"
  echo "  ${YELLOW}warns:${RESET}  $WARNS"
  echo "  ${RED}fails:${RESET}  $FAILS"
  if (( WARNS > 0 )); then
    echo
    echo "  ${YELLOW}Warnings:${RESET}"
    local w
    for w in "${WARN_NOTES[@]}"; do echo "    - $w"; done
  fi
  if (( FAILS > 0 )); then
    echo
    echo "  ${RED}Failures:${RESET}"
    local f
    for f in "${FAIL_NOTES[@]}"; do echo "    - $f"; done
    echo
    echo "${RED}${BOLD}Not ready to submit.${RESET}"
    exit 1
  fi
  echo
  if (( DRY_RUN == 1 )); then
    echo "${GREEN}${BOLD}Host preflight passed.${RESET} Build the image with:"
    echo "  just up"
    echo "  just trial-submission-image"
    echo "  just push-submission-image YOURHUBUSER/cmu-vln-ai:v1"
    return
  fi
  echo "${GREEN}${BOLD}Image is ready for the Hub link + public-fork push.${RESET}"
  echo "  Official eval command:  ros2 launch dummy_vlm dummy_vlm.launch"
  echo "  Live trial:             just trial-submission-image"
  if [[ "$DO_PUSH" != "1" ]]; then
    echo "  Push when ready:        just push-submission-image YOURHUBUSER/cmu-vln-ai:v1"
  fi
}

# -- main --------------------------------------------------------------------

preflight
if (( DRY_RUN == 1 )); then
  banner "Dry run — host checks only"
  echo "  Re-run without --dry-run after just up to check $TAG."
  summary
  exit 0
fi
if (( DO_BUILD == 1 )); then
  build_image
fi
check_image
trial_call
push_image
summary
