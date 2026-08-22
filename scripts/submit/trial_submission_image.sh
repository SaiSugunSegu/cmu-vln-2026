#!/usr/bin/env bash
# One-question smoke against the live sim: arabic_room Q01, TARE explores.
# Official Category 1 sample — "How many sofas are below a window?" (GT 2).
#
# Host-side, same path as just eval-cat1-sim: fetches the Unity mesh from Drive
# only when data/scenes/arabic_room is missing, docker-cps it into
# iros2026_system, starts challenge_simulation.sh, then launches
# smart_vlm.launch with scene_source:=sim.
#
#   just up
#   just trial-submission-image
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AI_NAME="iros2026_odyssey"
SYS_NAME="iros2026_system"

# Hardcoded trial item (benchmark_2 arabic_room category 1).
TRIAL_SCENE="arabic_room"
TRIAL_QUESTION_ID="Q01"
TRIAL_QUESTION="How many sofas are below a window?"
TRIAL_ANSWER=2
TRIAL_REPORT="/data/runs/trial_submission.json"

TRIAL_DISPLAY=""

usage() {
  cat <<'EOF'
Run arabic_room Q01 on the live sim (TARE explores).

Fetches the Unity mesh from Drive only if it is not already under data/scenes/.
Starts Xvfb :99 in iros2026_system (this box has no monitor).

Usage:
  just up
  just trial-submission-image

  scripts/submit/trial_submission_image.sh [--display :1]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) shift 2 ;;  # accepted for compatibility; the running containers are used
    --display) TRIAL_DISPLAY="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for name in "$AI_NAME" "$SYS_NAME"; do
  if ! docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null | grep -qx true; then
    echo "$name is not running — just up first" >&2
    exit 1
  fi
done

echo
echo "Trial: $TRIAL_SCENE $TRIAL_QUESTION_ID  (live sim, TARE explores)"
echo "  $TRIAL_QUESTION"
echo "  GT $TRIAL_ANSWER"
echo "  containers $AI_NAME + $SYS_NAME"
echo "  display ${TRIAL_DISPLAY:-Xvfb :99 (headless)}"
echo

sweep=(
  python3 "$ROOT/scripts/eval/run_sim_sweep.py"
  --category 1
  --scenes "$TRIAL_SCENE"
  --limit 1
  --question-id "$TRIAL_QUESTION_ID"
  --target-source vlm
  --report "$TRIAL_REPORT"
)
if [[ -n "$TRIAL_DISPLAY" ]]; then
  sweep+=(--display "$TRIAL_DISPLAY")
fi
"${sweep[@]}"

python3 - "$ROOT/data/runs/trial_submission.json" "$TRIAL_QUESTION_ID" "$TRIAL_ANSWER" <<'PY'
import json, sys
from pathlib import Path
path, qid, gt = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
if not path.is_file():
    print(f"no report at {path}", file=sys.stderr)
    sys.exit(1)
doc = json.loads(path.read_text())
rows = [r for r in (doc.get("results") or []) if r.get("id") == qid]
if not rows:
    print(f"report has no {qid} row", file=sys.stderr)
    sys.exit(1)
row = rows[-1]
pred = row.get("predicted")
ok = bool(row.get("correct"))
print()
print(f"  predicted={pred}  gt={row.get('gt', gt)}  correct={ok}  "
      f"error={row.get('error')!r}  views={row.get('n_context_views')}")
print(f"  report {path}")
sys.exit(0 if ok else 1)
PY
