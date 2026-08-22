#!/usr/bin/env bash
# One-question smoke of the running AI container: arabic_room Q01, bag replay.
# Official Category 1 sample — "How many sofas are below a window?" (GT 2).
#
#   just up
#   just trial-submission-image
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAME="iros2026_ai_module"

# Hardcoded trial item (benchmark_2 arabic_room category 1).
TRIAL_SCENE="arabic_room"
TRIAL_QUESTION_ID="Q01"
TRIAL_QUESTION="How many sofas are below a window?"
TRIAL_ANSWER=2
TRIAL_REPORT="/data/runs/trial_submission.json"

usage() {
  cat <<'EOF'
Run arabic_room Q01 on the running AI container.

Usage:
  just trial-submission-image

  scripts/submit/trial_submission_image.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) shift 2 ;;  # accepted for compatibility; the running container is used
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -qx true; then
  echo "$NAME is not running — just up first" >&2
  exit 1
fi

echo
echo "Trial: $TRIAL_SCENE $TRIAL_QUESTION_ID"
echo "  $TRIAL_QUESTION"
echo "  GT $TRIAL_ANSWER"
echo "  container $NAME"
echo

docker exec -it "$NAME" bash -lc \
  "source /home/docker/ai_module/install/setup.bash && \
   ros2 run smart_vlm eval_orchestrator --ros-args \
     -p scene:=${TRIAL_SCENE} \
     -p question_id:=${TRIAL_QUESTION_ID} \
     -p question_limit:=1 \
     -p target_source:=vlm \
     -p report_file:=${TRIAL_REPORT}"

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
      f"error={row.get('error')!r}")
print(f"  report {path}")
sys.exit(0 if ok else 1)
PY
