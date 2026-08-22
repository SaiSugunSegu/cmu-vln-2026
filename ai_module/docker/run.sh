#!/bin/bash
# Shell in the running AI container (image cmu-vln-odyssey:submission).
# Start the stack first with `just up` from the repo root.
set -euo pipefail
if ! docker inspect -f '{{.State.Running}}' iros2026_ai_module 2>/dev/null | grep -qx true; then
  echo "iros2026_ai_module is not running — just up first" >&2
  exit 1
fi
exec docker exec -it iros2026_ai_module bash
