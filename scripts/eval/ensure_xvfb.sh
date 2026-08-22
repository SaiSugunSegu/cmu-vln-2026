#!/usr/bin/env bash
# Start a dummy X server in iros2026_system. Unity needs a DISPLAY; this box
# has no monitor. VirtualGL (vglrun -d egl) renders on the NVIDIA GPU and
# blits into Xvfb.
#
# Prints the display on stdout (default :99). Logs on stderr.
#
# A live server is a unix socket that accepts a connection. pgrep -f on the
# Xvfb command line is not used: docker exec bash -lc "…Xvfb :99…" matches
# the wrapper itself, so the check would succeed with no X server at all.
# /tmp/.X11-unix is bind-mounted from the host and sticky-bitted, so a dead
# host socket for the same display must be unlinked as root before bind.
set -euo pipefail

NAME="${SYS_CONTAINER:-iros2026_system}"
DISPLAY_N="${1:-:99}"
N="${DISPLAY_N#:}"

if ! docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -qx true; then
  echo "$NAME is not running — just up first" >&2
  exit 1
fi

x_alive() {
  docker exec "$NAME" python3 -c "
import socket, sys
s = socket.socket(socket.AF_UNIX)
s.settimeout(1)
try:
    s.connect('/tmp/.X11-unix/X${N}')
except Exception:
    sys.exit(1)
"
}

if x_alive; then
  echo "$DISPLAY_N"
  exit 0
fi

if ! docker exec "$NAME" bash -lc "command -v Xvfb >/dev/null"; then
  echo "installing xvfb in $NAME" >&2
  docker exec -u root "$NAME" bash -lc \
    "apt-get update -qq && apt-get install -y --no-install-recommends xvfb" >&2
fi

docker exec -u root "$NAME" bash -lc "rm -f /tmp/.X11-unix/X${N}"

docker exec "$NAME" bash -lc \
  "mkdir -p /tmp/runtime-docker && chmod 700 /tmp/runtime-docker; \
   nohup Xvfb ${DISPLAY_N} -screen 0 1920x1080x24 -ac \
     +extension GLX +render -noreset >/tmp/xvfb.log 2>&1 & \
   sleep 0.4; \
   pgrep -x Xvfb >/dev/null"

for _ in 1 2 3 4 5 6 7 8 9 10; do
  if x_alive; then
    echo "Xvfb ${DISPLAY_N} in $NAME" >&2
    echo "$DISPLAY_N"
    exit 0
  fi
  sleep 0.3
done

echo "Xvfb ${DISPLAY_N} failed to come up in $NAME (see /tmp/xvfb.log)" >&2
docker exec "$NAME" bash -lc "tail -20 /tmp/xvfb.log" >&2 || true
exit 1
