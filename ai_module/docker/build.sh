#!/bin/bash
# Bake cmu-vln-odyssey:submission (weights + keys) and start the stack.
# Same entry as `just up` from the repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec just up
