#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
h3_load_env

HEAD_HOST="${HEAD_HOST:-spark-head}"
WORKER_HOST="${WORKER_HOST:-spark-peer}"

h3_require_command ssh
h3_require_safe_value HEAD_HOST "$HEAD_HOST"
h3_require_safe_value WORKER_HOST "$WORKER_HOST"

ssh "$HEAD_HOST" 'docker stop minimax-h3-2x-api >/dev/null 2>&1 || true; docker rm minimax-h3-2x-api >/dev/null 2>&1 || true'
ssh "$WORKER_HOST" 'docker stop minimax-h3-2x-ray-worker >/dev/null 2>&1 || true; docker rm minimax-h3-2x-ray-worker >/dev/null 2>&1 || true'
ssh "$HEAD_HOST" 'docker stop minimax-h3-2x-ray-head >/dev/null 2>&1 || true; docker rm minimax-h3-2x-ray-head >/dev/null 2>&1 || true'

echo "stopped MiniMax H3 two-Spark experiment containers"
