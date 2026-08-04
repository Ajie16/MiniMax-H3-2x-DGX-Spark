#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# This explicit wrapper override is applied after start-two-sparks.sh loads
# .env, so local defaults cannot accidentally turn this launch into no-cache.
export H3_CACHE_PROFILE_OVERRIDE=balanced

exec "$SCRIPT_DIR/start-two-sparks.sh"
