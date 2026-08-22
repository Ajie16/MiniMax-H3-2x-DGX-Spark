#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=../scripts/common.sh
source "$ROOT/scripts/common.sh"

expect_fail() {
  local needle="$1"
  local output=""
  local rc=0
  shift
  output="$("$@" 2>&1)" || rc=$?
  if [[ "$rc" -eq 0 ]]; then
    printf 'expected validation to refuse: %s\n%s\n' "$needle" "$output" >&2
    exit 1
  fi
  grep -Fq "$needle" <<<"$output" || {
    printf 'missing expected validation message: %s\n%s\n' "$needle" "$output" >&2
    exit 1
  }
}

reset_lora_env() {
  unset H3_LORA_MODE H3_LORA_DIR H3_LORA_CATALOG H3_LORA_NAME H3_LORA_SCALE
  unset H3_MAX_CPU_LORAS H3_LORA_ALLOW_TURBO
  unset H3_EXECUTION_MODE H3_CACHE_BACKEND H3_CACHE_PROFILE H3_CACHE_PROFILE_OVERRIDE
}

reset_lora_env
h3_validate_lora_profile

reset_lora_env
export H3_LORA_MODE=bogus
expect_fail "H3_LORA_MODE must be off, static, or request" h3_validate_lora_profile

reset_lora_env
export H3_LORA_MODE=request
export H3_EXECUTION_MODE=compile
export H3_LORA_DIR=/tmp/h3-lora-profile-test
expect_fail "request mode requires H3_EXECUTION_MODE=eager" h3_validate_lora_profile

reset_lora_env
export H3_LORA_MODE=static
export H3_LORA_DIR=/tmp/h3-lora-profile-test
export H3_LORA_NAME=turbo4
export H3_CACHE_PROFILE_OVERRIDE=balanced
expect_fail "LoRA is mutually exclusive with Cache-DiT" h3_validate_lora_profile

reset_lora_env
export H3_LORA_MODE=static
export H3_LORA_DIR=/tmp/h3-lora-profile-test
export H3_LORA_NAME=turbo4
export H3_LORA_SCALE=not-a-number
expect_fail "H3_LORA_SCALE must be a float in (0, 8]" h3_validate_lora_profile

reset_lora_env
export H3_LORA_MODE=static
export H3_LORA_DIR=/tmp/h3-lora-profile-test
export H3_LORA_NAME=turbo4
export H3_LORA_SCALE=9
expect_fail "H3_LORA_SCALE must be a float in (0, 8]" h3_validate_lora_profile

reset_lora_env
WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/h3-lora-profile.XXXXXX")"
trap 'rm -rf "$WORKDIR"' EXIT
mkdir -p "$WORKDIR"
export H3_LORA_MODE=request
export H3_EXECUTION_MODE=eager
export H3_LORA_DIR="$WORKDIR"
export H3_CACHE_BACKEND=none
expect_fail "LoRA catalog is missing" h3_validate_lora_profile

cat >"$WORKDIR/catalog.json" <<'JSON'
{
  "version": 1,
  "adapters": {
    "turbo4": {
      "path": "turbo4/",
      "format": "peft",
      "profile": "turbo",
      "default_scale": 1.0,
      "recommended_steps": 4,
      "sha256_manifest": "turbo4.sha256"
    }
  }
}
JSON

reset_lora_env
export H3_LORA_MODE=static
export H3_LORA_DIR="$WORKDIR"
export H3_LORA_NAME=turbo4
export H3_LORA_ALLOW_TURBO=false
export H3_CACHE_BACKEND=none
expect_fail "turbo adapters require H3_LORA_ALLOW_TURBO=true" h3_validate_lora_profile

reset_lora_env
export H3_LORA_MODE=static
export H3_LORA_DIR="$WORKDIR"
export H3_LORA_NAME=turbo4
export H3_LORA_ALLOW_TURBO=true
export H3_CACHE_BACKEND=none
h3_validate_lora_profile

reset_lora_env
export H3_LORA_MODE=off
export H3_CACHE_PROFILE_OVERRIDE=balanced
h3_validate_lora_profile

echo "LoRA profile validation tests passed"
