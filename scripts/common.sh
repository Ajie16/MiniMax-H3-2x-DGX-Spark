#!/usr/bin/env bash

H3_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

h3_load_env() {
  local env_file="$H3_PROJECT_ROOT/.env"
  if [[ -f "$env_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi
}

h3_fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

h3_require_command() {
  command -v "$1" >/dev/null || h3_fail "$1 is required"
}

h3_require_license_acknowledgement() {
  [[ "${MINIMAX_H3_LICENSE_ACKNOWLEDGED:-false}" == "true" ]] ||
    h3_fail "read MODEL-LICENSE.md and set MINIMAX_H3_LICENSE_ACKNOWLEDGED=true only if authorized"
}

h3_require_safe_value() {
  local name="$1" value="$2"
  [[ -n "$value" && "$value" =~ ^[A-Za-z0-9._@:/+-]+$ ]] ||
    h3_fail "$name contains unsupported characters"
}

h3_require_ipv4() {
  local name="$1" value="$2"
  [[ "$value" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] ||
    h3_fail "$name must be an IPv4 address"
}

h3_require_port() {
  local name="$1" value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value < 1 || value > 65535 )); then
    h3_fail "$name must be an integer from 1 to 65535"
  fi
}

h3_require_nonnegative_integer() {
  local name="$1" value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || h3_fail "$name must be a non-negative integer"
}
