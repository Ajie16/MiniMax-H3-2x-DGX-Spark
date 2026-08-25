#!/usr/bin/env bash

H3_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

h3_load_env() {
  local env_file="$H3_PROJECT_ROOT/.env"
  if [[ -f "$env_file" ]]; then
    # Preserve any caller-provided overrides so command-line exports win over .env.
    local -a preserved=(HEAD_HOST WORKER_HOST HEAD_IP WORKER_IP H3_QUANTIZATION MINIMAX_H3_MODEL_DIR H3_EXECUTION_MODE H3_INT8_W8A8 H3_INT8_DEBUG H3_INT8_EAGER H3_VAE_DECODER_TILE_SIZE H3_VAE_DECODER_TILE_OVERLAP H3_VAE_STACK_TILING H3_TORCH_PROFILER_DIR)
    local var saved
    declare -A h3_env_overrides
    for var in "${preserved[@]}"; do
      if [[ -n "${!var:-}" ]]; then
        h3_env_overrides[$var]="${!var}"
      fi
    done
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
    for var in "${preserved[@]}"; do
      if [[ -n "${h3_env_overrides[$var]:-}" ]]; then
        printf -v "$var" '%s' "${h3_env_overrides[$var]}"
        export "$var"
      fi
    done
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

h3_effective_cache_backend() {
  local profile="${H3_CACHE_PROFILE_OVERRIDE:-${H3_CACHE_PROFILE:-}}"
  if [[ "$profile" == balanced ]]; then
    printf '%s\n' cache_dit
  else
    printf '%s\n' "${H3_CACHE_BACKEND:-none}"
  fi
}

h3_lora_python() {
  local pythonpath="$H3_PROJECT_ROOT"
  if [[ -n "${PYTHONPATH:-}" ]]; then
    pythonpath="$H3_PROJECT_ROOT:$PYTHONPATH"
  fi
  PYTHONPATH="$pythonpath" python3 -m h3_multinode.lora_catalog "$@"
}

h3_validate_lora_profile() {
  local mode="${H3_LORA_MODE:-off}"
  local allow="${H3_LORA_ALLOW_TURBO:-false}"
  local max_cpu="${H3_MAX_CPU_LORAS:-1}"
  local dir="${H3_LORA_DIR:-}"
  local catalog="${H3_LORA_CATALOG:-}"

  case "$mode" in
    off|static|request) ;;
    *) h3_fail "H3_LORA_MODE must be off, static, or request" ;;
  esac
  case "$allow" in
    true|false) ;;
    *) h3_fail "H3_LORA_ALLOW_TURBO must be true or false" ;;
  esac
  if [[ "$mode" == off ]]; then
    return 0
  fi
  if [[ "$mode" == request && "${H3_EXECUTION_MODE:-compile}" != eager ]]; then
    h3_fail "request mode requires H3_EXECUTION_MODE=eager until compile+switch is measured"
  fi
  if [[ "$(h3_effective_cache_backend)" != none ]]; then
    h3_fail "LoRA is mutually exclusive with Cache-DiT"
  fi
  h3_require_safe_value H3_LORA_DIR "$dir"
  [[ "$dir" == /* ]] || h3_fail "H3_LORA_DIR must be an absolute path"
  if [[ -n "$catalog" ]]; then
    h3_require_safe_value H3_LORA_CATALOG "$catalog"
    [[ "$catalog" == /* ]] || h3_fail "H3_LORA_CATALOG must be an absolute path"
  fi
  h3_require_command python3
  if [[ -v H3_LORA_SCALE ]]; then
    python3 -c 'import os, sys
value = os.environ["H3_LORA_SCALE"]
try:
    scale = float(value)
except ValueError:
    sys.exit("H3_LORA_SCALE must be a float in (0, 8]")
if not 0.0 < scale <= 8.0:
    sys.exit("H3_LORA_SCALE must be a float in (0, 8]")
' || h3_fail "H3_LORA_SCALE must be a float in (0, 8]"
  fi
  [[ "$max_cpu" =~ ^[1-9][0-9]*$ ]] || h3_fail "H3_MAX_CPU_LORAS must be an integer >= 1"
  h3_lora_python validate-env || h3_fail "LoRA catalog/profile validation failed"
}

h3_preflight_lora_artifacts() {
  local mode="${H3_LORA_MODE:-off}"
  local head="${HEAD_HOST:-}"
  local worker="${WORKER_HOST:-}"
  local name resolved manifest host head_hash worker_hash

  [[ "$mode" == off ]] && return 0
  [[ -n "$head" && -n "$worker" ]] || h3_fail "HEAD_HOST and WORKER_HOST are required for LoRA preflight"
  h3_require_command python3
  h3_require_command ssh

  while IFS=$'\t' read -r name resolved manifest; do
    [[ -n "$name" && -n "$resolved" ]] || continue
    for host in "$head" "$worker"; do
      ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" \
        "test -d $(printf '%q' "$resolved") && \
         test -f $(printf '%q' "$resolved/adapter_config.json") && \
         { test -f $(printf '%q' "$resolved/adapter_model.safetensors") || \
           test -f $(printf '%q' "$resolved/adapter_model.bin"); }" ||
        h3_fail "LoRA adapter $name is missing PEFT files on $host"
      if [[ -n "$manifest" ]]; then
        ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" \
          "test -f $(printf '%q' "$H3_LORA_DIR/$manifest") && \
           (cd $(printf '%q' "$H3_LORA_DIR") && sha256sum -c $(printf '%q' "$manifest") --status)" ||
          h3_fail "LoRA adapter $name sha256 manifest failed on $host"
      fi
    done
    head_hash="$(ssh -o BatchMode=yes "$head" \
      "cd $(printf '%q' "$resolved") && find . -type f -print0 | sort -z | xargs -0 -r sha256sum")"
    worker_hash="$(ssh -o BatchMode=yes "$worker" \
      "cd $(printf '%q' "$resolved") && find . -type f -print0 | sort -z | xargs -0 -r sha256sum")"
    [[ "$head_hash" == "$worker_hash" ]] ||
      h3_fail "LoRA adapter $name SHA-256 tree differs between $head and $worker"
  done < <(h3_lora_python preflight-list)
}
