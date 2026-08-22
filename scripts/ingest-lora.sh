#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

SRC=""
NAME=""
LAYOUT=""
DEST=""
ALPHA=""

usage() {
  cat <<'EOF'
usage: ingest-lora.sh --src FILE --name NAME --qkv-layout grouped|qkv|identity --dest DIR [--lora-alpha N]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src)
      SRC="$2"
      shift 2
      ;;
    --name)
      NAME="$2"
      shift 2
      ;;
    --qkv-layout)
      LAYOUT="$2"
      shift 2
      ;;
    --dest)
      DEST="$2"
      shift 2
      ;;
    --lora-alpha)
      ALPHA="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      h3_fail "unknown argument: $1"
      ;;
  esac
done

[[ -n "$SRC" && -n "$NAME" && -n "$LAYOUT" && -n "$DEST" ]] || {
  usage >&2
  h3_fail "--src, --name, --qkv-layout, and --dest are required"
}
case "$LAYOUT" in
  grouped|qkv|identity) ;;
  *) h3_fail "--qkv-layout must be grouped, qkv, or identity" ;;
esac
[[ -f "$SRC" ]] || h3_fail "source file is missing: $SRC"
mkdir -p "$DEST"
h3_require_safe_value name "$NAME"
IMAGE="${H3_2X_IMAGE:-minimax-h3-2x-dgx-spark:experimental}"

run_python() {
  python3 "$SCRIPT_DIR/ingest_lora.py" "$@"
}

args=(--src "$SRC" --name "$NAME" --qkv-layout "$LAYOUT" --dest "$DEST")
if [[ -n "$ALPHA" ]]; then
  args+=(--lora-alpha "$ALPHA")
fi

if python3 -c 'import torch, safetensors' >/dev/null 2>&1; then
  run_python "${args[@]}"
else
  h3_require_command docker
  docker image inspect "$IMAGE" >/dev/null
  docker run --rm --network none --entrypoint python \
    -u "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$SRC:$SRC:ro" \
    -v "$DEST:$DEST" \
    -v "$H3_PROJECT_ROOT:/workspace:ro" \
    -w /workspace \
    "$IMAGE" \
    scripts/ingest_lora.py "${args[@]}"
fi
