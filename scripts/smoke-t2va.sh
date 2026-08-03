#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
h3_load_env
h3_require_license_acknowledgement

HEAD_IP="${HEAD_IP:-}"
API_PORT="${H3_API_PORT:-8000}"
API_URL="${H3_API_URL:-http://$HEAD_IP:$API_PORT/v1/videos/sync}"
PROJECT_DIR="$H3_PROJECT_ROOT"
OUT="${OUT:-$PROJECT_DIR/output/smoke-t2va-2x.mp4}"

h3_require_command curl
h3_require_command ffmpeg
h3_require_command ffprobe
h3_require_command sha256sum
h3_require_ipv4 HEAD_IP "$HEAD_IP"
h3_require_safe_value OUT "$OUT"
mkdir -p "$(dirname "$OUT")"

start_ns="$(date +%s%N)"
tmp_out="$(mktemp "${OUT}.partial.XXXXXX")"
trap 'rm -f "$tmp_out"' EXIT

code="$(curl -sS -w '%{http_code}' -o "$tmp_out" -X POST "$API_URL" \
  -F 'prompt=Macro soldering a PCB under warm bench light, soft room tone.' \
  -F 'width=768' \
  -F 'height=448' \
  -F 'fps=24' \
  -F 'num_inference_steps=20' \
  -F 'flow_shift=12' \
  -F 'seed=42' \
  -F 'extra_params={"task":"t2va","duration":2.0,"audio_flow_shift":3.0}')"
end_ns="$(date +%s%N)"

test "$code" = 200 || {
  echo "request failed with HTTP $code" >&2
  file "$tmp_out" >&2 || true
  exit 1
}

elapsed_ms="$(( (end_ns - start_ns) / 1000000 ))"
ffprobe -v error -show_entries stream=index,codec_name,codec_type,width,height,r_frame_rate,sample_rate,channels -show_entries format=duration,size -of json "$tmp_out"
ffmpeg -v error -i "$tmp_out" -f null -
mv "$tmp_out" "$OUT"
trap - EXIT
sha256sum "$OUT"
echo "client_elapsed_ms=$elapsed_ms"
