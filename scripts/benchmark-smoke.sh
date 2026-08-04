#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
h3_load_env
h3_require_license_acknowledgement

PROFILE="${1:-}"
RUN="${2:-1}"
HEAD_IP="${HEAD_IP:-}"
API_PORT="${H3_API_PORT:-8000}"
PROJECT_DIR="$H3_PROJECT_ROOT"
RESULT_DIR="${H3_RESULT_DIR:-$PROJECT_DIR/results/benchmarks}"
OUT="$RESULT_DIR/${PROFILE}-run${RUN}.mp4"
META="$RESULT_DIR/${PROFILE}-run${RUN}.txt"

[[ "$PROFILE" =~ ^[a-z0-9-]+$ ]] || h3_fail "profile must contain lowercase letters, digits, or hyphens"
[[ "$RUN" =~ ^[0-9]+$ ]] || h3_fail "run must be an integer"
h3_require_ipv4 HEAD_IP "$HEAD_IP"
h3_require_command curl
h3_require_command ffmpeg
h3_require_command ffprobe
h3_require_command sha256sum
mkdir -p "$RESULT_DIR"
if [[ -e "$OUT" || -e "$META" ]]; then
  h3_fail "refusing to overwrite existing benchmark output or metadata"
fi

tmp_out="$(mktemp "${OUT}.partial.XXXXXX")"
trap 'rm -f "$tmp_out"' EXIT
start_ns="$(date +%s%N)"
code="$(curl -sS --max-time 3600 -w '%{http_code}' -o "$tmp_out" -X POST "http://$HEAD_IP:$API_PORT/v1/videos/sync" \
  -F 'prompt=Macro soldering a PCB under warm bench light, soft room tone.' \
  -F 'width=768' \
  -F 'height=448' \
  -F 'fps=24' \
  -F 'num_inference_steps=20' \
  -F 'flow_shift=12' \
  -F 'seed=42' \
  -F 'extra_params={"task":"t2va","duration":2.0,"audio_flow_shift":3.0}')"
end_ns="$(date +%s%N)"
[[ "$code" = 200 ]] || h3_fail "request failed with HTTP $code"

ffmpeg -v error -i "$tmp_out" -f null -
mv "$tmp_out" "$OUT"
trap - EXIT
elapsed_ms="$(( (end_ns - start_ns) / 1000000 ))"
{
  echo "profile=$PROFILE"
  echo "run=$RUN"
  echo "client_elapsed_ms=$elapsed_ms"
  sha256sum "$OUT"
  ffprobe -v error -show_entries stream=index,codec_name,codec_type,width,height,r_frame_rate,sample_rate,channels,nb_frames -show_entries format=duration,size -of json "$OUT"
} | tee "$META"
