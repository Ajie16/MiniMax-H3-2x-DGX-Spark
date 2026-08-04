#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
h3_load_env
h3_require_license_acknowledgement

PROFILE="${PROFILE:-hq}"
HEAD_IP="${HEAD_IP:-127.0.0.1}"
API_PORT="${H3_API_PORT:-8000}"
API_URL="${H3_API_URL:-http://$HEAD_IP:$API_PORT/v1/videos/sync}"
OUT="${OUT:-$H3_PROJECT_ROOT/results/benchmarks/${PROFILE}-beach-seed314159.mp4}"
LOG="${LOG:-${OUT%.mp4}.log}"
PART="${OUT}.part"

PROMPT='Photorealistic live-action documentary footage, one continuous eye-level wide tracking shot of exactly five adult friends walking together along a remote tropical beach at golden hour. The group has varied skin tones, gender presentation, ages roughly 25 to 50, and body types, all wearing ordinary nonsexual casual beach clothing. Keep all five adults distinct and fully visible with stable faces, hands, limbs, clothing, and consistent identities. Physically coherent surf rolls in and recedes, footprints persist in wet sand, reflections follow each person, and wind moves hair, loose fabric, and palm leaves in the same direction. Natural unposed conversation gestures, no posing, no children, no extra people, no cuts, no text, no logos. Realistic ambient ocean and wind audio, no music.'

[[ "$PROFILE" =~ ^[a-z0-9-]+$ ]] || h3_fail "PROFILE must contain lowercase letters, digits, or hyphens"
h3_require_command curl
h3_require_command ffmpeg
h3_require_command ffprobe
h3_require_command sha256sum
mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"
if [[ -e "$OUT" || -e "$PART" ]]; then
  echo "Refusing to overwrite an existing benchmark artifact: $OUT or $PART" >&2
  exit 1
fi

start_iso="$(date --iso-8601=seconds)"
start_ns="$(date +%s%N)"
printf 'started=%s\noutput=%s\n' "$start_iso" "$OUT" | tee "$LOG"

http_code="$(curl --max-time 7200 --connect-timeout 10 -sS \
  -w '%{http_code}' \
  -o "$PART" \
  -X POST "$API_URL" \
  -F "prompt=$PROMPT" \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=314159' \
  -F 'extra_params={"task":"t2va","duration":4.0,"audio_flow_shift":3.0}')"

end_ns="$(date +%s%N)"
elapsed_ms="$(( (end_ns - start_ns) / 1000000 ))"
printf 'http_code=%s\nelapsed_ms=%s\nelapsed_seconds=%s.%03d\n' \
  "$http_code" "$elapsed_ms" "$((elapsed_ms / 1000))" "$((elapsed_ms % 1000))" | tee -a "$LOG"

if [[ "$http_code" != "200" ]]; then
  error_out="${OUT%.mp4}.error"
  mv "$PART" "$error_out"
  echo "Generation failed; response saved to $error_out" | tee -a "$LOG" >&2
  exit 1
fi

ffprobe -v error -show_entries \
  stream=index,codec_name,codec_type,width,height,r_frame_rate,nb_frames,sample_rate,channels:format=duration,size \
  -of json "$PART" | tee -a "$LOG"
ffmpeg -v error -i "$PART" -f null -
mv "$PART" "$OUT"
sha256sum "$OUT" | tee -a "$LOG"
printf 'completed=%s\n' "$(date --iso-8601=seconds)" | tee -a "$LOG"
