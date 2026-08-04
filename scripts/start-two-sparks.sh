#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
h3_load_env

HEAD_HOST="${HEAD_HOST:-spark-head}"
WORKER_HOST="${WORKER_HOST:-spark-peer}"
HEAD_IP="${HEAD_IP:-}"
WORKER_IP="${WORKER_IP:-}"
IMAGE="${H3_2X_IMAGE:-minimax-h3-2x-dgx-spark:experimental}"
MODEL_DIR="${MINIMAX_H3_MODEL_DIR:-}"
HF_CACHE="${HF_CACHE_DIR:-}"
API_PORT="${H3_API_PORT:-8000}"
RAY_PORT="${H3_RAY_PORT:-6379}"
MASTER_PORT="${H3_MASTER_PORT:-29500}"
IFACE="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}"
HCA="${NCCL_IB_HCA:-rocep1s0f1}"
HEAD_GID="${H3_HEAD_GID_INDEX:-${NCCL_IB_GID_INDEX:-3}}"
WORKER_GID="${H3_WORKER_GID_INDEX:-${NCCL_IB_GID_INDEX:-3}}"
PROJECT_DIR="$H3_PROJECT_ROOT"
OUTPUT_DIR="${H3_OUTPUT_DIR:-$PROJECT_DIR/output}"
ATTENTION_BACKEND="${H3_DIFFUSION_ATTENTION_BACKEND:-CUDNN_ATTN}"
EXECUTION_MODE="${H3_EXECUTION_MODE:-compile}"
CACHE_BACKEND="${H3_CACHE_BACKEND:-none}"
CACHE_CONFIG="${H3_CACHE_CONFIG:-}"
ENABLE_CACHE_DIT_SUMMARY="${H3_ENABLE_CACHE_DIT_SUMMARY:-false}"
CACHE_PROFILE="${H3_CACHE_PROFILE_OVERRIDE:-${H3_CACHE_PROFILE:-}}"

if [[ "$CACHE_PROFILE" = balanced ]]; then
  CACHE_BACKEND=cache_dit
  ENABLE_CACHE_DIT_SUMMARY=true
  CACHE_CONFIG='{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"max_cached_steps":-1,"residual_diff_threshold":0.15,"max_continuous_cached_steps":1,"enable_taylorseer":false}'
fi

for pair in \
  "HEAD_HOST:$HEAD_HOST" \
  "WORKER_HOST:$WORKER_HOST" \
  "H3_2X_IMAGE:$IMAGE" \
  "MINIMAX_H3_MODEL_DIR:$MODEL_DIR" \
  "HF_CACHE_DIR:$HF_CACHE" \
  "NCCL_SOCKET_IFNAME:$IFACE" \
  "NCCL_IB_HCA:$HCA" \
  "H3_HEAD_GID_INDEX:$HEAD_GID" \
  "H3_WORKER_GID_INDEX:$WORKER_GID" \
  "H3_OUTPUT_DIR:$OUTPUT_DIR"; do
  h3_require_safe_value "${pair%%:*}" "${pair#*:}"
done
h3_require_nonnegative_integer H3_HEAD_GID_INDEX "$HEAD_GID"
h3_require_nonnegative_integer H3_WORKER_GID_INDEX "$WORKER_GID"
case "$ATTENTION_BACKEND" in
  TORCH_SDPA|CUDNN_ATTN|FLASH_ATTN) ;;
  *) h3_fail "H3_DIFFUSION_ATTENTION_BACKEND must be TORCH_SDPA, CUDNN_ATTN, or FLASH_ATTN" ;;
esac
case "$EXECUTION_MODE" in
  eager|compile) ;;
  *) h3_fail "H3_EXECUTION_MODE must be eager or compile" ;;
esac
case "$CACHE_BACKEND" in
  none|cache_dit) ;;
  *) h3_fail "H3_CACHE_BACKEND must be none or cache_dit" ;;
esac
case "$CACHE_PROFILE" in
  ""|balanced) ;;
  *) h3_fail "H3_CACHE_PROFILE must be empty or balanced" ;;
esac
case "$ENABLE_CACHE_DIT_SUMMARY" in
  true|false) ;;
  *) h3_fail "H3_ENABLE_CACHE_DIT_SUMMARY must be true or false" ;;
esac
if [[ -n "$CACHE_CONFIG" ]]; then
  h3_require_command python3
  python3 -c 'import json,sys; value=json.loads(sys.argv[1]); assert isinstance(value, dict)' "$CACHE_CONFIG" ||
    h3_fail "H3_CACHE_CONFIG must be a JSON object"
fi
if [[ "$CACHE_BACKEND" = none && ( -n "$CACHE_CONFIG" || "$ENABLE_CACHE_DIT_SUMMARY" = true ) ]]; then
  h3_fail "cache configuration requires H3_CACHE_BACKEND=cache_dit"
fi
h3_require_ipv4 HEAD_IP "$HEAD_IP"
h3_require_ipv4 WORKER_IP "$WORKER_IP"
h3_require_port H3_API_PORT "$API_PORT"
h3_require_port H3_RAY_PORT "$RAY_PORT"
h3_require_port H3_MASTER_PORT "$MASTER_PORT"

COMMON_ENV=(
  -e NCCL_NET=IB
  -e NCCL_IB_DISABLE=0
  -e NCCL_IB_HCA="$HCA"
  -e NCCL_IB_ADDR_FAMILY=AF_INET
  -e NCCL_IB_ROCE_VERSION_NUM=2
  -e NCCL_SOCKET_IFNAME="$IFACE"
  -e GLOO_SOCKET_IFNAME="$IFACE"
  -e NCCL_CROSS_NIC=0
  -e NCCL_CUMEM_ENABLE=0
  -e NCCL_NVLS_ENABLE=0
  -e NCCL_IGNORE_CPU_AFFINITY=1
  -e NCCL_DEBUG=INFO
  -e FLASHINFER_DISABLE_VERSION_CHECK=1
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn
  -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  -e RAY_memory_monitor_refresh_ms=0
)

docker_args_text() {
  printf ' %q' "$@"
}

"$(dirname "$0")/preflight.sh"
"$(dirname "$0")/stop-two-sparks.sh"

head_common="$(docker_args_text "${COMMON_ENV[@]}" -e NCCL_IB_GID_INDEX="$HEAD_GID")"
worker_common="$(docker_args_text "${COMMON_ENV[@]}" -e NCCL_IB_GID_INDEX="$WORKER_GID")"
execution_args=()
if [[ "$EXECUTION_MODE" = eager ]]; then
  execution_args+=(--enforce-eager)
fi
execution_args_text=""
if (( ${#execution_args[@]} )); then
  execution_args_text="$(docker_args_text "${execution_args[@]}")"
fi
cache_args=()
if [[ "$CACHE_BACKEND" != none ]]; then
  cache_args+=(--cache-backend "$CACHE_BACKEND")
fi
if [[ -n "$CACHE_CONFIG" ]]; then
  cache_args+=(--cache-config "$CACHE_CONFIG")
fi
if [[ "$ENABLE_CACHE_DIT_SUMMARY" = true ]]; then
  cache_args+=(--enable-cache-dit-summary)
fi
cache_args_text=""
if (( ${#cache_args[@]} )); then
  cache_args_text="$(docker_args_text "${cache_args[@]}")"
fi

# Validated values are intentionally expanded on the client for remote Docker.
# shellcheck disable=SC2029
ssh "$HEAD_HOST" "docker run -d --name minimax-h3-2x-ray-head --network host --ipc host --gpus all --device /dev/infiniband --cap-add IPC_LOCK $head_common -v minimax-h3-2x-ray-tmp:/tmp/ray -v '$MODEL_DIR':'$MODEL_DIR':ro -v '$HF_CACHE':/root/.cache/huggingface --entrypoint ray '$IMAGE' start --head --node-ip-address='$HEAD_IP' --port='$RAY_PORT' --dashboard-host='$HEAD_IP' --dashboard-port=8265 --num-cpus=8 --num-gpus=1 --object-store-memory=2000000000 --disable-usage-stats --block >/dev/null"

for _ in $(seq 1 30); do
  if ssh "$HEAD_HOST" "docker exec minimax-h3-2x-ray-head ray status" >/dev/null 2>&1; then
    ray_head_ready=1
    break
  fi
  sleep 2
done
test "${ray_head_ready:-0}" = 1 || {
  echo "Ray head did not become ready" >&2
  exit 1
}

# shellcheck disable=SC2029
ssh "$WORKER_HOST" "docker run -d --name minimax-h3-2x-ray-worker --network host --ipc host --gpus all --device /dev/infiniband --cap-add IPC_LOCK $worker_common -v minimax-h3-2x-ray-worker-tmp:/tmp/ray -v '$MODEL_DIR':'$MODEL_DIR':ro -v '$HF_CACHE':/root/.cache/huggingface --entrypoint ray '$IMAGE' start --address='$HEAD_IP:$RAY_PORT' --node-ip-address='$WORKER_IP' --num-cpus=8 --num-gpus=1 --object-store-memory=2000000000 --disable-usage-stats --block >/dev/null"

for _ in $(seq 1 60); do
  nodes="$(ssh "$HEAD_HOST" "docker exec minimax-h3-2x-ray-head ray status 2>/dev/null" || true)"
  if grep -q '2 node' <<<"$nodes" || [ "$(grep -c 'node_' <<<"$nodes")" -ge 2 ]; then
    ray_pair_ready=1
    break
  fi
  sleep 2
done
test "${ray_pair_ready:-0}" = 1 || {
  echo "Ray cluster did not reach two active nodes" >&2
  exit 1
}

# shellcheck disable=SC2029
ssh "$HEAD_HOST" "docker run -d --name minimax-h3-2x-api --network host --ipc host --pid=container:minimax-h3-2x-ray-head --gpus all --device /dev/infiniband --cap-add IPC_LOCK $head_common -e H3_HEAD_IP='$HEAD_IP' -e H3_WORKER_IP='$WORKER_IP' -e H3_HEAD_GID_INDEX='$HEAD_GID' -e H3_WORKER_GID_INDEX='$WORKER_GID' -e H3_RAY_ADDRESS='$HEAD_IP:$RAY_PORT' -e H3_MASTER_PORT='$MASTER_PORT' -e H3_WORKER_START_TIMEOUT=2400 -e H3_API_PORT='$API_PORT' -v minimax-h3-2x-ray-tmp:/tmp/ray -v '$MODEL_DIR':'$MODEL_DIR':ro -v '$HF_CACHE':/root/.cache/huggingface -v '$OUTPUT_DIR':/output --entrypoint vllm '$IMAGE' serve '$MODEL_DIR' --omni --trust-remote-code --host '$HEAD_IP' --port '$API_PORT' --num-gpus 2 --usp 2 --ring 1 --vae-patch-parallel-size 2 --vae-parallel-mode tile --vae-use-tiling --num-weight-load-threads 2 $execution_args_text $cache_args_text --diffusion-attention-backend '$ATTENTION_BACKEND' --diffusion-quantization-config '{\"method\":\"fp8\",\"activation_scheme\":\"dynamic\",\"ignored_layers\":[\"video_patch_proj\",\"audio_patch_proj\",\"time_embedder.proj_in\",\"time_embedder.proj_out\",\"final_layer.video_out\",\"final_layer.audio_out\"]}' --force-cutlass-fp8 --distributed-executor-backend ray --stage-init-timeout 1800 --init-timeout 2400 >/dev/null"

echo "two-Spark H3 launch started: attention=$ATTENTION_BACKEND execution=$EXECUTION_MODE cache=$CACHE_BACKEND; API will appear at http://$HEAD_IP:$API_PORT/v1 after both ranks load"
