#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
h3_load_env

IMAGE="${H3_2X_IMAGE:-minimax-h3-2x-dgx-spark:experimental}"
BASE_IMAGE="${H3_BASE_IMAGE:-minimax-h3-dgx-spark:sm121-fp8}"
# The upstream acceptance recorded the author's own build ID, which is not
# reproducible across machines because Docker image IDs embed build metadata.
# This fork re-derives the base image from companion commit 8bd7628 and pins
# the locally measured ID below. The upstream digest, companion commit, layer
# ancestry, and runtime version set are still verified fail-closed.
EXPECTED_BASE_IMAGE_ID="sha256:e498adce4d08a27d88f7c2a23a50563d508815cb78f12442a765326e661e2146"
UPSTREAM_BASE_IMAGE="vllm/vllm-omni:minimax-h3@sha256:e930db8e225162d01e17a49dddc43fd0e844208908d8356a028e5c4e7357696e"
COMPANION_REPO_COMMIT="8bd7628dbdb51a0ea00c301ddcb1a098874870e4"
WORKER_HOST="${WORKER_HOST:-spark-peer}"
SYNC_WORKER="${SYNC_WORKER:-1}"
PROJECT_DIR="$H3_PROJECT_ROOT"

h3_require_command docker
h3_require_safe_value H3_2X_IMAGE "$IMAGE"
h3_require_safe_value H3_BASE_IMAGE "$BASE_IMAGE"
case "$SYNC_WORKER" in
  0) ;;
  1)
    h3_require_command ssh
    h3_require_safe_value WORKER_HOST "$WORKER_HOST"
    ;;
  *) h3_fail "SYNC_WORKER must be 0 or 1" ;;
esac

actual_base_image_id="$(docker image inspect "$BASE_IMAGE" --format '{{.Id}}')"
[[ "$actual_base_image_id" = "$EXPECTED_BASE_IMAGE_ID" ]] ||
  h3_fail "local base image ID is $actual_base_image_id; expected $EXPECTED_BASE_IMAGE_ID from companion commit $COMPANION_REPO_COMMIT"

docker image inspect "$UPSTREAM_BASE_IMAGE" >/dev/null ||
  h3_fail "pinned upstream image is not present locally: $UPSTREAM_BASE_IMAGE"
mapfile -t upstream_layers < <(
  docker image inspect "$UPSTREAM_BASE_IMAGE" --format '{{range .RootFS.Layers}}{{println .}}{{end}}' |
    sed -n '/./p'
)
mapfile -t base_layers < <(
  docker image inspect "$BASE_IMAGE" --format '{{range .RootFS.Layers}}{{println .}}{{end}}' |
    sed -n '/./p'
)
for index in "${!upstream_layers[@]}"; do
  [[ "${base_layers[$index]:-}" = "${upstream_layers[$index]}" ]] ||
    h3_fail "local base image does not descend from the accepted upstream digest"
done

BUILD_EXTRA_ARGS=()
if [[ -n "${H3_BUILD_NETWORK:-}" || -n "${H3_BUILD_PROXY:-}" ]]; then
  # Local deployments behind an HTTP proxy (or with direct egress) need host
  # networking so the build container can reach the host proxy on 127.0.0.1
  # or the host's direct internet path during the pip install step.
  BUILD_EXTRA_ARGS+=(--network host)
fi
if [[ -n "${H3_BUILD_PROXY:-}" ]]; then
  BUILD_EXTRA_ARGS+=(--build-arg "HTTP_PROXY=$H3_BUILD_PROXY")
  BUILD_EXTRA_ARGS+=(--build-arg "HTTPS_PROXY=$H3_BUILD_PROXY")
  BUILD_EXTRA_ARGS+=(--build-arg "NO_PROXY=localhost,127.0.0.1")
fi
if [[ -n "${H3_BUILD_PIP_INDEX_URL:-}" ]]; then
  BUILD_EXTRA_ARGS+=(--build-arg "PIP_INDEX_URL=$H3_BUILD_PIP_INDEX_URL")
fi
if [[ -n "${H3_BUILD_PIP_TRUSTED_HOST:-}" ]]; then
  BUILD_EXTRA_ARGS+=(--build-arg "PIP_TRUSTED_HOST=$H3_BUILD_PIP_TRUSTED_HOST")
fi
docker build "${BUILD_EXTRA_ARGS[@]}" \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "H3_UPSTREAM_BASE_IMAGE=$UPSTREAM_BASE_IMAGE" \
  --build-arg "H3_COMPANION_REPO_COMMIT=$COMPANION_REPO_COMMIT" \
  --build-arg "H3_ACCEPTED_BASE_IMAGE_ID=$EXPECTED_BASE_IMAGE_ID" \
  -t "$IMAGE" "$PROJECT_DIR"

docker run --rm --network none --entrypoint python \
  -v "$PROJECT_DIR/scripts/check-runtime-versions.py:/tmp/check-runtime-versions.py:ro" \
  "$IMAGE" /tmp/check-runtime-versions.py

if [[ "$SYNC_WORKER" == 1 ]]; then
  docker save "$IMAGE" | ssh "$WORKER_HOST" docker load
  head_id="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
  # Values interpolated into this remote command passed h3_require_safe_value.
  # shellcheck disable=SC2029
  worker_id="$(ssh "$WORKER_HOST" "docker image inspect '$IMAGE' --format '{{.Id}}'")"
  test "$head_id" = "$worker_id"
  echo "image ready on both Sparks: $IMAGE $head_id"
else
  head_id="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
  echo "image ready on head only (SYNC_WORKER=0): $IMAGE $head_id"
fi
