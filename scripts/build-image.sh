#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
h3_load_env

IMAGE="${H3_2X_IMAGE:-minimax-h3-2x-dgx-spark:experimental}"
BASE_IMAGE="${H3_BASE_IMAGE:-minimax-h3-dgx-spark:sm121-fp8}"
WORKER_HOST="${WORKER_HOST:-gx10}"
PROJECT_DIR="$H3_PROJECT_ROOT"

h3_require_command docker
h3_require_command ssh
h3_require_safe_value H3_2X_IMAGE "$IMAGE"
h3_require_safe_value H3_BASE_IMAGE "$BASE_IMAGE"
h3_require_safe_value WORKER_HOST "$WORKER_HOST"

docker image inspect "$BASE_IMAGE" >/dev/null
docker build --build-arg "BASE_IMAGE=$BASE_IMAGE" -t "$IMAGE" "$PROJECT_DIR"

if [[ "${SYNC_WORKER:-1}" == 1 ]]; then
  docker save "$IMAGE" | ssh "$WORKER_HOST" docker load
fi

head_id="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
# Values interpolated into this remote command passed h3_require_safe_value.
# shellcheck disable=SC2029
worker_id="$(ssh "$WORKER_HOST" "docker image inspect '$IMAGE' --format '{{.Id}}'")"
test "$head_id" = "$worker_id"
echo "image ready on both Sparks: $IMAGE $head_id"
