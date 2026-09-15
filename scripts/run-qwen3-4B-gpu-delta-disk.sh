#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export UPDATE_WEIGHT_MODE=delta
# Shared trainer/rollout path containing the published delta stream.
export UPDATE_WEIGHT_DISK_DIR="${UPDATE_WEIGHT_DISK_DIR:-/tmp/vime-gpu-delta-weights}"
# Rollout-host-local NVMe path containing the materialized checkpoint.
export UPDATE_WEIGHT_LOCAL_CHECKPOINT_DIR="${UPDATE_WEIGHT_LOCAL_CHECKPOINT_DIR:-/tmp/vime-gpu-rollout-checkpoint}"
export RAY_GCS_PORT="${RAY_GCS_PORT:-6399}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8267}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray-vime-gpu-delta-disk}"
exec bash "${SCRIPT_DIR}/run-qwen3-4B-gpu-disk-common.sh" "$@"
