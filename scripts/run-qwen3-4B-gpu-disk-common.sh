#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"

DATA_ROOT="${DATA_ROOT:-/root}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${DATA_ROOT}/Qwen3-4B}"
MEGATRON_CHECKPOINT="${MEGATRON_CHECKPOINT:-${DATA_ROOT}/Qwen3-4B_vime}"
REF_CHECKPOINT="${REF_CHECKPOINT:-${DATA_ROOT}/Qwen3-4B_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_ROOT}/dapo-math-17k/dapo-math-17k.jsonl}"
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-${DATA_ROOT}/Megatron-LM}"

ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-2}"

if [[ "${UPDATE_WEIGHT_MODE}" == delta ]]; then
   python3 - <<'PY'
from vllm.v1.worker.gpu_worker import Worker

if not hasattr(Worker, "pull_weights"):
    raise RuntimeError(
        "disk + delta requires vllm.v1.worker.gpu_worker.Worker.pull_weights; "
        "apply the GPU worker disk checkpoint patch to the installed vLLM source"
    )
PY
fi

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_CHECKPOINT}"
   --load "${MEGATRON_CHECKPOINT}"
   --save "${MEGATRON_CHECKPOINT}"
   --save-interval "${SAVE_INTERVAL:-20}"
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type "${RM_TYPE:-deepscaler}"
   --num-rollout "${NUM_ROLLOUT:-3000}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-32}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE:-1}"
   --global-batch-size "${GLOBAL_BATCH_SIZE:-256}"
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TRAIN_TP_SIZE:-2}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-9216}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-6}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

VLLM_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}"
   --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.7}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

SYNC_ARGS=(
   --update-weight-mode "${UPDATE_WEIGHT_MODE}"
   --update-weight-transport disk
   --update-weight-disk-dir "${UPDATE_WEIGHT_DISK_DIR}"
)
if [[ "${UPDATE_WEIGHT_MODE}" == delta ]]; then
   SYNC_ARGS+=(
      --update-weight-local-checkpoint-dir "${UPDATE_WEIGHT_LOCAL_CHECKPOINT_DIR}"
      --update-weight-delta-encoding "${UPDATE_WEIGHT_DELTA_ENCODING:-xor}"
   )
fi

cd "${SCRIPT_DIR}/.."
ray start --head \
   --port="${RAY_GCS_PORT}" \
   --temp-dir="${RAY_TEMP_DIR}" \
   --node-ip-address="${MASTER_ADDR:-127.0.0.1}" \
   --num-gpus="$((ACTOR_GPUS + ROLLOUT_GPUS))" \
   --disable-usage-stats \
   --dashboard-host=0.0.0.0 \
   --dashboard-port="${RAY_DASHBOARD_PORT}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}:${PYTHONPATH:-}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_GPUS}" \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${VLLM_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${SYNC_ARGS[@]}" \
   "$@"
