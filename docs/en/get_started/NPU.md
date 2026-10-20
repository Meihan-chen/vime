# Ascend NPU: build and validate this branch

This branch integrates Ascend support with the s10 main baseline. **It has not
passed GPU/NPU hardware acceptance.** Start with the small validation cases in
[the implementation guide](../development/npu-support.md), before large models.
The current target is Megatron training and vLLM Ascend serving on the pinned A3
stack. A2, FSDP on NPU, and alternate MindSpeed feature combinations need separate
validation.

## 1. Build a matching image

Build from a committed checkout that is fetchable from `VIME_REPOSITORY`:

```bash
VIME_SHA=$(git rev-parse HEAD)
docker build -f docker/Dockerfile.npu \
  --build-arg VIME_COMMIT="$VIME_SHA" \
  --build-arg VIME_REPOSITORY=https://github.com/YOUR_ORG/vime.git \
  -t "vime-npu:$VIME_SHA" .
```

The build requires a full commit SHA. It compares the fetched code's NPU patches
with the build context. Vendor versions are selected by the base image and
source pins in [Dockerfile.npu](../../../docker/Dockerfile.npu). Rebuild the image
when these versions change; replacing patch files does not upgrade dependencies.
The build itself still needs validation on the image builder.

Use the known working driver/device mounts for your Ascend host. The container
must expose actual compute devices (`/dev/davinci0`, etc.), in addition to the
manager/driver devices and libraries. A manager node alone is not a usable NPU.
Initialize the installed CANN/ATB environment inside the container:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export VIME_ACCELERATOR=npu
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

Use `VIME_ACCELERATOR`, including in conversion processes. The old NPU branch's
`VIME_PLATFORM` setting is no longer used. On mixed hosts, select the intended
backend explicitly; automatic selection preserves main's existing priorities.

## 2. Prepare a small model case

```bash
export MODEL_ROOT=/root
hf download Qwen/Qwen3-4B --local-dir "$MODEL_ROOT/models/Qwen3-4B"
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir "$MODEL_ROOT/datasets/dapo-math-17k"
```

For an 8-NPU example (4 training, 4 serving):

```bash
DATA_ROOT="$MODEL_ROOT" bash scripts/run-qwen3-4B-npu.sh
```

For initial acceptance, shorten the example to one rollout and a small batch;
then test two weight updates and a trainer restart. The [4-NPU fully async
example](../../../examples/fully_async/run-qwen3-4B-fully_async-npu.sh) uses the
same `train.py` entrypoint. Run async after synchronous transfer works.

Scripts preserve an existing Ray session. Use `VIME_SCRIPT_EXTERNAL_RAY=1` for a
head you manage separately. Shut down a session explicitly when you intend to
start a fresh one; restarting a trainer must preserve its retained serving
processes.

## 3. Configuration boundaries

| Setting | Behavior |
|---|---|
| `VIME_ACCELERATOR=npu` | Selects NPU. Metadata queries work without a local device; execution requires a visible NPU and runtime. |
| `VIME_PLUGINS` unset | Loads Ascend components for NPU; other built-in devices retain main defaults. |
| `VIME_PLUGINS=your.module` | Reads the component used by the current workflow (`env`, `megatron`, or `weight_transfer`). |
| `VIME_PLUGINS=` | Allows standalone device APIs. NPU workflows requiring backend components fail explicitly. |
| `VIME_NPU_PYTHONPATH`, `VIME_NPU_LD_LIBRARY_PATH` | Override the image-layout defaults for launch paths. |
| Buffer CPU backup | NPU offload keeps the full training region backed up. Main's automatic per-buffer backup suppression is enabled only where supported. Explicit `disable_param_buffers_cpu_backup=True` or `disable_grad_buffers_cpu_backup=True` configuration is rejected on this NPU pin. |

A CPU driver can inspect device metadata, construct Ray resource/environment inputs and
load backend definitions. Loading Megatron backend definitions still requires
the matching vendor libraries, even when the driver itself has no NPU.

For checkpoint correctness, keep validation enabled and verify a save/load
round trip, optimizer state, and the next training step. Choose HF loading or
HF-to-distributed conversion in the job script. `convert_checkpoint()` performs
the requested conversion on NPU as on other platforms; an HF-only job should
not call it. The large-model test scripts exercise both loading paths.
