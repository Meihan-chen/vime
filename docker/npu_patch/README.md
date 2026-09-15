# NPU patch developer guide

This directory contains the vendor patches needed by Vime's Ascend backend.
Keep shared training, model-loading and rollout logic in Vime's mainline
interfaces; use these patches for changes that belong in the vendor runtime.

## Build and source baselines

[Dockerfile.npu](../Dockerfile.npu) is the executable build recipe.
[series.conf](./series.conf) defines the patch application order and the
source paths used by CI to reconcile a checkout with an existing image.
Update these files together when changing dependencies or patch delivery.

The current recipe uses the base image
`quay.io/atlas-ci/vllm-ascend:v0.28.0-fd81546-a3`.
The serving sources below must match the installed compiled extensions and OPP;
an editable install or a source checkout alone does not rebuild those binaries.

| Component | Source revision |
| --- | --- |
| vLLM | `e6bfe03ad73a3330cb427885aa90d97a12e1c704` |
| vLLM-Ascend | `fd815467c221ee600137f6bdd53fe354d5e7c999` |
| Megatron-LM | `1dcf0dafa884ad52ffb243625717a3471643e087` |
| Megatron-Bridge | `3fd3768045422d0aa5c97e90a4e6c659aea9acb9` |
| MindSpeed | `fc63de5c48426dd019c3b3f39e65f5bdf56e4086` |
| MegatronAdaptor | `15582addff3f3d4680e350826fa70d012b475509` |
| TransformerEngineNPU | `d743c83d060d5edc48867ecb9e93ec80d81860e4` |
| mbridge | `89eb10887887bc74853f89a4de258c0702932a1c` |

The Dockerfile preserves the base image's serving package versions using pip
constraints while installing training dependencies. Keep torch, torch-npu,
Triton, CANN and compiled vendor operators compatible; do not independently
upgrade them to satisfy an unrelated package installation.

Vime uses native HF loading/export for the supported NPU model paths.
Megatron-Bridge and mbridge remain in the image recipe as separate dependencies;
their presence does not mean that model loading uses Bridge. Removing those
dependencies requires checking indirect imports and checkpoint tooling.

To build from the repository root, set `VIME_REVISION` to an immutable commit
available from the repository fetched by the Dockerfile:

```bash
docker build -f docker/Dockerfile.npu \
  --build-arg VIME_COMMIT="$VIME_REVISION" \
  -t vime-npu:dev .
```

The Dockerfile fetches Vime from `vllm-project/vime`; a fork-only commit is not
necessarily available there. The build context supplies the patch files while
`VIME_COMMIT` selects the installed Vime source. Keep those two inputs aligned.
The host driver and exposed NPU devices must also match the container stack.

## Patch inventory and order

| Order | Patch | Target | Purpose |
| --- | --- | --- | --- |
| 1 | [vllm.patch](./vllm.patch) | vLLM | Token-in/token-out serving behavior, weight-reload metadata and the GLM MTP graph-compatible mask. |
| 2 | [vllm-ascend.patch](./vllm-ascend.patch) | vLLM-Ascend | Stateful HCCL and packed IPC transfer, reload lifecycle, main/draft update targets, and KV/control-memory allocation boundaries. |
| 3 | [common megatron.patch](../patch/latest/megatron.patch) | Megatron-LM | Shared Vime Megatron changes; copied into the image as `megatron-common.patch`. |
| 4 | [megatron.patch](./megatron.patch) | Megatron-LM | NPU-specific changes on top of the common patch, including training memory and operator compatibility. |
| 5 | [megatron-bridge.patch](./megatron-bridge.patch) | Megatron-Bridge | Compatibility for the Bridge dependency retained in the image. |
| 6 | [mindspeed.patch](./mindspeed.patch) | MindSpeed | NPU feature, argument, attention and patch-registration compatibility. |

The common Megatron patch must be applied **before** the NPU Megatron patch.
Do not duplicate common hunks in the NPU patch or apply all patches from
`docker/patch/latest` indiscriminately: only the common Megatron patch is
included by this NPU recipe.

On clean source trees at the pinned revisions, check each patch before applying
it. For example, using the directory layout from the Dockerfile:

```bash
VIME_PATCH_DIR="$PWD/docker/npu_patch"

git -C /vllm-workspace/vllm apply --check "$VIME_PATCH_DIR/vllm.patch"
git -C /vllm-workspace/vllm apply "$VIME_PATCH_DIR/vllm.patch"

git -C /vllm-workspace/vllm-ascend apply --check "$VIME_PATCH_DIR/vllm-ascend.patch"
git -C /vllm-workspace/vllm-ascend apply "$VIME_PATCH_DIR/vllm-ascend.patch"

git -C /root/Megatron-LM apply --check "$PWD/docker/patch/latest/megatron.patch"
git -C /root/Megatron-LM apply "$PWD/docker/patch/latest/megatron.patch"
git -C /root/Megatron-LM apply --check "$VIME_PATCH_DIR/megatron.patch"
git -C /root/Megatron-LM apply "$VIME_PATCH_DIR/megatron.patch"

git -C /root/Megatron-Bridge apply --check "$VIME_PATCH_DIR/megatron-bridge.patch"
git -C /root/Megatron-Bridge apply "$VIME_PATCH_DIR/megatron-bridge.patch"

git -C /root/MindSpeed apply --check "$VIME_PATCH_DIR/mindspeed.patch"
git -C /root/MindSpeed apply "$VIME_PATCH_DIR/mindspeed.patch"
```

Do not rerun these commands on already-patched trees or discard unrelated
working-tree changes to make a patch apply. Use clean worktrees for rebasing.
After source changes affecting extensions, rebuild the affected binaries using
the matching vendor build instructions before testing.

## Maintaining a patch

1. Start from the pinned vendor revision in a clean worktree. For the NPU
   Megatron patch, apply the common patch first.
2. Rebase only the necessary vendor changes. Prefer an upstream implementation
   when it already satisfies Vime's contract; avoid copying Vime orchestration
   into the vendor patch.
3. Keep vendor regression tests separate from the production patch payload.
   Validate changed contracts in the vendor repository and the corresponding
   Vime tests.
4. Verify clean-base application and reverse application. For Megatron, check
   the common and NPU patches as an ordered pair.
5. When adding, deleting or renaming a patch, update Dockerfile COPY/apply
   operations and `series.conf` in the same change. Do not introduce
   patch-specific branches into the CI reconciler.
6. When changing source baselines, also update the build recipe, this table,
   dependencies and any affected compiled operators.

CI reconciliation uses the image's saved patch bytes under `/opt/npu_patch`
to reverse the old series, then applies the checkout's series. Reversal is in
reverse order. This mechanism updates patches, not vendor source revisions or
installed packages; a source/binary baseline change requires a matching image.

## Runtime integration and validation

- Non-colocate weight transfer uses `hccl`; colocate uses `npu_ipc`.
  Vime's NPU provider selects backend names and initialization types while
  preserving the shared stateful weight-update sequence.
- Serving recipes explicitly set
  `--vllm-additional-config '{"weight_nz_mode":0}'`.
  Do not rely on the legacy `VLLM_ASCEND_ENABLE_NZ` variable as a substitute.
- Do not pass the removed `--megatron-to-hf-mode` or
  `--vllm-weight-sync-mode` arguments. Qwen3-VL Megatron recipes must select
  `--spec vime_plugins.models.qwen3_vl get_qwen3_vl_model_provider`.
- Ray must advertise custom `NPU` resources, not CUDA `GPU` resources.
  Ensure the requested actor/rollout layout fits the visible devices.
- The Ascend `torch_memory_saver` build is specified in Dockerfile.npu.
  Its Python wheel is built from
  `sgl-kernel-npu/contrib/torch_memory_saver/python`; preserve that NPU build
  rather than installing the CUDA implementation.
- Run syntax/launch-contract checks before hardware tests. Cover non-colocate
  HCCL, colocate IPC/MoE, native VL loading and torch_dist/reference loading
  when changing shared NPU infrastructure. Draft-update changes also require
  explicit MTP coverage.
- Distinguish import checks, CPU contracts, operator tests and full E2E
  validation. A successful patch application alone does not establish runtime
  correctness or model support.

Qwen3.5 NPU requires a compatible training FLA and serving GDN/convolution
operator stack and is not covered by these recipes. Keep model enablement and
its dependency validation explicit rather than inferring support from a
vendor patch or a model configuration alone.
