# Platform integration: ownership and NPU maintenance

The implementation reuses PR #2 on main `4ad3486c` (M10). Its public workflows
remain main's workflows. One `PlatformIntegration` supplies the environment,
Megatron lifecycle/memory behavior, and vLLM weight transfer implementation.
Hardware acceptance is the next step; CPU checks cannot establish NPU numerical
correctness or communication reliability.

## Start here: what changed

```mermaid
flowchart LR
    A[Existing accelerator selection] --> B[Complete PlatformIntegration value]
    B --> C[Process environment]
    B --> D[Megatron integration]
    B --> E[vLLM transfer]
    C --> F[Shared Ray and training workflow]
    D --> F
    E --> F
```

| Responsibility | Read this file first |
|---|---|
| Choose/describe a device; keep main's CUDA/MUSA/SUPA selection | `vime/utils/accelerator/__init__.py` |
| Bind a platform factory, validate and publish it once | `vime/utils/platform_integration.py` |
| Construct the complete Ascend implementation | `vime_plugins/ascend/__init__.py` |
| Prepare child process environment | `vime/utils/device_env.py`, `vime_plugins/ascend/device_env.py` |
| Fixed training stages and offload | `vime/backends/megatron_integration.py`, `vime_plugins/ascend/training.py` |
| Choose transfer backend, trainer info and IPC identity | `vime/backends/vllm_weight_transfer.py`, `vime_plugins/ascend/weight_transfer.py` |
| Own Ray reuse/start, callback and job submission | `vime/utils/external_utils/command_utils.py` |

The separate env/offload/hook registries have been removed. Device selection
keeps its existing registry. `PlatformIntegration` has three explicit components; adding a
platform does not add another public registry or launch algorithm. There is one
default factory binding in `platform_integration.py`; custom platforms can use
`VIME_PLUGINS=module`. Factories must construct lightweight objects without
installing runtime patches. They may explicitly choose existing defaults.
Missing components or stale method signatures fail before publication.

The integration cannot switch after assembly. The resolved device and factory module
are propagated to child processes; conflicting overrides fail. Successful
Megatron bootstrap runs once. A partially failed vendor patch installation
requires a new process because arbitrary third-party mutations cannot safely
be rolled back.

## Keep the boundary small

`PlatformIntegration` is a platform composition layer. It does not replace
`Accelerator`, select a second device, or provide another device API. The name
replaces the draft's ambiguous `Support`; factories now expose
`build_integration()`, and training uses `MegatronIntegration` with an
`AscendMegatronIntegration` implementation. Out-of-tree factories using the
previous draft API must update these names.

Keep three decisions separate:

| Decision | Owner |
|---|---|
| Which accelerator runs this job? | Existing `VIME_ACCELERATOR` selection. `export VIME_ACCELERATOR=npu` is sufficient to request NPU. |
| Does this process need a local device now? | The caller: `initialize_accelerator(require_device=False)` prepares imports; actual device operations require a usable device. This rule also applies to newly registered backends. |
| Does this job need HF-to-`torch_dist` conversion? | The job script. Calling `convert_checkpoint()` requests conversion on any platform; HF-only jobs omit that call. |

There are no `imports_without_local_device` or `torch_dist_checkpoint_convert`
hardware capabilities. CUDA, MUSA and SUPA device implementation files retain
their M10 contents. Import preparation still runs the existing MUSA/SUPA runtime
bootstrap. Deferring the local-device check does not suppress errors raised by
a broken runtime or remove Megatron's vendor-library requirements.

The integration factory resolves the three components once and binds the child
environment to that selection. Consumers share the bound environment; they do
not repeatedly construct wrappers or re-query a separate factory-module API.
Ray workers may be remote or created by an already running raylet, so the
resolved selection still travels through the existing runtime-environment
mechanism. A shell export alone does not establish that propagation contract.

For another platform, register its device implementation through the existing
accelerator mechanism and provide the three integration components, explicitly
reusing defaults where appropriate. `VIME_PLUGINS` chooses the integration
module; it is not a replacement for registering a device backend. Individual
vendor quirks belong inside these implementations, not in new global registries
or hardware capability flags for job choices. The contract tests exercise an
independent backend with custom Ray resources and no extra launch environment.

## s8 to s10: what was carried forward

The comparison uses four fixed trees, not the whole N8-to-N10 diff:

| Tree | Commit |
|---|---|
| M8 | `791f101e20db377b28e15f12a00863baaea2605f` |
| N8 | `cb9437514b3ba9260f4329acdc2cb4c5aa90d788` |
| M10, this branch's base | `4ad3486cb740926fa064df1d3414dccbea3c1939` |
| N10, incremental reference | `3ab462a29d1f3c39e6d59eb1585ba330ed2bf6d5` |
| PR #2, reusable implementation | `f280827666f431e6a94c94ff8cb16dd4df559128` |

Compare `N8 - M8` with `N10 - M10`, then inspect changes in main's callers.
This avoids counting synchronized main code as new NPU work.

| Increment | Result in this branch |
|---|---|
| Trainer factory moved out of `update_weight/common.py` | The real M10 `connect_rollout_engines_from_distributed` calls the transfer component. No patch targets the deleted factory. |
| HCCL needs remote cleanup in mixed IPC/distributed mode | Both NCCL and HCCL use paired shutdown. Reconnection cleans up the engines attached to the old trainer before replacing the binding. |
| Retained serving and unified training entrypoint | Keep M10 `train.py`, Ray liveness/reuse, callback, submission switch and ROCm path. NPU launch code supplies inputs only. Shell examples stop killing sessions automatically. |
| New serving source pins and APIs | Carry N10's reviewed standalone vLLM and vLLM-Ascend patches: HCCL close, disk delta, sampling metadata, GDN and serving lifecycle changes. |
| Pinned NPU Megatron differs from main's vendor pin | Carry the standalone Megatron patch; leave GPU patches untouched. Capability-gate main's automatic per-buffer backup suppression. NPU retains full training-region backup; explicitly requesting unsupported suppression is rejected. |
| MindSpeed / MegatronAdaptor patch bytes | Reuse unchanged N8/N10 patches. Runtime installation now has explicit stages and failures. |
| Image and large-model test changes | Carry N10 constraints, Rust/router build, temporary checkpoint paths and test changes; replace stale `VIME_PLATFORM` uses. Require a fetched source SHA and verify patch/base identity. |

PR model/reward/example additions are retained as separate functionality. They
are not necessary costs of the support assembly mechanism. Their tests and
hardware needs remain separate from its contract tests.

## What to check on the next synchronization

Follow the three component boundaries. For each moved caller, verify the new
function still calls the component at the right stage. Run the tests through
that real function. Do not rely only on a conflict-free merge or source grep.

For a new vendor base, update `docker/npu_patch/versions.json`, Docker source
pins, and the corresponding patch together. Verify clean apply and exact
reverse against pristine source. The NPU CI update script refuses a different base before
reverting its old patches. The source identity checks cover the framework and
vendor source pins, not byte-for-byte reproducibility of all downloaded tools.

`tools/check_npu_stack.py` is build/update tooling, not a prerequisite for NPU
execution and not an integration interface. It only compares Git HEADs with the
recorded source commits; it does not verify installed packages, patch contents
or hardware compatibility. The NPU Megatron pin remains `1dcf0dafa884ad52ffb243625717a3471643e087`
in N8, N10 and this implementation. The M10 GPU image uses `4b4acac9a1d28ea6829c8d4f566d75698a21249d`;
the NPU integration does not upgrade to that GPU pin.

Current GPU defaults include CUDA TMS library search order and all three TMS
environment variables. MUSA/SUPA use their pre-existing main defaults explicitly;
this change does not claim new hardware validation for them.

## Validation order

CPU checks cover complete assembly and failures, fresh-process imports,
resource/environment combinations, the common launcher, actual weight-update
callers, allocator error propagation, checkpoint validation and attention
train/eval arguments. Existing main CPU tests cover model/recovery/serving
behavior. The synthetic third-platform case checks an independent custom
resource with an empty environment; it is not a hardware compatibility claim.

First GPU/NPU acceptance should proceed in this order:

| Case | Minimum observation required |
|---|---|
| One NPU actor + serving child | Same selected backend/factory, correct Ray resource and local device ID; final attention/TE classes match the pin. |
| Tiny dense model, offload disabled | Two forward/backward/optimizer steps; finite loss/gradients and expected parameter changes. |
| Nonzero attention dropout | Train/eval behavior and packed/unpacked numerical comparison, including packed sequence boundaries. |
| IPC, HCCL, then mixed transfer | Two updates each, changed topology, reconnect, and remote cleanup; inference observes the new weight version. |
| Offload repeated three times | Model/optimizer restoration, memory release, and errors surfaced rather than swallowed. |
| Checkpoint round trip | Validation on; model and optimizer match, next step matches; repeat with async/precision-aware options. |
| Trainer restart with retained serving | Same live serving session survives and resumes from the intended weight version. |
| GPU regression | Existing CUDA launch/offload/update/checkpoint case; ROCm smoke where available. |

Only expand to large MoE, multimodal, async, QAT or draft-model cases after these
pass. Alternative MindSpeed feature combinations may change the actual TE
classes; check them separately. The s10 sample validates one known evolution;
it does not guarantee every future upstream change will be compatible.
