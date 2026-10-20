# NPU integration: entry points and maintenance

Device selection stays in main's `accelerator` mechanism. Main owns Ray/job
startup, training, weight updates and checkpoints. Ascend supplies the process
environment and the training/transfer behavior those workflows actually need.
The code still needs GPU/NPU hardware acceptance.

## Read the code by workflow

| Workflow | Core entry point | Ascend implementation |
|---|---|---|
| Device API and Ray resource metadata | `vime/utils/accelerator/` | `accelerator/npu.py` |
| Child process environment | `vime/utils/device_env.py` | `vime_plugins/ascend/device_env.py` |
| Megatron patches and training allocation | `vime/backends/megatron_integration.py` | `vime_plugins/ascend/training.py`, `patches.py` |
| vLLM engine/trainer protocol and IPC keys | `vime/backends/vllm_weight_transfer.py` | `vime_plugins/ascend/weight_transfer.py` |
| Ray reuse/start, callback and submission | `vime/utils/external_utils/command_utils.py` | Environment/resource inputs only |
| Checkpoint save/load | Main's `megatron_utils/checkpoint.py` | No platform hook or validation bypass |

`platform_integration.component()` imports the selected module and reads the
requested component. There is no `PlatformIntegration` aggregate, factory,
full-stack signature check, environment proxy or second cache to reset.
For example, launching a job only requires the module's `env`; it does not
require its Megatron or weight-transfer contracts to pass validation.
Ascend's three exported objects are lightweight. Vendor libraries load only
when the owning backend invokes them, not when selecting an environment.

`for_child()` passes explicit selection through Ray runtime environments and
serving subprocesses, rejecting conflicting selectors. Implicit GPU environments
receive no new selection variables. Startup inputs are read before Ray starts;
submission defaults and caller/config overrides are read after the callback,
matching main. CUDA's TMS library search and environment contract remain intact.

## Adding another backend

1. Register the device through the existing accelerator API during application
   startup. Set its Ray resource metadata when it uses a custom resource.
2. Set `VIME_PLUGINS=your.module`. Export the components that your workflows
   use: `env`, `megatron`, and/or `weight_transfer`; reuse the existing default
   classes where their behavior applies. A missing requested component fails
   at that workflow's entry point. No new registry or launcher is needed.
3. Keep module imports lightweight and put vendor initialization in the owning
   backend method. Set device/module selection before startup and keep it fixed
   for the process. Partially installed third-party patches require a restart.
4. Test through the real callers, including one workflow that does not use the
   other components. A synthetic third-device test covers this loading contract;
   it does not prove third-device numerical correctness.

CUDA/MUSA/SUPA retain main defaults when no module is requested. Other devices
must supply a module; an empty override does not silently select CUDA behavior.
The earlier draft `build_integration()` API has been removed. Custom modules
using that draft must export their component objects instead.

## Dependency compatibility belongs to its backend

The generic argument validator no longer imports platform or Megatron policy.
Megatron argument validation applies main's automatic offload buffer defaults
when `_ParamAndGradBuffer._allocate_buffer` exists, because that is the entry
point wrapped by main's `memory.py`. This check describes the supported dependency
implementations, not a universal hardware capability. On the pinned older NPU
stack, keep full training-region backup; explicit unsupported buffer suppression
still fails before argument patch mutation. Role-specific overrides are applied
after the automatic defaults, as on main.

The latest s10 cleanup removes FC1 partition-axis overrides and stale TMS/NZ
environment settings. This branch follows it and leaves the shared weight
loading/gathering code identical to main. It does not add an unproven MindSpeed
patch. Check actual dense/expert parameter metadata during hardware acceptance,
including TP/expert-TP > 1. Keep main's checkpoint validation; there is no
`VIME_NPU_SKIP_CHECKPOINT_VALIDATION` bypass.

Device availability remains a caller decision: imports may use
`initialize_accelerator(require_device=False)`, while execution requires an
available device. Checkpoint conversion remains a job decision: calling
`convert_checkpoint()` requests conversion; HF-only jobs omit it. No hardware
capability or extra conversion-tool `after_args` stage controls that decision.

## s8 to s10: what was carried forward

The comparison uses four fixed trees, not the whole N8-to-N10 diff:

| Tree | Commit |
|---|---|
| M8 | `791f101e20db377b28e15f12a00863baaea2605f` |
| N8 | `cb9437514b3ba9260f4329acdc2cb4c5aa90d788` |
| M10, this branch's base | `4ad3486cb740926fa064df1d3414dccbea3c1939` |
| N10, incremental reference | `78207dd2cfd14b96eaa893e78c77c2799c68743b` |
| PR #2, reusable implementation | `f280827666f431e6a94c94ff8cb16dd4df559128` |

Compare `N8 - M8` with `N10 - M10`, then inspect changes in main's callers.
This avoids counting synchronized main code as new NPU work.

| Increment | Result in this branch |
|---|---|
| Trainer factory moved out of `update_weight/common.py` | The real M10 `connect_rollout_engines_from_distributed` calls the transfer component. No patch targets the deleted factory. |
| HCCL needs remote cleanup in mixed IPC/distributed mode | Both NCCL and HCCL use paired shutdown. Reconnection cleans up the engines attached to the old trainer before replacing the binding. |
| Retained serving and unified training entrypoint | Keep M10 `train.py`, Ray liveness/reuse, callback, submission switch and ROCm path. NPU launch code supplies inputs only. Shell examples stop killing sessions automatically. |
| New serving source pins and APIs | Carry N10's reviewed standalone vLLM and vLLM-Ascend patches: HCCL close, disk delta, sampling metadata, GDN and serving lifecycle changes. |
| Pinned NPU Megatron differs from main's vendor pin | Carry the standalone Megatron patch; leave GPU patches untouched. Resolve automatic per-buffer defaults inside Megatron argument validation using the dependency allocation entry point. NPU retains full training-region backup; explicitly requesting unsupported suppression is rejected. |
| MindSpeed / MegatronAdaptor patch bytes | Reuse unchanged N8/N10 patches. Runtime installation now has explicit stages and failures. |
| Image and large-model test changes | Carry N10 constraints, Rust/router build, temporary checkpoint paths and test changes; replace stale `VIME_PLATFORM` uses. Require a fetched source SHA and verify patch/base identity. |

PR model/reward/example additions are retained as separate functionality. They
are separate from backend component loading. Their tests and
hardware needs remain separate from its contract tests.

## What to check on the next synchronization

Follow the three component boundaries. For each moved caller, verify the new
function still calls the component at the right stage. Run the tests through
that real function. Do not rely only on a conflict-free merge or source grep.

For a new vendor base, update the Docker base image or source pins and the
corresponding patches together, then rebuild the image. Verify clean apply and
exact reverse against pristine source. The NPU CI update script reconciles
patches; it does not upgrade or validate the installed vendor versions.

The NPU Megatron pin remains `1dcf0dafa884ad52ffb243625717a3471643e087`
in N8, N10 and this implementation. The M10 GPU image uses `4b4acac9a1d28ea6829c8d4f566d75698a21249d`;
the NPU integration does not upgrade to that GPU pin.

Current GPU defaults include CUDA TMS library search order and all three TMS
environment variables. MUSA/SUPA use their pre-existing main defaults explicitly;
this change does not claim new hardware validation for them.

## Validation order

CPU checks cover component isolation and failures, fresh-process imports,
resource/environment combinations, the common launcher, actual weight-update
callers, allocator error propagation, checkpoint staging and attention
train/eval arguments. Existing main CPU tests cover model/recovery/serving
behavior. The synthetic third-platform case checks an independent custom
resource with an empty environment; it is not a hardware compatibility claim.

First GPU/NPU acceptance should proceed in this order:

| Case | Minimum observation required |
|---|---|
| One NPU actor + serving child | Same selected backend/module, correct Ray resource and local device ID; final attention/TE classes match the pin. |
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
