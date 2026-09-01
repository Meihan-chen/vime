"""Unit tests for colocated vLLM IPC weight sync."""

from __future__ import annotations

import gc
import importlib
import sys
import types
import weakref
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

MODULE_PATH = "vime.backends.megatron_utils.update_weight.update_weight_from_tensor"

_PURGE_PREFIXES = ("megatron", "mindspeed", "vime.backends.megatron_utils")


def _collect_subtree(prefix: str) -> list[str]:
    """Collect all modules in sys.modules that start with the given prefix."""
    return [k for k in sys.modules.keys() if k == prefix or k.startswith(prefix + ".")]


def _install_stubs():
    repo_root = Path(__file__).resolve().parents[5]
    megatron_utils_pkg = types.ModuleType("vime.backends.megatron_utils")
    megatron_utils_pkg.__path__ = [str(repo_root / "vime/backends/megatron_utils")]
    update_weight_pkg = types.ModuleType("vime.backends.megatron_utils.update_weight")
    update_weight_pkg.__path__ = [str(repo_root / "vime/backends/megatron_utils/update_weight")]
    sys.modules["vime.backends.megatron_utils"] = megatron_utils_pkg
    sys.modules["vime.backends.megatron_utils.update_weight"] = update_weight_pkg

    mpu_stub = MagicMock()
    mpu_stub.get_data_parallel_rank.return_value = 0
    mpu_stub.get_tensor_model_parallel_rank.return_value = 0
    mpu_stub.get_tensor_model_parallel_world_size.return_value = 2
    mpu_stub.get_tensor_model_parallel_group.return_value = "tp_group"
    mpu_stub.get_pipeline_model_parallel_rank.return_value = 0

    megatron_core = types.ModuleType("megatron.core")
    megatron_core.__path__ = []
    megatron_core.mpu = mpu_stub
    megatron_mod = types.ModuleType("megatron")
    megatron_mod.__path__ = []
    megatron_mod.core = megatron_core

    sys.modules.setdefault("megatron", megatron_mod)
    sys.modules.setdefault("megatron.core", megatron_core)

    ray_mod = types.ModuleType("ray")
    ray_mod.get = lambda refs: refs
    ray_mod.ObjectRef = object
    ray_mod.actor = types.ModuleType("ray.actor")
    ray_mod.actor.ActorHandle = object
    sys.modules.setdefault("ray", ray_mod)
    sys.modules.setdefault("ray.actor", ray_mod.actor)

    import torch.distributed as _dist

    dist_stub = MagicMock()
    dist_stub.get_rank.return_value = 0
    dist_stub.get_world_size.return_value = 1
    dist_stub.get_process_group_ranks.return_value = [0, 1]
    dist_stub.new_group.side_effect = lambda ranks, backend: (tuple(ranks), backend)
    dist_stub.barrier = MagicMock()
    dist_stub.all_gather_object = MagicMock()
    _dist.get_rank = dist_stub.get_rank
    _dist.get_world_size = dist_stub.get_world_size
    _dist.get_process_group_ranks = dist_stub.get_process_group_ranks
    _dist.new_group = dist_stub.new_group
    _dist.barrier = dist_stub.barrier
    _dist.all_gather_object = dist_stub.all_gather_object

    vime_utils = types.ModuleType("vime.utils.distributed_utils")
    vime_utils.get_gloo_group = MagicMock(return_value="gloo")
    sys.modules.setdefault("vime.utils.distributed_utils", vime_utils)

    hf_iter_stub = MagicMock()
    hf_iter_stub.get_hf_weight_chunks.return_value = iter([])

    hf_base_mod = types.ModuleType("vime.backends.megatron_utils.update_weight.hf_weight_iterator_base")
    hf_base_mod.HfWeightIteratorBase = MagicMock()
    hf_base_mod.HfWeightIteratorBase.create.return_value = hf_iter_stub

    upw_dist_mod = types.ModuleType("vime.backends.megatron_utils.update_weight.update_weight_from_distributed")
    upw_dist_mod.connect_rollout_engines_from_distributed = MagicMock(return_value="groups")
    upw_dist_mod.disconnect_rollout_engines_from_distributed = MagicMock()
    upw_dist_mod.post_process_weights = MagicMock()
    upw_dist_mod.update_weights_from_distributed = MagicMock(return_value=[])

    for key, mod in [
        ("vime.backends.megatron_utils.update_weight.hf_weight_iterator_base", hf_base_mod),
        ("vime.backends.megatron_utils.update_weight.update_weight_from_distributed", upw_dist_mod),
    ]:
        sys.modules.setdefault(key, mod)

    return hf_iter_stub, upw_dist_mod


_HF_ITER_STUB = MagicMock()
_HF_ITER_STUB.get_hf_weight_chunks.return_value = iter([])

_STUBBED_MODULES = (
    "megatron",
    "megatron.core",
    "ray",
    "ray.actor",
    "vime.backends.megatron_utils",
    "vime.backends.megatron_utils.update_weight",
    "vime.utils.distributed_utils",
    "vime.backends.megatron_utils.update_weight.hf_weight_iterator_base",
    "vime.backends.megatron_utils.update_weight.update_weight_from_distributed",
)
_DIST_ATTRS = (
    "get_rank",
    "get_world_size",
    "get_process_group_ranks",
    "new_group",
    "barrier",
    "all_gather_object",
)


@pytest.fixture(scope="module")
def upw_vllm():
    import torch.distributed as _dist

    purge_keys = set()
    for prefix in _PURGE_PREFIXES:
        purge_keys.update(_collect_subtree(prefix))
    for k in _STUBBED_MODULES:
        purge_keys.add(k)
    purge_keys.add(MODULE_PATH)

    saved_mods = {k: sys.modules.get(k) for k in purge_keys}
    saved_dist = {a: getattr(_dist, a, None) for a in _DIST_ATTRS}
    for k in purge_keys:
        sys.modules.pop(k, None)
    _install_stubs()
    sys.modules.pop(MODULE_PATH, None)

    try:
        yield importlib.import_module(MODULE_PATH)
    finally:
        for k, original in saved_mods.items():
            if original is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = original
        for a, original in saved_dist.items():
            if original is not None:
                setattr(_dist, a, original)


@dataclass
class _RemoteCall:
    args: tuple
    kwargs: dict


class RecordingRemoteMethod:
    def __init__(self):
        self.calls: list[_RemoteCall] = []

    def remote(self, *args, **kwargs):
        self.calls.append(_RemoteCall(args=args, kwargs=kwargs))
        return "ref"


@dataclass
class RecordingVLLMEngine:
    release_memory_occupation: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    resume_memory_occupation: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    init_weight_transfer_engine: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    start_weight_update: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    finish_weight_update: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    update_weights_from_tensor: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    update_weights: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    pause_generation: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    flush_cache: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    continue_generation: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)


def _default_args(**kwargs) -> Namespace:
    base = dict(
        actor_num_nodes=1,
        actor_num_gpus_per_node=4,
        rollout_num_gpus_per_engine=2,
        megatron_to_hf_mode="raw",
        update_weight_buffer_size=1 << 30,
    )
    base.update(kwargs)
    return Namespace(**base)


def _make_instance(upw_vllm, args=None):
    obj = object.__new__(upw_vllm.UpdateWeightFromTensor)
    obj.args = args or _default_args()
    obj.model = []
    obj.weights_getter = lambda: {}
    obj.model_name = "test"
    obj.quantization_config = None
    obj.weight_version = 0
    obj._hf_weight_iterator = _HF_ITER_STUB
    obj.rollout_engines = []
    obj.distributed_rollout_engines = []
    obj.use_distribute = False
    obj._model_update_groups = None
    obj._ipc_gather_group = None
    obj._ipc_gather_src = None
    obj._ipc_engine = None
    obj._is_distributed_src_rank = False
    obj._group_name = "vime"
    obj._ipc_initialized = False
    return obj


def _chunks(n=1):
    return [[(f"p.{i}", torch.zeros(2, 2)) for i in range(2)] for _ in range(n)]


def _run_update(obj, *, chunks=None, rank=0) -> dict[str, int]:
    chunks = chunks or _chunks(1)
    obj._hf_weight_iterator = MagicMock()
    obj._hf_weight_iterator.get_hf_weight_chunks.return_value = iter(chunks)

    counters = {"barrier": 0, "ipc_collect": 0}

    def counting_barrier(*args, **kwargs):
        counters["barrier"] += 1

    def counting_ipc_collect(*args, **kwargs):
        counters["ipc_collect"] += 1

    with patch("torch.distributed.get_rank", return_value=rank), patch(
        "torch.distributed.barrier", side_effect=counting_barrier
    ), patch("torch.cuda.ipc_collect", side_effect=counting_ipc_collect):
        obj.update_weights()
    return counters


@pytest.mark.unit
def test_colocated_lifecycle_uses_native_weight_transfer_session(upw_vllm):
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    obj.rollout_engines = [engine]
    obj._ipc_engine = engine
    obj._ipc_gather_src = 0
    obj._ipc_gather_group = "slot-0"

    with patch(f"{MODULE_PATH}._send_to_colocated_engine", return_value=([], [])) as send_to_colocated:
        counters = _run_update(obj, chunks=_chunks(2))

    assert len(engine.pause_generation.calls) == 1
    assert len(engine.flush_cache.calls) == 1
    assert len(engine.release_memory_occupation.calls) == 0
    assert len(engine.resume_memory_occupation.calls) == 0
    assert len(engine.start_weight_update.calls) == 1
    assert engine.start_weight_update.calls[0].kwargs == {"is_checkpoint_format": True}
    assert len(engine.finish_weight_update.calls) == 1
    assert engine.finish_weight_update.calls[0].kwargs == {}
    assert len(engine.continue_generation.calls) == 1

    assert send_to_colocated.call_count == 2
    assert counters["ipc_collect"] == 3
    assert counters["barrier"] >= 4


@pytest.mark.unit
def test_every_slot_leader_starts_and_finishes_its_engine_once(upw_vllm):
    engines = [RecordingVLLMEngine(), RecordingVLLMEngine()]

    for rank, slot_group, engine in ((0, "slot-0", engines[0]), (4, "slot-1", engines[1])):
        obj = _make_instance(upw_vllm)
        obj.rollout_engines = engines
        obj._ipc_engine = engine
        obj._ipc_gather_src = rank
        obj._ipc_gather_group = slot_group
        with patch(f"{MODULE_PATH}._send_to_colocated_engine", return_value=([], [])):
            _run_update(obj, chunks=_chunks(1), rank=rank)

    for engine in engines:
        assert len(engine.start_weight_update.calls) == 1
        assert engine.start_weight_update.calls[0].kwargs == {"is_checkpoint_format": True}
        assert len(engine.finish_weight_update.calls) == 1


@pytest.mark.unit
def test_producer_refs_live_through_ray_get_then_release(upw_vllm):
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    obj.rollout_engines = [engine]
    obj._ipc_engine = engine
    obj._ipc_gather_src = 0
    obj._ipc_gather_group = "slot-0"
    obj._hf_weight_iterator = MagicMock()
    obj._hf_weight_iterator.get_hf_weight_chunks.return_value = iter(_chunks(1))

    events: list[str] = []
    producer_refs: list[weakref.ReferenceType] = []

    class ProducerStorage:
        pass

    def fake_send(*args, **kwargs):
        producer = ProducerStorage()
        producer_refs.append(weakref.ref(producer))
        return ["chunk-update-ref"], [producer]

    def fake_ray_get(value):
        if value == ["chunk-update-ref"]:
            assert producer_refs[0]() is not None
            events.append("ray.get")
        return value

    def ipc_collect():
        gc.collect()
        assert producer_refs[0]() is None
        events.append("release")

    with patch(f"{MODULE_PATH}._send_to_colocated_engine", side_effect=fake_send), patch.object(
        upw_vllm.ray, "get", side_effect=fake_ray_get
    ), patch("torch.distributed.get_rank", return_value=0), patch("torch.distributed.barrier"), patch(
        "torch.cuda.ipc_collect", side_effect=ipc_collect
    ):
        obj.update_weights()

    assert events == ["ray.get", "release", "release"]


@pytest.mark.unit
def test_build_ipc_info_uses_npu_device_uuid_provider(upw_vllm):
    weight_transfer = MagicMock()
    weight_transfer.current_device_uuid.return_value = "device-uuid"
    platform = MagicMock(is_npu=True, weight_transfer=weight_transfer)
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3).T

    with patch(f"{MODULE_PATH}.current_platform", return_value=platform), patch(
        "torch.multiprocessing.reductions.reduce_tensor",
        return_value=(object(), ("native-ipc-args",)),
    ) as reduce_tensor:
        info, refs = upw_vllm._build_ipc_update_info_from_named_tensors([("layer.weight", source)])

    assert info == {
        "names": ["layer.weight"],
        "dtype_names": ["float32"],
        "shapes": [[3, 2]],
        "ipc_handles": [{"device-uuid": ("native-ipc-args",)}],
    }
    assert len(refs) == 1
    assert refs[0].is_contiguous()
    reduce_tensor.assert_called_once_with(refs[0])


@pytest.mark.unit
def test_current_gpu_uuid_keeps_main_cuda_path(upw_vllm):
    platform = MagicMock(is_npu=False)
    properties = MagicMock(uuid="cuda-device-uuid")

    with patch(f"{MODULE_PATH}.current_platform", return_value=platform), patch(
        "torch.cuda.current_device", return_value=3
    ) as current_device, patch("torch.cuda.get_device_properties", return_value=properties) as get_properties:
        assert upw_vllm._current_gpu_uuid() == "cuda-device-uuid"

    current_device.assert_called_once_with()
    get_properties.assert_called_once_with(3)
    platform.weight_transfer.current_device_uuid.assert_not_called()


@pytest.mark.unit
def test_send_to_single_rank_slot_uses_native_update_endpoint(upw_vllm):
    engine = RecordingVLLMEngine()
    tensors = [("layer.weight", torch.zeros(2, 2))]
    local_info = {
        "names": ["layer.weight"],
        "dtype_names": ["float32"],
        "shapes": [[2, 2]],
        "ipc_handles": [{"device-uuid": ("native-ipc-args",)}],
    }
    refs = [tensors[0][1]]

    with patch("torch.distributed.get_world_size", return_value=1), patch(
        f"{MODULE_PATH}._build_ipc_update_info_from_named_tensors", return_value=(local_info, refs)
    ):
        remote_refs, long_lived = upw_vllm._send_to_colocated_engine(
            tensors,
            ipc_engine=engine,
            ipc_gather_src=0,
            ipc_gather_group="slot-0",
            weight_version=42,
        )

    assert remote_refs == ["ref"]
    assert long_lived is refs
    assert len(engine.update_weights_from_tensor.calls) == 1
    call = engine.update_weights_from_tensor.calls[0]
    assert call.args == ()
    assert call.kwargs == {**local_info, "weight_version": "42"}
    assert len(engine.update_weights.calls) == 0


@pytest.mark.unit
def test_npu_worker_patch_skips_moe_transpose_during_wake_up(upw_vllm):
    hooks = importlib.import_module("vime.backends.megatron_utils.update_weight.npu_worker_extension")
    wake_quant_configs = []

    class FakeWorker:
        def __init__(self):
            self.vllm_config = types.SimpleNamespace(quant_config=None)
            self.moe_transposed = False

        def load_model(self):
            pass

        def start_weight_update(self, is_checkpoint_format=True):
            pass

        def update_weights(self, update_info):
            pass

        def finish_weight_update(self):
            pass

        def wake_up(self, tags=None):
            wake_quant_configs.append(self.vllm_config.quant_config)
            if self.vllm_config.quant_config is None and (tags is None or "weights" in tags):
                self.moe_transposed = True

    native_update_weights = FakeWorker.update_weights
    native_finish_weight_update = FakeWorker.finish_weight_update
    native_wake_up = FakeWorker.wake_up
    hooks._NPUVLLMHijack.patch_one_worker(FakeWorker)

    assert FakeWorker.update_weights is native_update_weights
    assert FakeWorker.finish_weight_update is native_finish_weight_update
    assert FakeWorker.wake_up is not native_wake_up

    worker = FakeWorker()
    worker.wake_up(tags=["weights"])

    assert wake_quant_configs[0] is not None
    assert not worker.moe_transposed
    assert worker.vllm_config.quant_config is None
    assert upw_vllm.vLLMColocateWorkerExtension is hooks.vLLMColocateWorkerExtension
    assert upw_vllm.vLLMWorkerExtension is hooks.vLLMWorkerExtension


@pytest.mark.unit
def test_npu_worker_extension_entries_install_only_their_hooks(upw_vllm):
    hooks = importlib.import_module("vime.backends.megatron_utils.update_weight.npu_worker_extension")

    with patch.object(hooks._NPUVLLMHijack, "patch_a3_moe_alltoall_expert_ids") as patch_expert_ids, patch.object(
        hooks._NPUVLLMHijack, "patch_npu_worker"
    ) as patch_worker, patch.object(hooks._NPUVLLMHijack, "patch_npu_rotary_emb") as patch_rotary:
        hooks.vLLMColocateWorkerExtension()

    patch_expert_ids.assert_called_once_with()
    patch_worker.assert_called_once_with()
    patch_rotary.assert_called_once_with()

    with patch.object(hooks._NPUVLLMHijack, "patch_a3_moe_alltoall_expert_ids") as patch_expert_ids, patch.object(
        hooks._NPUVLLMHijack, "patch_npu_worker"
    ) as patch_worker, patch.object(hooks._NPUVLLMHijack, "patch_npu_rotary_emb") as patch_rotary:
        hooks.vLLMWorkerExtension()

    patch_expert_ids.assert_not_called()
    patch_worker.assert_called_once_with()
    patch_rotary.assert_called_once_with()


@pytest.mark.unit
def test_npu_moe_weight_loader_hook_restores_missing_parameter_loader(upw_vllm):
    hooks = importlib.import_module("vime.backends.megatron_utils.update_weight.npu_worker_extension")
    loader = object()
    w13 = types.SimpleNamespace()
    w2 = types.SimpleNamespace()
    unrelated = types.SimpleNamespace()
    experts = types.SimpleNamespace(weight_loader=loader)
    mlp = types.SimpleNamespace(
        experts=experts,
        named_parameters=lambda: [
            ("experts.w13_weight", w13),
            ("experts.w2_weight", w2),
            ("shared.weight", unrelated),
        ],
    )
    model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[types.SimpleNamespace(mlp=mlp)]))

    hooks._NPUVLLMHijack.patch_moe_weight_loader(model)

    assert w13.weight_loader is loader
    assert w2.weight_loader is loader
    assert not hasattr(unrelated, "weight_loader")


@pytest.mark.unit
def test_send_hf_params_combines_colocated_and_distributed_refs(upw_vllm):
    obj = _make_instance(upw_vllm)
    obj.rollout_engines = [RecordingVLLMEngine()]
    obj.distributed_rollout_engines = [RecordingVLLMEngine()]
    obj.use_distribute = True
    obj._is_distributed_src_rank = True
    obj._model_update_groups = "groups"
    tensors = _chunks(1)[0]

    long_lived = [torch.zeros(1)]
    with patch(
        f"{MODULE_PATH}._send_to_colocated_engine", return_value=(["colocated-ref"], long_lived)
    ) as send_to_colocated, patch(
        f"{MODULE_PATH}.update_weights_from_distributed", return_value=["distributed-ref"]
    ) as send_distributed:
        refs, returned_long_lived = obj._send_hf_params(tensors)

    send_to_colocated.assert_called_once_with(
        tensors,
        ipc_engine=obj._ipc_engine,
        ipc_gather_src=obj._ipc_gather_src,
        ipc_gather_group=obj._ipc_gather_group,
        weight_version=obj.weight_version,
    )
    send_distributed.assert_called_once()
    assert refs == ["colocated-ref", "distributed-ref"]
    assert returned_long_lived is long_lived


@pytest.mark.unit
def test_send_to_colocated_engine_all_gathers_per_slot_and_leader_sends(upw_vllm):
    engine = RecordingVLLMEngine()
    tensors = [("layer.weight", torch.zeros(2, 2))]
    local_info = {
        "names": ["layer.weight"],
        "dtype_names": ["float32"],
        "shapes": [[2, 2]],
        "ipc_handles": [{"device-0": (1, 2, 3)}],
    }
    peer_info = {
        "names": ["layer.weight"],
        "dtype_names": ["float32"],
        "shapes": [[2, 2]],
        "ipc_handles": [{"device-1": (4, 5, 6)}],
    }

    def gather_into_slot(output, payload, *, group):
        assert group == "slot-0"
        output[:] = [payload, upw_vllm._serialize_ipc_update_info(peer_info)]

    with patch("torch.distributed.get_world_size", return_value=2), patch(
        "torch.distributed.get_rank", return_value=0
    ), patch("torch.distributed.all_gather_object", side_effect=gather_into_slot), patch(
        f"{MODULE_PATH}._build_ipc_update_info_from_named_tensors",
        return_value=(local_info, [tensors[0][1]]),
    ):
        remote_refs, long_lived = upw_vllm._send_to_colocated_engine(
            tensors,
            ipc_engine=engine,
            ipc_gather_src=0,
            ipc_gather_group="slot-0",
            weight_version=7,
        )

    assert remote_refs == ["ref"]
    assert long_lived == [tensors[0][1]]
    sent = engine.update_weights_from_tensor.calls[0]
    assert sent.args == ()
    assert sent.kwargs["weight_version"] == "7"
    assert sent.kwargs["ipc_handles"] == [{"device-0": (1, 2, 3), "device-1": (4, 5, 6)}]
    assert len(engine.update_weights.calls) == 0


@pytest.mark.unit
def test_non_leader_gathers_but_does_not_send_rpc(upw_vllm):
    engine = RecordingVLLMEngine()
    tensors = [("layer.weight", torch.zeros(2, 2))]
    local_info = {
        "names": ["layer.weight"],
        "dtype_names": ["float32"],
        "shapes": [[2, 2]],
        "ipc_handles": [{"device-1": (4, 5, 6)}],
    }
    leader_info = {
        "names": ["layer.weight"],
        "dtype_names": ["float32"],
        "shapes": [[2, 2]],
        "ipc_handles": [{"device-0": (1, 2, 3)}],
    }

    def gather_into_slot(output, payload, *, group):
        assert group == "slot-0"
        output[:] = [upw_vllm._serialize_ipc_update_info(leader_info), payload]

    with patch("torch.distributed.get_world_size", return_value=2), patch(
        "torch.distributed.get_rank", return_value=1
    ), patch("torch.distributed.all_gather_object", side_effect=gather_into_slot) as gather, patch(
        f"{MODULE_PATH}._build_ipc_update_info_from_named_tensors",
        return_value=(local_info, [tensors[0][1]]),
    ):
        remote_refs, long_lived = upw_vllm._send_to_colocated_engine(
            tensors,
            ipc_engine=engine,
            ipc_gather_src=0,
            ipc_gather_group="slot-0",
            weight_version=7,
        )

    assert remote_refs == []
    assert long_lived == [tensors[0][1]]
    gather.assert_called_once()
    assert len(engine.update_weights_from_tensor.calls) == 0
    assert len(engine.update_weights.calls) == 0


@pytest.mark.unit
def test_placeholder_rank_skips_ipc_export_and_collective(upw_vllm):
    with patch(f"{MODULE_PATH}._build_ipc_update_info_from_named_tensors") as build, patch(
        "torch.distributed.all_gather_object"
    ) as gather:
        refs, long_lived = upw_vllm._send_to_colocated_engine(
            _chunks(1)[0],
            ipc_engine=None,
            ipc_gather_src=None,
            ipc_gather_group=None,
            weight_version=1,
        )

    assert refs == []
    assert long_lived is None
    build.assert_not_called()
    gather.assert_not_called()


@pytest.mark.unit
def test_connect_maps_heterogeneous_slots_with_placeholder_gap(upw_vllm):
    engines = [RecordingVLLMEngine(), RecordingVLLMEngine()]
    args = _default_args(actor_num_gpus_per_node=8)

    def connect_as_rank(rank):
        obj = _make_instance(upw_vllm, args=args)

        def new_group(*, ranks, backend):
            assert backend == "gloo"
            return tuple(ranks)

        with patch("torch.distributed.get_rank", return_value=rank), patch(
            "torch.distributed.new_group", side_effect=new_group
        ):
            obj.connect_rollout_engines(
                engines,
                rollout_engine_lock=MagicMock(),
                engine_gpu_counts=[2, 3],
                engine_gpu_offsets=[0, 4],
            )
        return obj

    slot_zero = connect_as_rank(0)
    placeholder = connect_as_rank(2)
    slot_one = connect_as_rank(4)

    assert slot_zero._ipc_gather_group == (0, 1)
    assert slot_zero._ipc_gather_src == 0
    assert slot_zero._ipc_engine is engines[0]

    assert placeholder._ipc_gather_group is None
    assert placeholder._ipc_gather_src is None
    assert placeholder._ipc_engine is None

    assert slot_one._ipc_gather_group == (4, 5, 6)
    assert slot_one._ipc_gather_src == 4
    assert slot_one._ipc_engine is engines[1]


@pytest.mark.unit
def test_connect_keeps_colocated_engines_and_initializes_once(upw_vllm):
    engines = [RecordingVLLMEngine() for _ in range(2)]
    obj = _make_instance(
        upw_vllm,
        args=_default_args(actor_num_gpus_per_node=4, rollout_num_gpus_per_engine=2),
    )

    with patch("torch.distributed.get_rank", return_value=0):
        obj.connect_rollout_engines(
            engines,
            rollout_engine_lock=MagicMock(),
            engine_gpu_counts=[2, 2],
            engine_gpu_offsets=[0, 2],
        )

    assert obj.rollout_engines == engines
    assert obj.distributed_rollout_engines == []
    assert obj.use_distribute is False
    assert obj._ipc_gather_src == 0
    assert obj._ipc_gather_group == ((0, 1), "gloo")
    assert obj._ipc_engine is engines[0]
    assert obj._ipc_initialized is True
    assert len(engines[0].init_weight_transfer_engine.calls) == 1
    assert len(engines[1].init_weight_transfer_engine.calls) == 1

    engines2 = [RecordingVLLMEngine() for _ in range(2)]
    with patch("torch.distributed.get_rank", return_value=0):
        obj.connect_rollout_engines(
            engines2,
            rollout_engine_lock=MagicMock(),
            engine_gpu_counts=[2, 2],
            engine_gpu_offsets=[0, 2],
        )

    assert len(engines2[0].init_weight_transfer_engine.calls) == 0
    assert len(engines2[1].init_weight_transfer_engine.calls) == 0
