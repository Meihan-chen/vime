"""CPU unit tests for ``vime.backends.megatron_utils.update_weight.update_weight_from_distributed``."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

_tests_root = Path(__file__).resolve().parents[1]
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import _unit_stubs
import pytest
import torch

MODULE_PATH = "vime.backends.megatron_utils.update_weight.update_weight_from_distributed"

NUM_GPUS = 0


# Modules stubbed by _install_stubs(). These are installed only for this module's
# tests and restored on teardown so they cannot affect sibling test collection.
_STUBBED_MODULES = (
    "megatron",
    "megatron.core",
    "megatron.core.parallel_state",
    "megatron.core.transformer",
    "megatron.core.transformer.transformer_layer",
    "ray",
    "ray.actor",
    "vime.utils.distributed_utils",
    "vllm",
    "vllm.utils",
    "vllm.utils.deep_gemm",
    "vllm.third_party",
    "vllm.third_party.deep_gemm",
    "vllm.third_party.deep_gemm.utils",
    "vllm.third_party.deep_gemm.utils.layout",
    "vllm.distributed",
    "vllm.distributed.weight_transfer",
    "vllm.distributed.weight_transfer.nccl_engine",
    "triton",
    "triton.language",
)


@pytest.fixture(scope="module")
def upw():
    saved = {k: sys.modules.get(k) for k in (*_STUBBED_MODULES, MODULE_PATH)}
    saved_platform = os.environ.get("VIME_PLATFORM")
    os.environ["VIME_PLATFORM"] = "cuda"
    # Pop first so _install_stubs()'s setdefault() actually installs the stubs (hermetic),
    # then drop the module-under-test so it re-imports against the stubs.
    for k in _STUBBED_MODULES:
        sys.modules.pop(k, None)
    _install_stubs()
    sys.modules.pop(MODULE_PATH, None)
    try:
        yield importlib.import_module(MODULE_PATH)
    finally:
        _unit_stubs.restore_sys_modules(saved)


def _install_stubs():
    _unit_stubs.install_megatron_mpu_stub()
    _unit_stubs.install_ray_stub()
    _unit_stubs.install_vime_distributed_utils_stub()
    _unit_stubs.install_triton_stub()


@dataclass
class _RemoteCall:
    args: tuple
    kwargs: dict


class RecordingRemoteMethod:
    def __init__(self, return_value: str = "ref"):
        self._return_value = return_value
        self.calls: list[_RemoteCall] = []

    def remote(self, *args, **kwargs):
        self.calls.append(_RemoteCall(args=args, kwargs=kwargs))
        return self._return_value


@dataclass
class RecordingEngine:
    update_weights_from_distributed: RecordingRemoteMethod = field(
        default_factory=lambda: RecordingRemoteMethod("ref")
    )
    init_weights_update_group: RecordingRemoteMethod = field(default_factory=lambda: RecordingRemoteMethod("init_ref"))
    destroy_weights_update_group: RecordingRemoteMethod = field(
        default_factory=lambda: RecordingRemoteMethod("destroy_ref")
    )
    start_weight_update: RecordingRemoteMethod = field(default_factory=lambda: RecordingRemoteMethod("start_ref"))
    finish_weight_update: RecordingRemoteMethod = field(default_factory=lambda: RecordingRemoteMethod("finish_ref"))


@dataclass
class RecordingLock:
    acquire: RecordingRemoteMethod = field(default_factory=lambda: RecordingRemoteMethod("acquired"))
    release: RecordingRemoteMethod = field(default_factory=lambda: RecordingRemoteMethod("released"))


@dataclass
class DummyGroup:
    token: str = "dummy"


def _real_tensors(n: int = 2):
    return [(f"layer.{i}.weight", torch.zeros(2, 2)) for i in range(n)]


def _patch_platform(
    monkeypatch,
    upw,
    *,
    send_seen: list[dict] | None = None,
    init_seen: list[dict] | None = None,
):
    def trainer_send_weights(iterator, *, group, packed):
        items = list(iterator)
        if send_seen is not None:
            send_seen.append({"items": items, "group": group, "packed": packed})

    def trainer_init(config):
        if init_seen is not None:
            init_seen.append(config)
        return DummyGroup("group-from-trainer-init")

    weight_transfer = types.SimpleNamespace(
        distributed_trainer_send_weights=MagicMock(side_effect=trainer_send_weights),
        distributed_trainer_init=MagicMock(side_effect=trainer_init),
    )
    platform = types.SimpleNamespace(is_npu=True, weight_transfer=weight_transfer)
    monkeypatch.setattr(upw, "current_platform", lambda: platform)
    return platform


def _patch_trainer_send(monkeypatch, upw, seen: list[dict]) -> None:
    _patch_platform(monkeypatch, upw, send_seen=seen)


def _make_instance(upw):
    obj = object.__new__(upw.UpdateWeightFromDistributed)
    obj.args = type("Args", (), {"update_weight_buffer_size": 1 << 30, "vllm_weight_sync_packed": True})()
    obj.model = []
    obj.weights_getter = lambda: {}
    obj.model_name = "test"
    obj.quantization_config = None
    obj.weight_version = 0
    obj._model_update_groups = DummyGroup()
    obj._hf_weight_iterator = None
    obj._is_pp_src_rank = True
    obj._group_name = "g"
    obj.rollout_engines = []
    obj.rollout_engine_lock = RecordingLock()
    return obj


@pytest.mark.unit
def test_signature_no_use_vllm(upw):
    sig = inspect.signature(upw.update_weights_from_distributed)
    params = sig.parameters
    assert "use_vllm" not in params
    for p in ("group_name", "group", "weight_version", "rollout_engines", "converted_named_tensors", "packed"):
        assert p in params


@pytest.mark.unit
def test_signature_rejects_legacy_use_vllm_call(upw):
    with pytest.raises(TypeError, match="use_vllm"):
        upw.update_weights_from_distributed(
            "g",
            DummyGroup(),
            1,
            [RecordingEngine()],
            _real_tensors(),
            use_vllm=True,
            packed=False,
        )


@pytest.mark.unit
def test_packed_true_uses_npu_trainer_send_weights(upw, monkeypatch):
    group = DummyGroup()
    engine = RecordingEngine()
    tensors = _real_tensors()
    seen = []
    _patch_trainer_send(monkeypatch, upw, seen)

    refs = upw.update_weights_from_distributed("groupA", group, 7, [engine], tensors, packed=True)

    assert len(seen) == 1
    sent = seen[0]["items"]
    assert [n for n, _ in sent] == [n for n, _ in tensors]
    assert seen[0]["group"] is group
    assert seen[0]["packed"] is True
    assert refs == ["ref"]


@pytest.mark.unit
def test_packed_false_still_uses_npu_trainer_send_weights(upw, monkeypatch):
    group = DummyGroup()
    engine = RecordingEngine()
    tensors = _real_tensors()
    seen = []
    _patch_trainer_send(monkeypatch, upw, seen)

    refs = upw.update_weights_from_distributed("groupB", group, 7, [engine], tensors, packed=False)

    assert len(seen) == 1
    assert len(seen[0]["items"]) == len(tensors)
    assert seen[0]["packed"] is False
    assert refs == ["ref"]


@pytest.mark.unit
def test_default_packed_is_false(upw, monkeypatch):
    group = DummyGroup()
    engine = RecordingEngine()
    seen = []
    _patch_trainer_send(monkeypatch, upw, seen)

    upw.update_weights_from_distributed("g", group, 1, [engine], _real_tensors())

    assert len(seen) == 1
    assert seen[0]["packed"] is False


@pytest.mark.unit
def test_no_dist_broadcast_fallback(upw, monkeypatch):
    import torch.distributed as dist

    seen_broadcast = []
    seen_send = []

    def fake_broadcast(*a, **k):
        seen_broadcast.append((a, k))

    monkeypatch.setattr(dist, "broadcast", fake_broadcast)
    _patch_trainer_send(monkeypatch, upw, seen_send)

    group = DummyGroup()
    engine = RecordingEngine()
    upw.update_weights_from_distributed("g", group, 1, [engine], _real_tensors(), packed=False)

    assert seen_broadcast == []
    assert len(seen_send) == 1


@pytest.mark.unit
def test_remote_kwargs_include_packed_true(upw, monkeypatch):
    group = DummyGroup()
    engine = RecordingEngine()
    tensors = _real_tensors(n=1)
    seen_send = []
    _patch_trainer_send(monkeypatch, upw, seen_send)

    upw.update_weights_from_distributed("myg", group, 42, [engine], tensors, packed=True)

    assert len(seen_send) == 1
    assert seen_send[0]["packed"] is True
    assert len(engine.update_weights_from_distributed.calls) == 1
    kw = engine.update_weights_from_distributed.calls[0].kwargs
    assert kw["packed"] is True
    assert kw["group_name"] == "myg"
    assert kw["weight_version"] == "42"
    assert kw["names"] == ["layer.0.weight"]
    assert kw["shapes"] == [torch.Size([2, 2])]
    assert kw["dtypes"] == [torch.float32]


@pytest.mark.unit
def test_remote_kwargs_include_packed_false(upw, monkeypatch):
    group = DummyGroup()
    engine = RecordingEngine()
    tensors = _real_tensors(n=2)
    seen_send = []
    _patch_trainer_send(monkeypatch, upw, seen_send)

    upw.update_weights_from_distributed("g", group, 99, [engine], tensors, packed=False)

    assert len(seen_send) == 1
    assert seen_send[0]["packed"] is False
    kw = engine.update_weights_from_distributed.calls[0].kwargs
    assert kw["packed"] is False
    assert kw["weight_version"] == "99"
    assert kw["names"] == ["layer.0.weight", "layer.1.weight"]


@pytest.mark.unit
def test_remote_kwargs_no_use_vllm(upw, monkeypatch):
    group = DummyGroup()
    engine = RecordingEngine()
    seen_send = []
    _patch_trainer_send(monkeypatch, upw, seen_send)

    upw.update_weights_from_distributed("g", group, 1, [engine], _real_tensors(), packed=False)

    assert len(seen_send) == 1
    kw = engine.update_weights_from_distributed.calls[0].kwargs
    assert "use_vllm" not in kw


@pytest.mark.unit
def test_multiple_engines_each_get_call(upw, monkeypatch):
    group = DummyGroup()
    engines = [RecordingEngine() for _ in range(3)]
    seen_send = []
    _patch_trainer_send(monkeypatch, upw, seen_send)

    upw.update_weights_from_distributed("g", group, 1, engines, _real_tensors(), packed=True)
    assert len(seen_send) == 1
    assert seen_send[0]["packed"] is True
    for e in engines:
        assert len(e.update_weights_from_distributed.calls) == 1


@pytest.mark.unit
def test_empty_tensor_list_still_dispatches(upw, monkeypatch):
    group = DummyGroup()
    engine = RecordingEngine()
    seen_send = []
    _patch_trainer_send(monkeypatch, upw, seen_send)

    refs = upw.update_weights_from_distributed("g", group, 1, [engine], [], packed=False)

    assert refs == ["ref"]
    kw = engine.update_weights_from_distributed.calls[0].kwargs
    assert kw["names"] == []
    assert kw["shapes"] == []
    assert len(seen_send) == 1
    assert seen_send[0]["items"] == []


@pytest.mark.unit
def test_raw_packed_path_sends_dense_chunks_only(upw, monkeypatch):
    obj = _make_instance(upw)
    obj._is_pp_src_rank = True
    obj._group_name = "g"
    obj._hf_weight_iterator = None
    obj._use_vllm_packed = lambda: True
    obj._iter_non_expert_chunks = lambda: iter([[("dense.0", torch.zeros(1))], [("dense.1", torch.zeros(1))]])
    obj._iter_expert_chunks = lambda: (_ for _ in ()).throw(AssertionError("expert pass should be skipped"))

    seen: list[tuple[list[str], bool, str]] = []
    monkeypatch.setattr(
        upw.UpdateWeightFromDistributed,
        "_update_bucket_weights_from_distributed",
        lambda self, converted_named_tensors, pbar=None, packed=False: seen.append(
            ([name for name, _ in converted_named_tensors], packed, pbar)
        ),
    )
    monkeypatch.setattr(upw.dist, "barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr(upw, "get_gloo_group", lambda: "gloo")

    upw.UpdateWeightFromDistributed._send_weights(obj, pbar="pbar")

    assert seen == [(["dense.0"], True, "pbar"), (["dense.1"], True, "pbar")]


@pytest.mark.unit
def test_raw_nonpacked_path_runs_dense_then_expert(upw, monkeypatch):
    obj = _make_instance(upw)
    obj._is_pp_src_rank = True
    obj._group_name = "g"
    obj._hf_weight_iterator = None
    obj._use_vllm_packed = lambda: False
    obj._iter_non_expert_chunks = lambda: iter([[("dense.0", torch.zeros(1))], [("dense.1", torch.zeros(1))]])
    obj._iter_expert_chunks = lambda: iter([[("expert.0", torch.zeros(1))]])

    seen: list[tuple[list[str], bool, str]] = []
    monkeypatch.setattr(
        upw.UpdateWeightFromDistributed,
        "_update_bucket_weights_from_distributed",
        lambda self, converted_named_tensors, pbar=None, packed=False: seen.append(
            ([name for name, _ in converted_named_tensors], packed, pbar)
        ),
    )
    barriers: list[str] = []
    monkeypatch.setattr(upw.dist, "barrier", lambda *args, **kwargs: barriers.append(kwargs.get("group")))
    monkeypatch.setattr(upw, "get_gloo_group", lambda: "gloo")

    upw.UpdateWeightFromDistributed._send_weights(obj, pbar="pbar")

    assert seen == [
        (["dense.0"], False, "pbar"),
        (["dense.1"], False, "pbar"),
        (["expert.0"], False, "pbar"),
    ]
    assert barriers == ["gloo", "gloo"]


@pytest.mark.unit
def test_bridge_path_forwards_packed_flag_and_listifies_chunks(upw, monkeypatch):
    obj = _make_instance(upw)
    obj._is_pp_src_rank = True
    obj._group_name = "g"
    obj.weights_getter = lambda: {"actor": torch.zeros(1)}
    obj._hf_weight_iterator = MagicMock()
    obj._hf_weight_iterator.get_hf_weight_chunks.return_value = iter(
        ((("bridge.0", torch.zeros(1)),), (("bridge.1", torch.zeros(1)),))
    )

    seen: list[tuple[list[str], bool, str]] = []
    monkeypatch.setattr(
        upw.UpdateWeightFromDistributed,
        "_update_bucket_weights_from_distributed",
        lambda self, converted_named_tensors, pbar=None, packed=False: seen.append(
            ([name for name, _ in converted_named_tensors], packed, pbar)
        ),
    )
    monkeypatch.setattr(upw.dist, "barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr(upw, "get_gloo_group", lambda: "gloo")

    upw.UpdateWeightFromDistributed._sync_bridge_weights_to_rollout_engines(obj, pbar="pbar", use_vllm_packed=True)

    assert seen == [
        (["bridge.0"], True, "pbar"),
        (["bridge.1"], True, "pbar"),
    ]


@pytest.mark.unit
def test_source_no_standalone_use_vllm_param(upw):
    src = inspect.getsource(upw)
    lines = [line.strip() for line in src.splitlines() if "use_vllm=" in line and "use_vllm_packed" not in line]
    assert lines == []


@pytest.mark.unit
def test_source_no_dist_broadcast_fallback(upw):
    src = inspect.getsource(upw)
    assert "dist.broadcast(" not in src


@pytest.mark.unit
def test_source_no_materialized_named_gpu_list(upw):
    src = inspect.getsource(upw.update_weights_from_distributed)
    assert "named_gpu = []" not in src
    assert "named_gpu_iter =" in src


@pytest.mark.unit
def test_connect_rollout_engines_uses_platform_collective_provider(upw, monkeypatch):
    args = type("Args", (), {"rollout_num_gpus_per_engine": 1})()
    engines = [RecordingEngine(), RecordingEngine()]
    seen: list[dict] = []

    platform = _patch_platform(
        monkeypatch,
        upw,
        init_seen=seen,
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr(upw.ray, "get", lambda refs: refs)
    monkeypatch.setattr(upw.ray._private.services, "get_node_ip_address", lambda: "127.0.0.1")
    cuda_synchronize = MagicMock()
    cuda_empty_cache = MagicMock()
    cuda_current_device = MagicMock(return_value=3)
    monkeypatch.setattr(upw.torch.cuda, "synchronize", cuda_synchronize)
    monkeypatch.setattr(upw.torch.cuda, "empty_cache", cuda_empty_cache)
    monkeypatch.setattr(upw.torch.cuda, "current_device", cuda_current_device)
    log_calls = []
    monkeypatch.setattr(upw.logger, "info", lambda *args: log_calls.append(args))

    group = upw.connect_rollout_engines_from_distributed(args, "g", engines, engine_gpu_counts=[1, 2])

    assert isinstance(group, DummyGroup)
    assert len(seen) == 1
    assert seen[0]["master_address"] == "127.0.0.1"
    assert seen[0]["world_size"] == 4  # 1 + (1 + 2)
    assert len(engines[0].init_weights_update_group.calls) == 1
    assert len(engines[1].init_weights_update_group.calls) == 1
    assert engines[0].init_weights_update_group.calls[0].kwargs["backend"] == "nccl"
    assert engines[1].init_weights_update_group.calls[0].kwargs["backend"] == "nccl"
    cuda_synchronize.assert_called_once_with()
    cuda_empty_cache.assert_called_once_with()
    cuda_current_device.assert_called_once_with()
    platform.weight_transfer.distributed_trainer_init.assert_called_once_with(seen[0])
    assert log_calls[-1][-1] == "3"


@pytest.mark.unit
def test_weight_update_session_calls_start_and_finish(upw, monkeypatch):
    import torch.distributed as dist

    engines = [RecordingEngine(), RecordingEngine()]
    ray_refs = []
    barrier_calls: list[object] = []

    def fake_barrier(*, group=None, **kwargs):
        barrier_calls.append(group)

    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "barrier", fake_barrier)
    monkeypatch.setattr(upw, "get_gloo_group", lambda: "dummy-gloo-group")
    monkeypatch.setattr(upw.ray, "get", lambda refs: ray_refs.extend(refs) or refs)

    upw._begin_vllm_weight_update_session(engines)
    upw._end_vllm_weight_update_session(engines)

    assert len(engines[0].start_weight_update.calls) == 1
    assert engines[0].start_weight_update.calls[0].kwargs["is_checkpoint_format"] is True
    assert len(engines[1].start_weight_update.calls) == 1
    assert len(engines[0].finish_weight_update.calls) == 1
    assert len(engines[1].finish_weight_update.calls) == 1
    assert barrier_calls == ["dummy-gloo-group", "dummy-gloo-group"]


@pytest.mark.unit
def test_source_wraps_sync_with_weight_update_session(upw):
    src = inspect.getsource(upw.UpdateWeightFromDistributed.update_weights)
    assert "_begin_vllm_weight_update_session" in src
    assert "_end_vllm_weight_update_session" in src
    assert "_send_weights" in src


@pytest.mark.unit
def test_source_keeps_main_nccl_sender_and_adds_npu_override(upw):
    src = inspect.getsource(upw.update_weights_from_distributed)
    connect_src = inspect.getsource(upw.connect_rollout_engines_from_distributed)
    assert "platform.weight_transfer.distributed_trainer_send_weights" in src
    assert "NCCLWeightTransferEngine.trainer_send_weights" in src
    assert "NCCLTrainerSendWeightsArgs" in src
    assert "platform.weight_transfer.distributed_trainer_init" in connect_src
    assert "NCCLWeightTransferEngine.trainer_init" in connect_src
    assert "weight_transfer_compat" not in src


@pytest.mark.unit
def test_cuda_path_keeps_main_nccl_sender(upw, monkeypatch):
    group = DummyGroup()
    engine = RecordingEngine()
    tensors = _real_tensors()
    seen = []

    monkeypatch.setattr(upw, "current_platform", lambda: types.SimpleNamespace(is_npu=False))
    monkeypatch.setattr(
        upw,
        "NCCLTrainerSendWeightsArgs",
        lambda *, group, packed: types.SimpleNamespace(group=group, packed=packed),
    )
    monkeypatch.setattr(
        upw.NCCLWeightTransferEngine,
        "trainer_send_weights",
        lambda iterator, args: seen.append((list(iterator), args)),
    )

    refs = upw.update_weights_from_distributed("g", group, 3, [engine], tensors, packed=True)

    assert refs == ["ref"]
    assert [name for name, _ in seen[0][0]] == [name for name, _ in tensors]
    assert seen[0][1].group is group
    assert seen[0][1].packed is True


@pytest.mark.unit
def test_cuda_sync_once_after_all_buckets_not_per_bucket(upw, monkeypatch):
    send_src = inspect.getsource(upw.update_weights_from_distributed)
    sync_src = inspect.getsource(upw.UpdateWeightFromDistributed.update_weights)
    assert "torch.cuda.synchronize" not in send_src
    assert "torch.cuda.synchronize" in sync_src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
