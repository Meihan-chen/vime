"""Behaviour of the accelerator-keyed registries, on both the default and NPU.

Each registry answers one question for the accelerator in use. An accelerator
with no entry gets the default, which is CUDA's behaviour, so the pairs below
assert both sides of every question.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from types import ModuleType, SimpleNamespace

import pytest

from vime.backends.weight_transfer_registry import WeightTransfer
from vime.ray.resources import RayAcceleratorResource
from vime.utils.offload import OffloadPolicy


def _npu_ray_resource():
    from vime_plugins.ascend.ray import NPU_RESOURCE

    return NPU_RESOURCE


def _npu_weight_transfer():
    from vime_plugins.ascend.weight_transfer import NpuWeightTransfer

    return NpuWeightTransfer()


def _npu_offload_policy():
    from vime_plugins.ascend.offload import NpuOffloadPolicy

    return NpuOffloadPolicy()


# ── Ray resources ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("resource", "bundle", "actor_options"),
    [
        (RayAcceleratorResource(), {"GPU": 2, "CPU": 3}, {"num_gpus": 0.4}),
        (_npu_ray_resource, {"NPU": 2, "CPU": 3}, {"num_gpus": 0, "resources": {"NPU": 0.4}}),
    ],
)
@pytest.mark.unit
def test_ray_resource_shapes(resource, bundle, actor_options):
    resource = resource() if callable(resource) else resource

    assert resource.bundle(device_count=2, cpu_count=3) == bundle
    assert resource.actor_options(0.4) == actor_options


@pytest.mark.unit
def test_an_unregistered_accelerator_falls_back_to_ray_gpus(monkeypatch):
    from vime.ray import resources

    monkeypatch.setattr(resources.accelerator, "device_type", lambda: "something-else")

    assert resources.current() is resources._DEFAULT
    assert resources.current().native_gpu_resource is True


@pytest.mark.unit
def test_a_registered_accelerator_is_resolved_by_name(monkeypatch):
    from vime.ray import resources

    monkeypatch.setitem(resources._REGISTRY, "npu", _npu_ray_resource())
    monkeypatch.setattr(resources.accelerator, "device_type", lambda: "npu")

    assert resources.current().resource_name == "NPU"


# ── Weight transfer and serving launch ──────────────────────────────────────


@pytest.mark.parametrize(
    ("transfer", "nccl_name", "ipc_name"),
    [(WeightTransfer(), "nccl", "ipc"), (_npu_weight_transfer, "hccl", "npu_ipc")],
)
@pytest.mark.unit
def test_transport_names_map_to_engines(transfer, nccl_name, ipc_name):
    transfer = transfer() if callable(transfer) else transfer

    assert transfer.backend("nccl") == nccl_name
    assert transfer.backend("ipc") == ipc_name
    for passthrough in ("npu_ipc", "hccl", "custom"):
        assert transfer.backend(passthrough) == passthrough


@pytest.mark.parametrize(
    ("accelerator_name", "backends"), [("cuda", ("nccl", "ipc")), ("npu", ("hccl", "npu_ipc"))]
)
@pytest.mark.unit
def test_trainer_init_info_matches_the_selected_engine(monkeypatch, accelerator_name, backends):
    plugin_calls = []
    for colocate, backend in zip((False, True), backends, strict=True):
        if accelerator_name == "cuda":
            module_name = f"vllm.distributed.weight_transfer.{backend}_engine"
            cls_name = "IPCTrainerInitInfo" if colocate else "NCCLTrainerInitInfo"
        else:
            module_name = f"vllm_ascend.distributed.weight_transfer.{backend}_engine"
            cls_name = "NPUIPCTrainerInitInfo" if colocate else "HCCLTrainerInitInfo"
        module = ModuleType(module_name)
        info_cls = type(cls_name, (SimpleNamespace,), {"backend": backend})
        setattr(module, cls_name, info_cls)
        monkeypatch.setitem(sys.modules, module_name, module)

        plugins = ModuleType("vllm.plugins")
        plugins.load_general_plugins = lambda: plugin_calls.append("load")
        monkeypatch.setitem(sys.modules, "vllm.plugins", plugins)
        monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))

        transfer = WeightTransfer() if accelerator_name == "cuda" else _npu_weight_transfer()
        info = transfer.trainer_init_info(colocate=colocate, rank=3, packed=True)

        assert isinstance(info, info_cls)
        assert info.backend == backend
        assert (info.rank, info.packed) == (3, True)

    # Only a vendor engine needs the general plugins loaded first.
    assert plugin_calls == (["load", "load"] if accelerator_name == "npu" else [])


@pytest.mark.unit
def test_default_subprocess_env_is_left_alone():
    base = {"KEEP": "1", "CUDA_VISIBLE_DEVICES": "0,1"}

    assert WeightTransfer().subprocess_env(base, visible_devices="4,5", colocate=True) == base


@pytest.mark.unit
def test_npu_subprocess_env_replaces_cuda_and_rocm_visibility():
    env = _npu_weight_transfer().subprocess_env(
        {
            "KEEP": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "HIP_VISIBLE_DEVICES": "0,1",
        },
        visible_devices="4,5",
        colocate=True,
    )

    assert env["KEEP"] == "1"
    assert "PYTORCH_CUDA_ALLOC_CONF" not in env
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert "HIP_VISIBLE_DEVICES" not in env
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "4,5"
    assert env["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:False"


@pytest.mark.unit
def test_npu_engine_runtime_env_carries_the_vendor_settings(monkeypatch, tmp_path):
    toolkit = tmp_path / "toolkit"
    (toolkit / "python" / "site-packages" / "acl").mkdir(parents=True)
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(toolkit))
    args = Namespace(colocate=True)

    env = _npu_weight_transfer().engine_runtime_env(args, {"BASE": "1"})

    assert env["BASE"] == "1"
    assert env["VLLM_USE_AOT_COMPILE"] == "0"
    assert env["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:False"
    assert str(toolkit / "python" / "site-packages") in env["PYTHONPATH"]


@pytest.mark.unit
def test_default_engine_runtime_env_is_the_identity():
    assert WeightTransfer().engine_runtime_env(Namespace(colocate=True), {"BASE": "1"}) == {"BASE": "1"}


# ── Offload ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_npu_offload_runtime_env_carries_the_region_settings(monkeypatch, tmp_path):
    toolkit = tmp_path / "toolkit"
    (toolkit / "python" / "site-packages" / "acl").mkdir(parents=True)
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(toolkit))
    args = Namespace(offload_train=True, train_backend="megatron", colocate=True)

    env = _npu_offload_policy().runtime_env(args, {"BASE": "1"})

    assert env["BASE"] == "1"
    assert env["TMS_HOOK_MODE"] == "torch"
    assert env["TMS_REGION_TAG"] == "training"
    assert env["TMS_ENABLE_CPU_BACKUP"] == "1"
    assert env["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:False"
    assert str(toolkit / "python" / "site-packages") in env["PYTHONPATH"]


@pytest.mark.parametrize("policy", [OffloadPolicy, _npu_offload_policy])
@pytest.mark.unit
def test_offload_runtime_env_is_untouched_without_offload(policy):
    args = Namespace(offload_train=False, train_backend="megatron", colocate=True)

    assert policy().runtime_env(args, {"BASE": "1"}) == {"BASE": "1"}


@pytest.mark.parametrize("chained", [False, True])
@pytest.mark.unit
def test_npu_optimizer_state_is_materialized_before_leaving_the_region(chained):
    calls = []

    def init_state(optimizer, config):
        calls.append((optimizer, config))

    optimizers = [
        SimpleNamespace(optimizer=object(), config=object(), init_state_fn=init_state)
        for _ in range(2 if chained else 1)
    ]
    optimizer = SimpleNamespace(chained_optimizers=optimizers) if chained else optimizers[0]

    _npu_offload_policy().initialize_optimizer_state(optimizer)

    assert calls == [(opt.optimizer, opt.config) for opt in optimizers]


@pytest.mark.parametrize("chained", [False, True])
@pytest.mark.unit
def test_default_policy_has_no_optimizer_state_to_materialize(chained):
    optimizers = [SimpleNamespace(optimizer=object(), config=object(), init_state_fn=_fail)]
    optimizer = SimpleNamespace(chained_optimizers=optimizers) if chained else optimizers[0]

    OffloadPolicy().initialize_optimizer_state(optimizer)


def _fail(*args):
    raise AssertionError("the default policy allocates outside a region and has nothing to materialize")


@pytest.mark.parametrize("empty_optimizer", [False, True])
@pytest.mark.unit
def test_npu_skips_an_optimizer_with_nothing_to_materialize(empty_optimizer):
    optimizer = SimpleNamespace(
        optimizer=None if empty_optimizer else object(),
        config=object(),
        init_state_fn=_fail if empty_optimizer else None,
    )

    _npu_offload_policy().initialize_optimizer_state(optimizer)


@pytest.mark.parametrize("policy", [OffloadPolicy, _npu_offload_policy])
@pytest.mark.unit
def test_eval_only_has_no_optimizer_state_to_materialize(policy):
    policy().initialize_optimizer_state(None)


# ── Accelerator surface ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_memory_utils_keep_main_accelerator_surface(monkeypatch):
    from vime.utils import memory_utils

    calls = []
    fake_accelerator = SimpleNamespace(
        synchronize=lambda: calls.append("synchronize"),
        empty_cache=lambda: calls.append("empty_cache"),
    )
    fake_torch = SimpleNamespace(_C=SimpleNamespace(_host_emptyCache=lambda: calls.append("empty_host_cache")))
    monkeypatch.setattr(memory_utils, "accelerator", fake_accelerator)
    monkeypatch.setattr(memory_utils, "torch", fake_torch)
    monkeypatch.setattr(memory_utils.gc, "collect", lambda: calls.append("gc"))

    memory_utils.clear_memory(clear_host_memory=True)

    assert calls == ["synchronize", "gc", "empty_cache", "empty_host_cache"]
