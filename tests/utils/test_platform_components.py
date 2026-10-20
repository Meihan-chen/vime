"""Shared component contracts for the existing default and Ascend integration."""

from __future__ import annotations

import sys
from argparse import Namespace
from types import ModuleType, SimpleNamespace

import pytest

from vime.backends.megatron_integration import MegatronIntegration
from vime.backends.vllm_weight_transfer import WeightTransfer
from vime.utils.accelerator.base import Accelerator
from vime.utils.accelerator.npu import NPUAccelerator
from vime.utils.device_env import DeviceEnv
from vime_plugins.ascend.weight_transfer import NpuWeightTransfer


def _npu_device_env():
    from vime_plugins.ascend.device_env import NpuDeviceEnv

    return NpuDeviceEnv()


def _npu_offload_policy():
    from vime_plugins.ascend.training import AscendMegatronIntegration

    return AscendMegatronIntegration()


# ── Ray resources ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("accelerator_cls", "resource_name", "native"),
    [(Accelerator, "GPU", True), (NPUAccelerator, "NPU", False)],
)
@pytest.mark.unit
def test_ray_resource_name_comes_from_the_accelerator(accelerator_cls, resource_name, native):
    # Ray resolves the visible-devices variable from the resource name, so the
    # name and that flag have to travel with the device.
    assert accelerator_cls.ray_resource_name == resource_name
    assert accelerator_cls.ray_uses_native_gpu_resource is native


@pytest.mark.parametrize(
    ("resource_name", "native", "bundle", "actor_options"),
    [
        ("GPU", True, {"GPU": 2, "CPU": 3}, {"num_gpus": 0.4}),
        ("NPU", False, {"NPU": 2, "CPU": 3}, {"num_gpus": 0, "resources": {"NPU": 0.4}}),
    ],
)
@pytest.mark.unit
def test_ray_request_shapes(monkeypatch, resource_name, native, bundle, actor_options):
    from vime.ray import resources

    monkeypatch.setattr(resources.accelerator, "ray_resource_name", lambda: resource_name)
    monkeypatch.setattr(resources.accelerator, "ray_uses_native_gpu_resource", lambda: native)

    assert resources.bundle(device_count=2, cpu_count=3) == bundle
    assert resources.actor_options(0.4) == actor_options


# ── Weight transfer and serving launch ──────────────────────────────────────


@pytest.mark.parametrize(
    "provider,distributed,ipc", [(WeightTransfer, "nccl", "ipc"), (NpuWeightTransfer, "hccl", "npu_ipc")]
)
@pytest.mark.unit
def test_transport_names_map_to_engines(provider, distributed, ipc):
    assert provider().backend("distributed") == distributed
    assert provider().backend("ipc") == ipc
    with pytest.raises(KeyError):
        provider().backend("typo")


@pytest.mark.parametrize(("accelerator_name", "backends"), [("cuda", ("nccl", "ipc")), ("npu", ("hccl", "npu_ipc"))])
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

        provider = WeightTransfer() if accelerator_name == "cuda" else NpuWeightTransfer()
        info = provider.init_info("ipc" if colocate else "distributed", rank=3, packed=True)

        assert isinstance(info, info_cls)
        assert info.backend == backend
        assert (info.rank, info.packed) == (3, True)

    # Only a vendor engine needs the general plugins loaded first.
    assert plugin_calls == (["load", "load"] if accelerator_name == "npu" else [])


@pytest.mark.unit
def test_default_subprocess_env_is_left_alone():
    base = {"KEEP": "1", "CUDA_VISIBLE_DEVICES": "0,1"}

    assert DeviceEnv().serving_subprocess_env(base, visible_devices="4,5", colocate=True) == base


@pytest.mark.unit
def test_npu_subprocess_env_replaces_cuda_and_rocm_visibility():
    env = _npu_device_env().serving_subprocess_env(
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
def test_npu_engine_actor_env_carries_the_vendor_settings(monkeypatch, tmp_path):
    toolkit = tmp_path / "toolkit"
    (toolkit / "python" / "site-packages" / "acl").mkdir(parents=True)
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(toolkit))
    args = Namespace(colocate=True)

    env = _npu_device_env().engine_actor_env(args, {"BASE": "1"})

    assert env["BASE"] == "1"
    assert env["VLLM_USE_AOT_COMPILE"] == "0"
    assert env["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:False"
    assert str(toolkit / "python" / "site-packages") in env["PYTHONPATH"]


@pytest.mark.unit
def test_default_engine_actor_env_is_the_identity():
    assert DeviceEnv().engine_actor_env(Namespace(colocate=True), {"BASE": "1"}) == {"BASE": "1"}


# ── Offload ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_npu_train_actor_env_carries_the_region_settings(monkeypatch, tmp_path):
    toolkit = tmp_path / "toolkit"
    (toolkit / "python" / "site-packages" / "acl").mkdir(parents=True)
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(toolkit))
    args = Namespace(offload_train=True, train_backend="megatron", colocate=True)

    env = _npu_device_env().train_actor_env(args, {"BASE": "1"})

    assert env["BASE"] == "1"
    assert env["TMS_HOOK_MODE"] == "torch"
    assert "TMS_REGION_TAG" not in env
    assert "TMS_ENABLE_CPU_BACKUP" not in env
    assert env["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:False"
    assert str(toolkit / "python" / "site-packages") in env["PYTHONPATH"]


@pytest.mark.parametrize("device_env", [DeviceEnv, _npu_device_env])
@pytest.mark.unit
def test_train_actor_env_is_untouched_without_offload(device_env):
    args = Namespace(offload_train=False, train_backend="megatron", colocate=True)

    assert device_env().train_actor_env(args, {"BASE": "1"}) == {"BASE": "1"}


@pytest.mark.unit
def test_npu_does_not_enable_megatron_offload_for_other_backends():
    args = Namespace(offload_train=True, train_backend="fsdp", colocate=True)
    assert _npu_device_env().train_actor_env(args, {"BASE": "1"}) == {"BASE": "1"}


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

    MegatronIntegration().initialize_optimizer_state(optimizer)


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


@pytest.mark.parametrize("policy", [MegatronIntegration, _npu_offload_policy])
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
