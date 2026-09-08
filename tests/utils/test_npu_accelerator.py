"""CPU contracts for NPU selection and pre-Megatron bootstrap ordering."""

from types import SimpleNamespace

import pytest
import torch

from vime.platforms import current_platform, get_platform, reset_platform_cache
from vime.platforms import npu
from vime.platforms.npu import NPUAccelerator
from vime.utils import accelerator


@pytest.fixture(autouse=True)
def npu_runtime(monkeypatch):
    reset_platform_cache()
    monkeypatch.setenv("VIME_PLATFORM", "npu")
    monkeypatch.delenv("VIME_ACCELERATOR", raising=False)
    monkeypatch.delenv("MUSA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("MUSA_PATCH_PATH", raising=False)
    monkeypatch.setattr(accelerator, "_REGISTRY", {})
    monkeypatch.setattr(accelerator, "_ACCELERATOR", None)
    monkeypatch.setattr(accelerator, "_cuda_available", lambda: False)
    monkeypatch.setattr(accelerator, "is_musa_available", lambda: False)
    calls = []
    fake_npu = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 1,
        device_count=lambda: 2,
        set_device=lambda index: calls.append(("set_device", index)),
        synchronize=lambda: calls.append("synchronize"),
        empty_cache=lambda: calls.append("empty_cache"),
        ipc_collect=lambda: calls.append("ipc_collect"),
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    yield calls
    reset_platform_cache()


def test_platform_registers_npu_before_main_auto_selection(monkeypatch, npu_runtime):
    # MindSpeed may also make CUDA's availability probe return true.
    monkeypatch.setattr(accelerator, "_cuda_available", lambda: True)
    assert current_platform().is_npu
    assert isinstance(accelerator.initialize_accelerator(), NPUAccelerator)
    assert accelerator.device_type() == "npu"
    assert accelerator.process_group_backend() == "hccl"
    assert accelerator.process_group_backend("gloo") == "gloo"
    assert accelerator.is_accelerator_backend("cpu:gloo,npu:hccl")
    assert not accelerator.is_accelerator_backend("gloo")
    assert accelerator.distributed_device_id() is None
    assert accelerator.visible_devices_env_key() == "ASCEND_RT_VISIBLE_DEVICES"
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,7")
    assert accelerator.resolve_visible_device_id("7") == 1
    accelerator.set_device(1)
    accelerator.synchronize()
    accelerator.ipc_collect()
    accelerator.empty_cache()
    assert npu_runtime == [("set_device", 1), "synchronize", "ipc_collect", "empty_cache"]


def test_npu_allocator_remains_owned_by_existing_runtime_hooks(monkeypatch):
    monkeypatch.setenv("VIME_ENABLE_EXPANDABLE_SEGMENTS", "1")
    assert NPUAccelerator().set_allocator_expandable_segments() is False


def test_main_npu_override_selects_matching_platform(monkeypatch):
    monkeypatch.delenv("VIME_PLATFORM")
    monkeypatch.setenv("VIME_ACCELERATOR", "npu")
    assert current_platform().is_npu
    assert accelerator.get_accelerator().name == "npu"


@pytest.mark.parametrize("platform,backend", [("npu", "cuda"), ("cuda", "npu"), ("npu", "musa")])
def test_conflicting_overrides_fail_before_bootstrap(monkeypatch, platform, backend):
    monkeypatch.setenv("VIME_PLATFORM", platform)
    monkeypatch.setenv("VIME_ACCELERATOR", backend)
    with pytest.raises(ValueError, match="Conflicting VIME_PLATFORM"):
        current_platform()


def test_registered_npu_does_not_override_explicit_cuda_platform(monkeypatch):
    get_platform("npu")
    monkeypatch.setenv("VIME_PLATFORM", "cuda")
    monkeypatch.setattr(accelerator, "_cuda_available", lambda: True)
    monkeypatch.setattr(accelerator.CUDAAccelerator, "is_available", lambda self: True)
    assert current_platform().name == "cuda"
    assert accelerator.initialize_accelerator().name == "cuda"


def test_bootstrap_selects_npu_before_mindspeed_and_attention(monkeypatch):
    events = []
    bootstrap = current_platform().megatron
    monkeypatch.setattr(npu, "_ensure_torch_npu", lambda: events.append("torch_npu"))
    monkeypatch.setattr(npu, "_install_safe_empty_cache", lambda: events.append("empty_cache_guard"))
    original_import = npu.importlib.import_module

    def import_module(name, *args, **kwargs):
        if name in {"mindspeed.megatron_adaptor", "vime.backends.megatron_utils.npu_attention_patch"}:
            assert accelerator.get_accelerator().name == "npu"
            events.append(name)
            bootstrap.bootstrap()  # Recursive imports must not repeat initialization.
            return SimpleNamespace()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(npu.importlib, "import_module", import_module)
    bootstrap.bootstrap()
    bootstrap.bootstrap()
    assert events == [
        "torch_npu",
        "empty_cache_guard",
        "mindspeed.megatron_adaptor",
        "vime.backends.megatron_utils.npu_attention_patch",
    ]


def test_bootstrap_rejects_preselected_cuda_without_replacing_it(monkeypatch):
    selected = accelerator.CUDAAccelerator()
    monkeypatch.setattr(accelerator, "_ACCELERATOR", selected)
    monkeypatch.setattr(npu, "_ensure_torch_npu", lambda: None)
    bootstrap = current_platform().megatron
    with pytest.raises(RuntimeError, match="already selected 'cuda'"):
        bootstrap.bootstrap()
    assert accelerator._ACCELERATOR is selected
    assert not bootstrap._bootstrapping
    assert not bootstrap._bootstrapped
