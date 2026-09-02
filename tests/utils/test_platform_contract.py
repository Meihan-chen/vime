from __future__ import annotations

import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest

from vime.platforms import current_platform, reset_platform_cache


@pytest.fixture(autouse=True)
def _reset_platform_selection():
    reset_platform_cache()
    yield
    reset_platform_cache()


def test_vime_platform_override_selects_cuda(monkeypatch):
    monkeypatch.setenv("VIME_PLATFORM", "cuda")

    platform = current_platform()

    assert platform.name == "cuda"
    assert platform.ray.resource_name == "GPU"
    assert platform.ray.visible_devices_env == "CUDA_VISIBLE_DEVICES"
    assert platform.checkpoint.default_megatron_to_hf_mode == "raw"


def test_vime_platform_override_selects_npu_without_vendor_import(monkeypatch):
    monkeypatch.setenv("VIME_PLATFORM", "npu")
    before = {name for name in ("torch_npu", "vllm_ascend", "mindspeed") if name in sys.modules}

    platform = current_platform()

    after = {name for name in ("torch_npu", "vllm_ascend", "mindspeed") if name in sys.modules}
    assert platform.name == "npu"
    assert platform.ray.resource_name == "NPU"
    assert platform.checkpoint.default_megatron_to_hf_mode == "bridge"
    assert before == after


def test_unknown_explicit_platform_has_clear_error(monkeypatch):
    monkeypatch.setenv("VIME_PLATFORM", "not-registered")

    with pytest.raises(ValueError, match="Unknown Vime platform"):
        current_platform()


def test_npu_auto_detection_does_not_import_vendor_without_device_nodes(monkeypatch):
    from vime.platforms import npu

    monkeypatch.setattr(npu.os.path, "exists", lambda path: False)
    monkeypatch.setattr(npu, "glob", lambda pattern: [])

    def fail_import(name):
        raise AssertionError(f"unexpected import during negative NPU detection: {name}")

    monkeypatch.setattr(npu.importlib, "import_module", fail_import)

    assert npu.detect_npu() is False


def test_npu_safe_empty_cache_wraps_original_once(monkeypatch):
    from vime.platforms import npu

    calls = []

    def original_empty_cache():
        calls.append("original")
        raise RuntimeError("allocator is between offload states")

    fake_torch = SimpleNamespace(
        npu=SimpleNamespace(empty_cache=original_empty_cache),
        cuda=SimpleNamespace(empty_cache=lambda: None),
    )
    monkeypatch.setattr(npu.importlib, "import_module", lambda name: fake_torch if name == "torch" else None)

    npu._install_safe_empty_cache()
    wrapped = fake_torch.npu.empty_cache
    wrapped()
    npu._install_safe_empty_cache()

    assert calls == ["original"]
    assert fake_torch.npu.empty_cache is wrapped
    assert fake_torch.cuda.empty_cache is wrapped


@pytest.mark.parametrize(
    ("name", "bundle", "actor_options"),
    [
        ("cuda", {"GPU": 2, "CPU": 3}, {"num_gpus": 0.4}),
        ("npu", {"NPU": 2, "CPU": 3}, {"resources": {"NPU": 0.4}}),
    ],
)
def test_ray_resource_contract(monkeypatch, name, bundle, actor_options):
    monkeypatch.setenv("VIME_PLATFORM", name)
    ray_spec = current_platform().ray

    assert ray_spec.bundle_resources(device_count=2, cpu_count=3) == bundle
    assert ray_spec.actor_options(0.4) == actor_options


def test_npu_runtime_env_is_scoped_to_npu_provider(monkeypatch, tmp_path):
    toolkit = tmp_path / "toolkit"
    (toolkit / "python" / "site-packages" / "acl").mkdir(parents=True)
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(toolkit))
    monkeypatch.setenv("VIME_PLATFORM", "npu")
    args = Namespace(offload_train=True, train_backend="megatron", colocate=True)

    train_env = current_platform().ray.train_runtime_env(args, {"BASE": "1"})
    rollout_env = current_platform().ray.rollout_runtime_env(args, {"BASE": "1"})

    assert train_env["TMS_HOOK_MODE"] == "torch"
    assert train_env["TMS_REGION_TAG"] == "training"
    assert train_env["TMS_ENABLE_CPU_BACKUP"] == "1"
    assert train_env["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:False"
    assert str(toolkit / "python" / "site-packages") in train_env["PYTHONPATH"]
    assert rollout_env["VLLM_USE_AOT_COMPILE"] == "0"
    assert rollout_env["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:False"


def test_memory_utils_keep_main_cuda_compatibility_surface(monkeypatch):
    from vime.utils import memory_utils

    calls = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            synchronize=lambda: calls.append("synchronize"),
            empty_cache=lambda: calls.append("empty_cache"),
        ),
        _C=SimpleNamespace(_host_emptyCache=lambda: calls.append("empty_host_cache")),
    )
    monkeypatch.setattr(memory_utils, "torch", fake_torch)
    monkeypatch.setattr(memory_utils.gc, "collect", lambda: calls.append("gc"))

    memory_utils.clear_memory(clear_host_memory=True)

    assert calls == ["synchronize", "gc", "empty_cache", "empty_host_cache"]
