"""CUDA offload (torch_memory_saver) environment contract."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from vime.utils.device_env import DeviceEnv


@pytest.fixture
def tms_library_dir(monkeypatch, tmp_path):
    module = ModuleType("torch_memory_saver")
    module.__file__ = str(tmp_path / "torch_memory_saver" / "__init__.py")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return tmp_path


@pytest.mark.unit
@pytest.mark.parametrize(
    "library",
    [
        "torch_memory_saver_hook_mode_preload_cu13.abi3.so",
        "torch_memory_saver_hook_mode_preload_cu12.abi3.so",
        "torch_memory_saver_hook_mode_preload.abi3.so",
    ],
)
def test_cuda_offload_sets_preload_and_tms_switches(tms_library_dir, library):
    (tms_library_dir / library).write_bytes(b"")

    env = DeviceEnv().train_offload_env(SimpleNamespace(), {"EXISTING": "keep"})

    assert env["LD_PRELOAD"] == str(tms_library_dir / library)
    assert env["TMS_INIT_ENABLE"] == "1"
    assert env["TMS_INIT_ENABLE_CPU_BACKUP"] == "1"
    assert env["EXISTING"] == "keep", "the helper must not drop caller-supplied variables"


@pytest.mark.unit
def test_cuda_offload_prefers_the_newest_cuda_library(tms_library_dir):
    for library in (
        "torch_memory_saver_hook_mode_preload_cu12.abi3.so",
        "torch_memory_saver_hook_mode_preload_cu13.abi3.so",
    ):
        (tms_library_dir / library).write_bytes(b"")

    env = DeviceEnv().train_offload_env(SimpleNamespace(), {})

    assert env["LD_PRELOAD"].endswith("preload_cu13.abi3.so")


@pytest.mark.unit
def test_missing_tms_library_raises(tms_library_dir):
    with pytest.raises(FileNotFoundError, match="torch_memory_saver"):
        DeviceEnv().train_offload_env(SimpleNamespace(), {})


@pytest.mark.unit
def test_offload_env_does_not_mutate_the_caller_mapping(tms_library_dir):
    (tms_library_dir / "torch_memory_saver_hook_mode_preload.abi3.so").write_bytes(b"")
    original = {"EXISTING": "keep"}

    DeviceEnv().train_offload_env(SimpleNamespace(), original)

    assert original == {"EXISTING": "keep"}


@pytest.mark.unit
def test_default_policy_allows_suspending_an_allocated_region():
    from vime.backends.megatron_integration import MegatronIntegration

    assert MegatronIntegration().supports_disable is True


@pytest.mark.unit
def test_ascend_policy_refuses_to_suspend_an_allocated_region():
    # A subclass must be able to say no here; if the base class ever becomes a
    # dataclass again, the field default silently wins over this override.
    from vime_plugins.ascend.training import AscendMegatronIntegration

    assert AscendMegatronIntegration().supports_disable is False
