"""CUDA offload (torch_memory_saver) environment contract."""

from __future__ import annotations

import pytest

from vime.utils.offload import cuda_offload_env_vars


@pytest.mark.unit
@pytest.mark.parametrize(
    "library",
    [
        "torch_memory_saver_hook_mode_preload_cu13.abi3.so",
        "torch_memory_saver_hook_mode_preload_cu12.abi3.so",
        "torch_memory_saver_hook_mode_preload.abi3.so",
    ],
)
def test_cuda_offload_sets_preload_and_tms_switches(tmp_path, library):
    (tmp_path / library).write_bytes(b"")

    env = cuda_offload_env_vars({"EXISTING": "keep"}, library_dir=str(tmp_path))

    assert env["LD_PRELOAD"] == str(tmp_path / library)
    assert env["TMS_INIT_ENABLE"] == "1"
    assert env["TMS_INIT_ENABLE_CPU_BACKUP"] == "1"
    assert env["EXISTING"] == "keep", "the helper must not drop caller-supplied variables"


@pytest.mark.unit
def test_cuda_offload_prefers_the_newest_cuda_library(tmp_path):
    for library in (
        "torch_memory_saver_hook_mode_preload_cu12.abi3.so",
        "torch_memory_saver_hook_mode_preload_cu13.abi3.so",
    ):
        (tmp_path / library).write_bytes(b"")

    env = cuda_offload_env_vars({}, library_dir=str(tmp_path))

    assert env["LD_PRELOAD"].endswith("preload_cu13.abi3.so")


@pytest.mark.unit
def test_missing_tms_library_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="torch_memory_saver"):
        cuda_offload_env_vars({}, library_dir=str(tmp_path / "nonexistent"))


@pytest.mark.unit
def test_helper_does_not_mutate_the_caller_mapping(tmp_path):
    (tmp_path / "torch_memory_saver_hook_mode_preload.abi3.so").write_bytes(b"")
    original = {"EXISTING": "keep"}

    cuda_offload_env_vars(original, library_dir=str(tmp_path))

    assert original == {"EXISTING": "keep"}


@pytest.mark.unit
def test_default_policy_allows_suspending_an_allocated_region():
    from vime.utils.offload import OffloadPolicy

    assert OffloadPolicy().supports_disable is True


@pytest.mark.unit
def test_ascend_policy_refuses_to_suspend_an_allocated_region():
    # A subclass must be able to say no here; if the base class ever becomes a
    # dataclass again, the field default silently wins over this override.
    from vime_plugins.ascend.offload import NpuOffloadPolicy

    assert NpuOffloadPolicy().supports_disable is False


@pytest.mark.unit
def test_default_policy_runtime_env_carries_the_cuda_preload(tmp_path, monkeypatch):
    import types

    from vime.utils import offload

    (tmp_path / "torch_memory_saver_hook_mode_preload.abi3.so").write_bytes(b"")
    monkeypatch.setattr(offload, "cuda_offload_env_vars", lambda env: {**env, "LD_PRELOAD": "stub"})
    args = types.SimpleNamespace(offload_train=True, train_backend="megatron")

    assert offload.OffloadPolicy().runtime_env(args, {"A": "1"}) == {"A": "1", "LD_PRELOAD": "stub"}


@pytest.mark.unit
def test_default_policy_leaves_the_env_alone_without_offload():
    import types

    from vime.utils.offload import OffloadPolicy

    args = types.SimpleNamespace(offload_train=False, train_backend="megatron")

    assert OffloadPolicy().runtime_env(args, {"A": "1"}) == {"A": "1"}
