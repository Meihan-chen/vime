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
