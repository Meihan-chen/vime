"""What each seam does when the thing it describes is absent.

Every seam falls back to the behaviour the GPU path uses, which is right for a
host that has no accelerator of its own but wrong for one that has the device
and not its support package. These pin which case gets which answer.
"""

from __future__ import annotations

import argparse

import pytest

from vime.utils import accelerator, device_env
from vime.utils.accelerator.cuda import CUDAAccelerator
from vime.utils.accelerator.npu import NPUAccelerator


@pytest.mark.unit
def test_resource_name_needs_no_usable_device(monkeypatch):
    # A driver builds scheduling requests without owning a device.
    monkeypatch.setattr(accelerator, "_ACCELERATOR", None)
    monkeypatch.setattr(accelerator, "_requested_name", lambda: None)
    monkeypatch.setattr(accelerator, "initialize_accelerator", lambda: None)

    assert accelerator.ray_resource_name() == "GPU"
    assert accelerator.ray_uses_native_gpu_resource() is True


@pytest.mark.unit
def test_a_requested_backend_describes_itself_without_being_available(monkeypatch):
    monkeypatch.setattr(accelerator, "_ACCELERATOR", None)
    monkeypatch.setattr(accelerator, "_requested_name", lambda: "npu")

    def unusable():
        raise AssertionError("describing a backend must not check its runtime")

    monkeypatch.setattr(accelerator, "initialize_accelerator", unusable)

    assert accelerator.ray_resource_name() == "NPU"
    assert accelerator.ray_uses_native_gpu_resource() is False


@pytest.mark.unit
def test_an_unknown_requested_backend_is_still_an_error(monkeypatch):
    monkeypatch.setattr(accelerator, "_ACCELERATOR", None)
    monkeypatch.setattr(accelerator, "_requested_name", lambda: "nosuchdevice")

    with pytest.raises(ValueError, match="nosuchdevice"):
        accelerator.ray_resource_name()


@pytest.mark.unit
def test_a_missing_support_package_is_fatal(monkeypatch):
    # Falling back would run the device under another device's policies.
    monkeypatch.setenv("VIME_PLUGINS", "vime_plugins.does_not_exist")

    with pytest.raises(RuntimeError, match="VIME_PLUGINS"):
        accelerator._load_plugin("npu")


@pytest.mark.unit
def test_an_empty_plugin_setting_registers_nothing(monkeypatch):
    monkeypatch.setenv("VIME_PLUGINS", "")

    assert accelerator._load_plugin("npu") is None


@pytest.mark.unit
def test_offload_env_follows_offload_train_alone():
    # The shared default must match what the GPU path did before the seam.
    production_shaped = argparse.Namespace(offload_train=True, colocate=True)

    assert device_env.offload_requested(production_shaped) is True
    assert device_env.offload_requested(argparse.Namespace(offload_train=False)) is False


@pytest.mark.unit
def test_cuda_answers_for_a_named_device_and_ascend_refuses():
    # CUDA can identify any local device; Ascend's identifier is per process.
    assert "index" in CUDAAccelerator.device_uuid.__code__.co_varnames

    with pytest.raises(NotImplementedError, match="per device index"):
        NPUAccelerator.device_uuid(NPUAccelerator(), 3)
