"""What each seam does when the thing it describes is absent.

Every seam falls back to the behaviour the GPU path uses, which is right for a
host that has no accelerator of its own but wrong for one that has the device
and not its support package. These pin which case gets which answer.
"""

from __future__ import annotations

import pytest

from vime.utils import accelerator


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
