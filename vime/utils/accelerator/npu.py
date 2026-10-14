"""Ascend NPU accelerator implementation.

Importing this module never imports ``torch_npu``. The runtime bootstrap in
``vime.utils.accelerator`` may attach ``torch.npu`` before selection.
"""

from __future__ import annotations

import os
from glob import glob
from typing import Any

import torch

from .torch_accelerator import TorchAccelerator


def npu_module() -> Any:
    return getattr(torch, "npu", None)


def has_visible_device() -> bool:
    """Whether an Ascend device is exposed to this process.

    Checked before anything would import ``torch_npu``, so a host that merely
    has it installed does not get the NPU backend selected for it.
    """
    return os.path.exists("/dev/davinci_manager") or bool(glob("/dev/davinci[0-9]*"))


def is_npu_available() -> bool:
    if not has_visible_device():
        return False
    module = npu_module()
    checker = getattr(module, "is_available", None)
    return bool(module is not None and checker is not None and checker())


class NPUAccelerator(TorchAccelerator):
    name = "npu"
    device_type = "npu"
    communication_backend_name = "hccl"

    # Ray resolves the visible-devices variable from the resource name, and only
    # its own NPU resource maps to ASCEND_RT_VISIBLE_DEVICES.
    ray_resource_name = "NPU"
    ray_uses_native_gpu_resource = False

    @property
    def visible_devices_env(self) -> str:
        return "ASCEND_RT_VISIBLE_DEVICES"

    def _module(self) -> Any:
        module = npu_module()
        if module is None:
            raise RuntimeError("Ascend NPU backend requires torch_npu and a runtime that exposes torch.npu")
        return module

    def is_available(self) -> bool:
        return is_npu_available()

    def weight_update_backend(self, default: str = "nccl") -> str:
        return "cpu:gloo,npu:hccl" if default == "nccl" else default

    def distributed_device_id(self, index: int | str | torch.device | None = None) -> None:
        # HCCL selects the local device through torch.npu.set_device/LOCAL_RANK.
        return None

    def empty_cache(self) -> None:
        module = self._module()
        is_capturing = getattr(module, "is_current_stream_capturing", None)
        if is_capturing is not None and is_capturing():
            return
        module.empty_cache()

    def set_allocator_expandable_segments(self) -> bool:
        # Allocator policy is carried by the TMS and runtime-env hooks instead.
        return False

    def weight_transfer_backend(self, transport: str) -> str:
        return {"ipc": "npu_ipc", "nccl": "hccl"}.get(transport, transport)

    def trainer_init_info(self, *, colocate: bool, **kwargs: Any) -> Any:
        import torch_npu  # noqa: F401
        from vllm.plugins import load_general_plugins

        # A trainer, unlike a vLLM worker, has not loaded the general plugins.
        load_general_plugins()
        if colocate:
            from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCTrainerInitInfo

            return NPUIPCTrainerInitInfo(**kwargs)
        from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLTrainerInitInfo

        return HCCLTrainerInitInfo(**kwargs)

    def device_uuid(self, index: int | str | None = None) -> str:
        # The host-IP/physical-chip identifier vLLM Ascend derives, so a trainer
        # and a receiver on the same host agree on the mapping.
        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import npu_generate_uuid

        return npu_generate_uuid()

    def supports(self, capability: str) -> bool:
        if capability in {
            "cuda_int4_extension",
            "nvml_affinity",
            # Checkpoints are loaded from HF directly rather than converted.
            "torch_dist_checkpoint_convert",
            "vllm_fp8_utils",
            "triton_kernels",
        }:
            return False
        if capability == "requires_cpu_initialization":
            return True
        if capability == "bf16":
            checker = getattr(self._module(), "is_bf16_supported", None)
            return bool(checker and checker())
        return super().supports(capability)
