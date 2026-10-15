"""Ascend NPU support, registered against vime's accelerator-keyed registries.

Loaded by the accelerator package once the NPU backend has been selected, so
nothing here runs on a host that is not using NPUs.
"""

from __future__ import annotations


def register_all() -> None:
    from vime.utils import device_env, offload, vendor_hooks

    from .device_env import NpuDeviceEnv
    from .offload import NpuOffloadPolicy
    from .patches import after_megatron_args, before_megatron_import, relax_global_plan_validation

    device_env.register("npu", NpuDeviceEnv())
    offload.register("npu", NpuOffloadPolicy())
    vendor_hooks.register("before_megatron_import", before_megatron_import)
    vendor_hooks.register("after_megatron_args", after_megatron_args)
    vendor_hooks.register("after_dcp_planner_patch", relax_global_plan_validation)
