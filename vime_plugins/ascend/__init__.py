"""Ascend NPU support, registered against vime's accelerator-keyed registries.

Loaded by the accelerator package once the NPU backend has been selected, so
nothing here runs on a host that is not using NPUs.
"""

from __future__ import annotations


def register_all() -> None:
    from vime.backends import weight_transfer_registry
    from vime.ray import resources
    from vime.utils import offload, vendor_hooks

    from .megatron_bootstrap import after_megatron_args, before_megatron_import
    from .offload import NpuOffloadPolicy
    from .ray import NPU_RESOURCE
    from .weight_transfer import NpuWeightTransfer

    resources.register("npu", NPU_RESOURCE)
    weight_transfer_registry.register("npu", NpuWeightTransfer())
    offload.register("npu", NpuOffloadPolicy())
    vendor_hooks.register("before_megatron_import", before_megatron_import)
    vendor_hooks.register("after_megatron_args", after_megatron_args)
