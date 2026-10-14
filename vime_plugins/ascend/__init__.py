"""Ascend NPU support, registered against vime's accelerator-keyed registries.

Loaded by the accelerator package once the NPU backend has been selected, so
nothing here runs on a host that is not using NPUs.
"""

from __future__ import annotations


def register_all() -> None:
    from vime.backends import weight_transfer_registry
    from vime.ray import resources

    from .ray import NPU_RESOURCE
    from .weight_transfer import NpuWeightTransfer

    resources.register("npu", NPU_RESOURCE)
    weight_transfer_registry.register("npu", NpuWeightTransfer())
