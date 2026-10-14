"""Ascend NPU support, registered against vime's accelerator-keyed registries.

Loaded by the accelerator package once the NPU backend has been selected, so
nothing here runs on a host that is not using NPUs.
"""

from __future__ import annotations


def register_all() -> None:
    from vime.ray import resources

    from .ray import NPU_RESOURCE

    resources.register("npu", NPU_RESOURCE)
