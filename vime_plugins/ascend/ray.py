"""How Ray addresses Ascend devices."""

from __future__ import annotations

from vime.ray.resources import RayAcceleratorResource

NPU_RESOURCE = RayAcceleratorResource(resource_name="NPU", native_gpu_resource=False)
