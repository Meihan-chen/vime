"""How Ray advertises and addresses one accelerator.

Ray models an accelerator as a resource name plus a visible-devices variable,
and exposes whole families through ``ray._private.accelerators``. This mirrors
that model for the accelerators vime runs on. Anything not registered falls
back to Ray's own GPU resource, which is what CUDA and ROCm want.
"""

from __future__ import annotations

from dataclasses import dataclass

from vime.utils import accelerator


@dataclass(frozen=True)
class RayAcceleratorResource:
    resource_name: str = "GPU"
    native_gpu_resource: bool = True

    def bundle(self, device_count: float = 1, cpu_count: float = 1) -> dict[str, float]:
        return {self.resource_name: device_count, "CPU": cpu_count}

    def actor_options(self, fraction: float) -> dict[str, object]:
        if self.native_gpu_resource:
            return {"num_gpus": fraction}
        # Ray would otherwise schedule against its own GPU resource as well.
        return {"num_gpus": 0, "resources": {self.resource_name: fraction}}

    def accelerator_ids(self) -> list[str]:
        import ray

        if self.native_gpu_resource:
            ids = ray.get_gpu_ids()
        else:
            ids = ray.get_runtime_context().get_accelerator_ids().get(self.resource_name, [])
        return [str(device_id) for device_id in ids]


_REGISTRY: dict[str, RayAcceleratorResource] = {}
_DEFAULT = RayAcceleratorResource()


def register(accelerator_name: str, resource: RayAcceleratorResource) -> None:
    _REGISTRY[accelerator_name.lower()] = resource


def current() -> RayAcceleratorResource:
    return _REGISTRY.get(accelerator.device_type(), _DEFAULT)
