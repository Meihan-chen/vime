"""How Ray advertises and addresses the accelerator in use.

Ray models an accelerator as a resource name plus a visible-devices variable,
and resolves the second from the first: only its own NPU resource maps to
ASCEND_RT_VISIBLE_DEVICES, while anything advertised as GPU maps to
CUDA_VISIBLE_DEVICES. Asking for the wrong name therefore leaves the process
with no visible device, so these read the name off the accelerator rather than
assuming Ray's GPU resource.
"""

from __future__ import annotations

from vime.utils import accelerator


def resource_name() -> str:
    return accelerator.ray_resource_name()


def bundle(device_count: float = 1, cpu_count: float = 1) -> dict[str, float]:
    return {accelerator.ray_resource_name(): device_count, "CPU": cpu_count}


def actor_options(fraction: float) -> dict[str, object]:
    if accelerator.ray_uses_native_gpu_resource():
        return {"num_gpus": fraction}
    # Ray would otherwise schedule against its own GPU resource as well.
    return {"num_gpus": 0, "resources": {accelerator.ray_resource_name(): fraction}}


def accelerator_ids() -> list[str]:
    import ray

    if accelerator.ray_uses_native_gpu_resource():
        ids = ray.get_gpu_ids()
    else:
        ids = ray.get_runtime_context().get_accelerator_ids().get(accelerator.ray_resource_name(), [])
    return [str(device_id) for device_id in ids]
