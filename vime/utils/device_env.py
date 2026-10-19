"""The environment a child process needs for the accelerator in use.

Three kinds of process need it: a training actor, a serving engine actor, and
the serving subprocess that actor launches. They are launched by different
subsystems and the first two receive it through Ray's runtime env while the
third receives it through the process environment, but what has to go in is
one accelerator-specific question, so it is answered in one place.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


_TMS_PRELOAD_LIBRARIES = (
    "torch_memory_saver_hook_mode_preload_cu13.abi3.so",
    "torch_memory_saver_hook_mode_preload_cu12.abi3.so",
    "torch_memory_saver_hook_mode_preload.abi3.so",
)


def cuda_offload_env_vars(env_vars: Mapping[str, str], library_dir: str | None = None) -> dict[str, str]:
    """Return ``env_vars`` plus the torch_memory_saver preload contract.

    All three variables are required for TMS to take over the allocator in the
    actor process, so a missing library raises rather than letting training
    start without it.
    """
    if library_dir is None:
        import torch_memory_saver

        library_dir = os.path.dirname(os.path.dirname(torch_memory_saver.__file__))

    for name in _TMS_PRELOAD_LIBRARIES:
        dynlib_path = os.path.join(library_dir, name)
        if os.path.exists(dynlib_path):
            break
    else:
        raise FileNotFoundError(
            "Cannot find torch_memory_saver dynamic library. Please make sure torch_memory_saver is properly installed."
        )

    env_vars = dict(env_vars)
    env_vars["LD_PRELOAD"] = dynlib_path
    env_vars["TMS_INIT_ENABLE"] = "1"
    env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"
    return env_vars


def offload_requested(args: Any) -> bool:
    return bool(getattr(args, "offload_train", False))


class DeviceEnv:
    """Per-accelerator child-process environment, defaulting to CUDA's."""

    def train_actor_env(self, args: Any, env_vars: Mapping[str, str] | None = None) -> dict[str, str]:
        env = dict(env_vars or {})
        if not offload_requested(args):
            return env
        return cuda_offload_env_vars(env)

    def engine_actor_env(self, args: Any, env_vars: Mapping[str, str] | None = None) -> dict[str, str]:
        return dict(env_vars or {})

    def launch_env(self) -> dict[str, str]:
        """Environment a launched job needs exported before ``ray start``.

        Resource selection is independent of whether this mapping is empty.
        """
        return {}

    def launch_job_env(self) -> dict[str, str]:
        from vime.utils.external_utils.command_utils import check_has_nvlink

        nvls = os.environ.get("NCCL_NVLS_ENABLE")
        return {
            "PYTHONPATH": "/root/Megatron-LM/",
            "RAY_USE_UVLOOP": "0",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": nvls if nvls is not None else str(int(check_has_nvlink())),
        }

    def serving_subprocess_env(
        self, base_env: Mapping[str, str], *, visible_devices: str, colocate: bool
    ) -> dict[str, str]:
        return dict(base_env)


class _SelectedDeviceEnv(DeviceEnv):
    def __init__(self, policy: DeviceEnv, name: str, module: str):
        self.policy, self.name, self.module = policy, name, module

    def _pin(self, env: Mapping[str, str]) -> dict[str, str]:
        result = dict(env)
        required = {"VIME_ACCELERATOR": self.name, "VIME_PLUGINS": self.module}
        for key, value in required.items():
            if key in result and result[key] != value:
                raise ValueError(f"Child {key} must match the selected accelerator support")
            result[key] = value
        return result

    def train_actor_env(self, args, env_vars=None):
        return self._pin(self.policy.train_actor_env(args, env_vars))

    def engine_actor_env(self, args, env_vars=None):
        return self._pin(self.policy.engine_actor_env(args, env_vars))

    def launch_env(self):
        return self._pin(self.policy.launch_env())

    def launch_job_env(self):
        return self._pin(self.policy.launch_job_env())

    def serving_subprocess_env(self, base_env, *, visible_devices, colocate):
        return self._pin(
            self.policy.serving_subprocess_env(base_env, visible_devices=visible_devices, colocate=colocate)
        )


def current() -> DeviceEnv:
    from vime.utils import accelerator
    from vime.utils.support import current_support, current_support_module

    support = current_support()
    return _SelectedDeviceEnv(support.env, accelerator.describe_accelerator().name, current_support_module())
