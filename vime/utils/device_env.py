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

from vime.utils import accelerator

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

        Empty means the default launch path applies unchanged.
        """
        return {}

    def serving_subprocess_env(
        self, base_env: Mapping[str, str], *, visible_devices: str, colocate: bool
    ) -> dict[str, str]:
        return dict(base_env)


_REGISTRY: dict[str, DeviceEnv] = {}
_DEFAULT = DeviceEnv()


def register(accelerator_name: str, device_env: DeviceEnv) -> None:
    _REGISTRY[accelerator_name.lower()] = device_env


def current() -> DeviceEnv:
    return _REGISTRY.get(accelerator.device_type(), _DEFAULT)
