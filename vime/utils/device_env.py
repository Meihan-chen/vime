"""Device-specific environment values for jobs, actors and serving subprocesses."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


_TMS_PRELOAD_LIBRARIES = (
    "torch_memory_saver_hook_mode_preload_cu13.abi3.so",
    "torch_memory_saver_hook_mode_preload_cu12.abi3.so",
    "torch_memory_saver_hook_mode_preload.abi3.so",
)


class DeviceEnv:
    """Device environment values; callers own launch and offload decisions."""

    def train_offload_env(self, args: Any, env_vars: Mapping[str, str]) -> dict[str, str]:
        """CUDA's existing TMS preload setup, called only with offload enabled."""
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
        env = dict(env_vars)
        env["LD_PRELOAD"] = dynlib_path
        env["TMS_INIT_ENABLE"] = "1"
        env["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"
        return env

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


def for_child(env_vars: Mapping[str, str]) -> dict[str, str]:
    """Propagate explicit selection at process boundaries; keep GPU defaults.

    Check supplied selectors even without propagation, so a child cannot
    silently select a different backend.
    """
    from vime.utils import accelerator
    from vime.utils.platform_integration import plugin_module

    name = accelerator.describe_accelerator().name
    module = plugin_module()
    required = {"VIME_ACCELERATOR": name, "VIME_PLUGINS": module or ""}
    env = dict(env_vars)
    for key, value in required.items():
        if key in env and env[key] != value:
            raise ValueError(f"Child {key} must match the selected accelerator")
    explicit = os.environ.get("VIME_ACCELERATOR", "").strip().lower() not in {"", "auto"}
    if explicit or module is not None or "VIME_PLUGINS" in os.environ:
        env["VIME_ACCELERATOR"] = name
        if module is not None or "VIME_PLUGINS" in os.environ:
            env["VIME_PLUGINS"] = module or ""
    return env


_default = DeviceEnv()


def current() -> DeviceEnv:
    from vime.utils.platform_integration import component

    return component("env", _default)
