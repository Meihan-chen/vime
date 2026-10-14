"""The environment an Ascend child process needs."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from vime.utils.device_env import DeviceEnv, offload_requested


def _cann_site_packages() -> str | None:
    """Locate the CANN python bindings, which a child process imports as acl."""
    candidates: list[str] = []
    for env_key in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH"):
        base = os.environ.get(env_key)
        if not base:
            continue
        candidates.extend(
            [
                os.path.join(base, "python", "site-packages"),
                os.path.normpath(os.path.join(base, "..", "python", "site-packages")),
            ]
        )
    candidates.append("/usr/local/Ascend/ascend-toolkit/latest/python/site-packages")
    for path in candidates:
        if os.path.isdir(os.path.join(path, "acl")):
            return path
    return None


def _with_cann_and_allocator(env: dict[str, str], colocate: bool) -> dict[str, str]:
    cann_python_path = _cann_site_packages()
    if cann_python_path is not None:
        existing = env.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
        parts = {part for part in existing.split(os.pathsep) if part}
        if cann_python_path not in parts:
            env["PYTHONPATH"] = os.pathsep.join([cann_python_path, existing] if existing else [cann_python_path])
    if colocate:
        env["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"
    return env


class NpuDeviceEnv(DeviceEnv):
    def train_actor_env(self, args: Any, env_vars: Mapping[str, str] | None = None) -> dict[str, str]:
        env = dict(env_vars or {})
        if not offload_requested(args):
            return env
        env["TMS_HOOK_MODE"] = "torch"
        env["TMS_REGION_TAG"] = "training"
        env["TMS_ENABLE_CPU_BACKUP"] = "1"
        return _with_cann_and_allocator(env, getattr(args, "colocate", False))

    def engine_actor_env(self, args: Any, env_vars: Mapping[str, str] | None = None) -> dict[str, str]:
        env = dict(env_vars or {})
        env["VLLM_USE_AOT_COMPILE"] = "0"
        return _with_cann_and_allocator(env, getattr(args, "colocate", False))

    def serving_subprocess_env(
        self, base_env: Mapping[str, str], *, visible_devices: str, colocate: bool
    ) -> dict[str, str]:
        env = dict(base_env)
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("HIP_VISIBLE_DEVICES", None)
        env["ASCEND_RT_VISIBLE_DEVICES"] = visible_devices
        env["VLLM_USE_AOT_COMPILE"] = "0"
        return _with_cann_and_allocator(env, colocate)
