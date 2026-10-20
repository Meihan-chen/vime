"""The environment an Ascend child process needs."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from vime.utils.device_env import DeviceEnv, offload_requested


_ASCEND_HOME_DEFAULT = "/usr/local/Ascend"
_REPO_ROOT_DEFAULT = "/root"
_ALLOC_CONF = "expandable_segments:False"


def _ascend_home() -> str:
    return os.environ.get("ASCEND_HOME", _ASCEND_HOME_DEFAULT)


def _toolkit_home() -> str:
    return os.environ.get("ASCEND_TOOLKIT_HOME", f"{_ascend_home()}/ascend-toolkit/latest")


def _launch_pythonpath() -> str:
    """Megatron forks plus the CANN python bindings."""
    if (configured := os.environ.get("VIME_NPU_PYTHONPATH")) is not None:
        return configured
    repos = os.environ.get("VIME_REPO_ROOT", _REPO_ROOT_DEFAULT)
    toolkit = _toolkit_home()
    return ":".join(
        [
            f"{repos}/Megatron-LM",
            f"{repos}/vime",
            f"{repos}/MegatronAdaptor",
            f"{repos}/TransformerEngineNPU",
            f"{toolkit}/python/site-packages",
            f"{toolkit}/tools/ms_fmk_transplt/torch_npu_bridge",
        ]
    )


def _launch_ld_library_path() -> str:
    """Driver, toolkit, ATB and CANN runtime libraries."""
    if (configured := os.environ.get("VIME_NPU_LD_LIBRARY_PATH")) is not None:
        return configured
    ascend = _ascend_home()
    toolkit = _toolkit_home()
    return ":".join(
        [
            f"{ascend}/driver/lib64",
            f"{ascend}/driver/lib64/driver",
            f"{ascend}/driver/lib64/common",
            f"{toolkit}/lib64",
            f"{toolkit}/compiler/lib64/plugin/opskernel",
            f"{toolkit}/compiler/lib64/plugin/nnengine",
            f"{toolkit}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/",
            f"{ascend}/nnal/atb/latest/atb/cxx_abi_1/lib",
            f"{ascend}/cann/lib64",
            f"{ascend}/cann/aarch64-linux/devlib",
        ]
    )


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
        env["PYTORCH_NPU_ALLOC_CONF"] = _ALLOC_CONF
    return env


class NpuDeviceEnv(DeviceEnv):
    def train_actor_env(self, args: Any, env_vars: Mapping[str, str] | None = None) -> dict[str, str]:
        env = dict(env_vars or {})
        if not offload_requested(args) or getattr(args, "train_backend", None) != "megatron":
            return env
        env["TMS_HOOK_MODE"] = "torch"
        return _with_cann_and_allocator(env, getattr(args, "colocate", False))

    def engine_actor_env(self, args: Any, env_vars: Mapping[str, str] | None = None) -> dict[str, str]:
        env = dict(env_vars or {})
        env["VLLM_USE_AOT_COMPILE"] = "0"
        return _with_cann_and_allocator(env, getattr(args, "colocate", False))

    def launch_env(self) -> dict[str, str]:
        return {
            "PYTHONPATH": _launch_pythonpath(),
            "LD_LIBRARY_PATH": _launch_ld_library_path(),
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
            "HCCL_HOST_SOCKET_PORT_RANGE": "60000-60050",
            "HCCL_NPU_SOCKET_PORT_RANGE": "61000-61050",
            "HCCL_CONNECT_TIMEOUT": "7200",
            "PYTORCH_NPU_ALLOC_CONF": _ALLOC_CONF,
            "VLLM_DISABLE_COMPILE_CACHE": "1",
            "HYDRA_FULL_ERROR": "1",
            "TRANSFORMERS_VERBOSITY": "error",
        }

    def launch_job_env(self) -> dict[str, str]:
        return {"RAY_USE_UVLOOP": "0", **self.launch_env()}

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
