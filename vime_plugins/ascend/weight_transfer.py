"""Weight transfer and serving launch on Ascend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vime.backends.weight_transfer_registry import WeightTransfer

from ._env import cann_site_packages, prepend_pythonpath


class NpuWeightTransfer(WeightTransfer):
    def backend(self, backend: str) -> str:
        return {"ipc": "npu_ipc", "nccl": "hccl"}.get(backend, backend)

    def trainer_init_info(self, *, colocate: bool, **kwargs) -> Any:
        import torch_npu  # noqa: F401
        from vllm.plugins import load_general_plugins

        # A trainer, unlike a vLLM worker, has not loaded the general plugins.
        load_general_plugins()
        if colocate:
            from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCTrainerInitInfo

            return NPUIPCTrainerInitInfo(**kwargs)
        from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLTrainerInitInfo

        return HCCLTrainerInitInfo(**kwargs)

    def subprocess_env(self, base_env: Mapping[str, str], *, visible_devices: str, colocate: bool) -> dict[str, str]:
        env = dict(base_env)
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("HIP_VISIBLE_DEVICES", None)
        env["ASCEND_RT_VISIBLE_DEVICES"] = visible_devices
        env["VLLM_USE_AOT_COMPILE"] = "0"
        cann_python_path = cann_site_packages()
        if cann_python_path is not None:
            prepend_pythonpath(env, cann_python_path)
        if colocate:
            env["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"
        return env

    def engine_runtime_env(self, args: Any, env_vars: Mapping[str, str] | None = None) -> dict[str, str]:
        env = dict(env_vars or {})
        cann_python_path = cann_site_packages()
        if cann_python_path is not None:
            prepend_pythonpath(env, cann_python_path)
        if getattr(args, "colocate", False):
            env["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"
        env["VLLM_USE_AOT_COMPILE"] = "0"
        return env
