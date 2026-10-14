"""Offload behaviour on Ascend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vime.utils.offload import OffloadPolicy

from ._env import cann_site_packages, prepend_pythonpath


class NpuOffloadPolicy(OffloadPolicy):
    # The Ascend region cannot be suspended once the model is allocated in it.
    supports_disable = False

    def runtime_env(self, args: Any, env_vars: Mapping[str, str] | None = None) -> dict[str, str]:
        env = dict(env_vars or {})
        if not (getattr(args, "offload_train", False) and getattr(args, "train_backend", None) == "megatron"):
            return env

        env["TMS_HOOK_MODE"] = "torch"
        env["TMS_REGION_TAG"] = "training"
        env["TMS_ENABLE_CPU_BACKUP"] = "1"
        if getattr(args, "colocate", False):
            env["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"
        cann_python_path = cann_site_packages()
        if cann_python_path is not None:
            prepend_pythonpath(env, cann_python_path)
        return env

    def training_region(self, offload_train: bool):
        if not offload_train:
            from contextlib import nullcontext

            return nullcontext()
        from torch_memory_saver import torch_memory_saver

        return torch_memory_saver.region(tag="training", enable_cpu_backup=True)

    def initialize_optimizer_state(self, optimizer: Any) -> None:
        if optimizer is None:
            return
        for opt in getattr(optimizer, "chained_optimizers", [optimizer]):
            if opt.optimizer is not None and opt.init_state_fn is not None:
                opt.init_state_fn(opt.optimizer, opt.config)
