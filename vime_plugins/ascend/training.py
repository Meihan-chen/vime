"""Ascend training installation and offload, without import-time vendor effects."""

from contextlib import nullcontext
from threading import RLock
from typing import Any

from vime.backends.megatron_support import MegatronSupport


class NpuMegatronSupport(MegatronSupport):
    supports_disable = False
    supports_buffer_cpu_backup_disable = False

    def __init__(self):
        self._import_state = "new"
        self._lock = RLock()
        self._args_failed = False

    def before_import(self) -> None:
        with self._lock:
            if self._import_state == "done":
                return
            if self._import_state != "new":
                raise RuntimeError("Ascend patch installation did not complete; restart this process")
            self._import_state = "installing"
            from .patches import before_megatron_import

            before_megatron_import()
            self._import_state = "done"

    def after_args(self, args: Any) -> None:
        # The pinned NPU Megatron has no _ParamAndGradBuffer._allocate_buffer.
        # main's new per-buffer policy would otherwise silently do nothing.
        for flag in ("disable_param_buffers_cpu_backup", "disable_grad_buffers_cpu_backup"):
            if getattr(args, flag, False):
                raise ValueError(f"{flag}=True is unsupported by the pinned Ascend Megatron stack")
        with self._lock:
            if self._args_failed:
                raise RuntimeError("Ascend argument patch installation failed; restart this process")
            self._args_failed = True
            from .patches import after_megatron_args

            after_megatron_args(args)
            self._args_failed = False

    def after_checkpoint_setup(self, default_planner: Any) -> None:
        from .patches import relax_global_plan_validation

        relax_global_plan_validation(default_planner)

    def training_region(self, offload_train: bool):
        if not offload_train:
            return nullcontext()
        from torch_memory_saver import torch_memory_saver

        return torch_memory_saver.region(tag="training", enable_cpu_backup=True)

    def initialize_optimizer_state(self, optimizer: Any) -> None:
        if optimizer is None:
            return
        for opt in getattr(optimizer, "chained_optimizers", [optimizer]):
            if opt.optimizer is not None and opt.init_state_fn is not None:
                opt.init_state_fn(opt.optimizer, opt.config)
