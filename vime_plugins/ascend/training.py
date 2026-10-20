"""Ascend training installation and offload, without import-time vendor effects."""

from contextlib import nullcontext
from threading import RLock
from typing import Any

from vime.backends.megatron_integration import MegatronIntegration


class AscendMegatronIntegration(MegatronIntegration):
    supports_disable = False

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
        with self._lock:
            if self._args_failed:
                raise RuntimeError("Ascend argument patch installation failed; restart this process")
            self._args_failed = True
            from .patches import after_megatron_args

            after_megatron_args(args)
            self._args_failed = False

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
