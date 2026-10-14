"""Where Ascend allocates training state."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from vime.utils.offload import OffloadPolicy


class NpuOffloadPolicy(OffloadPolicy):
    # The Ascend region cannot be suspended once the model is allocated in it.
    supports_disable = False

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
