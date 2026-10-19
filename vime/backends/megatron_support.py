"""Lightweight training contracts; importing this module does not import Megatron."""

from contextlib import nullcontext
from typing import Any


class MegatronSupport:
    """Default training behavior, with explicit installation and allocation stages."""

    supports_disable = True
    supports_buffer_cpu_backup_disable = True

    def before_import(self) -> None:
        pass

    def after_args(self, args: Any) -> None:
        pass

    def after_checkpoint_setup(self, default_planner: Any) -> None:
        pass

    def training_region(self, offload_train: bool):
        return nullcontext()

    def initialize_optimizer_state(self, optimizer: Any) -> None:
        pass


def current() -> MegatronSupport:
    from vime.utils.support import current_support

    return current_support().megatron
