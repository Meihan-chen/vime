"""Lightweight training contracts; importing this module does not import Megatron."""

from contextlib import nullcontext
from typing import Any


class MegatronIntegration:
    """Default training behavior, with explicit installation and allocation stages."""

    supports_disable = True

    def before_import(self) -> None:
        pass

    def after_args(self, args: Any) -> None:
        pass

    def training_region(self, offload_train: bool):
        return nullcontext()

    def initialize_optimizer_state(self, optimizer: Any) -> None:
        pass


_default = MegatronIntegration()


def current() -> MegatronIntegration:
    from vime.utils.platform_integration import component

    return component("megatron", _default)
