"""Where training state is allocated so it can be released again.

An accelerator that can hand memory back mid-run allocates the model inside a
region it can later suspend; the default instead relies on the allocator hook
preloaded into the actor, which can be disabled directly. The environment
either of those needs lives in vime/utils/device_env.py.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from vime.utils import accelerator


class OffloadPolicy:
    """Per-accelerator offload behaviour, defaulting to CUDA's."""

    # Whether an allocated region can be suspended again.
    supports_disable = True

    def training_region(self, offload_train: bool):
        """Context the model and optimizer are allocated inside."""
        return nullcontext()

    def initialize_optimizer_state(self, optimizer: Any) -> None:
        """Materialize lazy optimizer state before leaving the region."""
        return None


_REGISTRY: dict[str, OffloadPolicy] = {}
_DEFAULT = OffloadPolicy()


def register(accelerator_name: str, policy: OffloadPolicy) -> None:
    _REGISTRY[accelerator_name.lower()] = policy


def current() -> OffloadPolicy:
    return _REGISTRY.get(accelerator.device_type(), _DEFAULT)
