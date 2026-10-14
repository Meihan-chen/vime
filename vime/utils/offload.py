"""Accelerator memory-offload policy.

Offloading training state needs three things that differ by accelerator: the
environment the actor is launched with, a region to allocate the model inside,
and whether that region can be suspended once allocated. CUDA gets all three
from torch_memory_saver's preload hook, which is the default here.

The policy itself imports no accelerator, so it can be constructed and
asserted without a device.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

from vime.utils import accelerator


def cuda_offload_env_vars(env_vars: dict, library_dir: str | None = None) -> dict:
    """Return ``env_vars`` plus the torch_memory_saver preload contract.

    All three variables are required for TMS to take over the allocator in the
    actor process, so a missing library raises rather than letting training
    start without it.
    """
    if library_dir is None:
        import torch_memory_saver

        library_dir = os.path.dirname(os.path.dirname(torch_memory_saver.__file__))

    for path in [
        "torch_memory_saver_hook_mode_preload_cu13.abi3.so",
        "torch_memory_saver_hook_mode_preload_cu12.abi3.so",
        "torch_memory_saver_hook_mode_preload.abi3.so",
    ]:
        dynlib_path = os.path.join(library_dir, path)
        if os.path.exists(dynlib_path):
            break
    else:
        raise FileNotFoundError(
            "Cannot find torch_memory_saver dynamic library. Please make sure torch_memory_saver is properly installed."
        )

    env_vars = dict(env_vars)
    env_vars["LD_PRELOAD"] = dynlib_path
    env_vars["TMS_INIT_ENABLE"] = "1"
    env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"
    return env_vars


class OffloadPolicy:
    """Per-accelerator offload behaviour, defaulting to CUDA's."""

    # Whether an allocated region can be suspended again.
    supports_disable = True

    def runtime_env(self, args: Any, env_vars: Mapping[str, str] | None = None) -> dict[str, str]:
        env = dict(env_vars or {})
        if not (getattr(args, "offload_train", False) and getattr(args, "train_backend", None) == "megatron"):
            return env
        return cuda_offload_env_vars(env)

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
