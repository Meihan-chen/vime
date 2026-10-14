"""Which weight-transfer engine and launch environment an accelerator uses.

The transfer lifecycle is shared; what differs per accelerator is the engine
name behind "ipc"/"nccl", the init-info type that engine expects, and the
environment its serving subprocess needs. Anything not registered keeps the
CUDA behaviour.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vime.utils import accelerator


class WeightTransfer:
    """Per-accelerator weight-transfer and serving-launch behaviour."""

    def backend(self, backend: str) -> str:
        """Map a transport name onto the engine that implements it."""
        return backend

    def trainer_init_info(self, *, colocate: bool, **kwargs) -> Any:
        if colocate:
            from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo

            return IPCTrainerInitInfo(**kwargs)
        from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo

        return NCCLTrainerInitInfo(**kwargs)

    def subprocess_env(self, base_env: Mapping[str, str], *, visible_devices: str, colocate: bool) -> dict[str, str]:
        """Adjust the environment a serving subprocess is launched with."""
        return dict(base_env)

    def engine_runtime_env(self, args: Any, env_vars: Mapping[str, str] | None = None) -> dict[str, str]:
        """Adjust the Ray runtime env of a serving actor."""
        return dict(env_vars or {})


_REGISTRY: dict[str, WeightTransfer] = {}
_DEFAULT = WeightTransfer()


def register(accelerator_name: str, transfer: WeightTransfer) -> None:
    _REGISTRY[accelerator_name.lower()] = transfer


def current() -> WeightTransfer:
    return _REGISTRY.get(accelerator.device_type(), _DEFAULT)
