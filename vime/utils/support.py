"""Assemble one complete support value for the selected accelerator.

Factories only construct lightweight policies. Runtime and vendor patch installation
remain at their owning backend's explicit lifecycle stages.
"""

from __future__ import annotations

import importlib
import inspect
import os
from dataclasses import dataclass
from threading import RLock

from vime.backends.megatron_support import MegatronSupport
from vime.backends.vllm_weight_transfer import WeightTransfer
from vime.utils.device_env import DeviceEnv


@dataclass(frozen=True)
class Support:
    env: DeviceEnv
    megatron: MegatronSupport
    weight_transfer: WeightTransfer

    def validate(self) -> None:
        for name, expected in (("env", DeviceEnv), ("megatron", MegatronSupport), ("weight_transfer", WeightTransfer)):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"Support.{name} must explicitly provide {expected.__name__}")
        # Catch a stale lifecycle signature before partially installing patches.
        inspect.signature(self.megatron.before_import).bind()
        inspect.signature(self.megatron.after_args).bind(object())
        inspect.signature(self.megatron.after_checkpoint_setup).bind(object())
        inspect.signature(self.env.launch_env).bind()
        inspect.signature(self.env.launch_job_env).bind()
        inspect.signature(self.env.train_actor_env).bind(object(), {})
        inspect.signature(self.env.engine_actor_env).bind(object(), {})
        inspect.signature(self.env.serving_subprocess_env).bind({}, visible_devices="0", colocate=False)
        inspect.signature(self.megatron.training_region).bind(False)
        inspect.signature(self.megatron.initialize_optimizer_state).bind(None)
        inspect.signature(self.weight_transfer.backend).bind("distributed")
        inspect.signature(self.weight_transfer.init_info).bind("distributed", rank=0)
        inspect.signature(self.weight_transfer.requires_remote_cleanup).bind(object())
        inspect.signature(self.weight_transfer.ipc_handle_key).bind()


_DEFAULT_MODULES = {"npu": "vime_plugins.ascend"}
# These are the pre-existing main implementations, explicitly using its defaults.
_DEFAULT_COMPATIBLE = frozenset({"cuda", "musa", "supa"})
_lock = RLock()
_selected: tuple[tuple[str, str | None], Support] | None = None
_building = False


def current_support() -> Support:
    from vime.utils import accelerator

    global _selected, _building
    name = accelerator.describe_accelerator().name
    override = os.environ.get("VIME_PLUGINS")
    module_path = _DEFAULT_MODULES.get(name) if override is None else override.strip() or None
    key = (name, module_path)
    with _lock:
        if _selected is not None:
            if _selected[0] != key:
                raise RuntimeError("Accelerator support cannot change after assembly; start a new process")
            return _selected[1]
        if _building:
            raise RuntimeError("Recursive support assembly: factories must not query current_support()")
        _building = True
        try:
            if module_path is not None:
                try:
                    module = importlib.import_module(module_path)
                except ImportError as exc:
                    raise RuntimeError(f"Accelerator {name!r} needs support package {module_path}: {exc}") from exc
                factory = getattr(module, "build_support", None)
                if not callable(factory):
                    raise TypeError(f"Support package {module_path} must define build_support()")
                support = factory()
            elif name in _DEFAULT_COMPATIBLE:
                support = Support(DeviceEnv(), MegatronSupport(), WeightTransfer())
            else:
                raise RuntimeError(
                    f"Accelerator {name!r} requires complete support; configure VIME_PLUGINS with its module. "
                    "An empty value only permits standalone device APIs, not training or serving."
                )
            if not isinstance(support, Support):
                raise TypeError("build_support() must return a complete Support value")
            support.validate()
            _selected = (key, support)
            return support
        finally:
            _building = False


def reset_support() -> None:
    """Test isolation only. Does not undo installed third-party patches."""
    global _selected
    with _lock:
        _selected = None


def current_support_module() -> str:
    """Pin the resolved factory in child processes, including default bindings."""
    current_support()
    return _selected[0][1] or ""
