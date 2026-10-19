"""Assemble one complete platform integration value for the selected accelerator.

Factories only construct lightweight policies. Runtime and vendor patch installation
remain at their owning backend's explicit lifecycle stages.
"""

from __future__ import annotations

import importlib
import inspect
import os
from dataclasses import dataclass, replace
from threading import RLock

from vime.backends.megatron_integration import MegatronIntegration
from vime.backends.vllm_weight_transfer import WeightTransfer
from vime.utils.device_env import DeviceEnv, _SelectedDeviceEnv


@dataclass(frozen=True)
class PlatformIntegration:
    """Process, training and transfer implementations for one selected platform.

    Device selection and device operations stay in ``accelerator``. Factories
    construct these components; their methods run at the owning workflow stage.
    """

    env: DeviceEnv
    megatron: MegatronIntegration
    weight_transfer: WeightTransfer

    def validate(self) -> None:
        for name, expected in (
            ("env", DeviceEnv),
            ("megatron", MegatronIntegration),
            ("weight_transfer", WeightTransfer),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"PlatformIntegration.{name} must explicitly provide {expected.__name__}")
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
_selected: tuple[tuple[str, str | None], PlatformIntegration] | None = None
_building = False


def current_integration() -> PlatformIntegration:
    from vime.utils import accelerator

    global _selected, _building
    name = accelerator.describe_accelerator().name
    override = os.environ.get("VIME_PLUGINS")
    module_path = _DEFAULT_MODULES.get(name) if override is None else override.strip() or None
    key = (name, module_path)
    with _lock:
        if _selected is not None:
            if _selected[0] != key:
                raise RuntimeError("Platform integration cannot change after assembly; start a new process")
            return _selected[1]
        if _building:
            raise RuntimeError(
                "Recursive platform integration assembly: factories must not query current_integration()"
            )
        _building = True
        try:
            if module_path is not None:
                try:
                    module = importlib.import_module(module_path)
                except ImportError as exc:
                    raise RuntimeError(
                        f"Accelerator {name!r} needs platform integration package {module_path}: {exc}"
                    ) from exc
                factory = getattr(module, "build_integration", None)
                if not callable(factory):
                    raise TypeError(f"Platform integration module {module_path} must define build_integration()")
                integration = factory()
            elif name in _DEFAULT_COMPATIBLE:
                integration = PlatformIntegration(DeviceEnv(), MegatronIntegration(), WeightTransfer())
            else:
                raise RuntimeError(
                    f"Accelerator {name!r} requires complete platform integration; configure VIME_PLUGINS with its module. "
                    "An empty value only permits standalone device APIs, not training or serving."
                )
            if not isinstance(integration, PlatformIntegration):
                raise TypeError("build_integration() must return a complete PlatformIntegration value")
            integration.validate()
            integration = replace(integration, env=_SelectedDeviceEnv(integration.env, name, module_path or ""))
            _selected = (key, integration)
            return integration
        finally:
            _building = False


def reset_integration() -> None:
    """Test isolation only. Does not undo installed third-party patches."""
    global _selected
    with _lock:
        _selected = None
