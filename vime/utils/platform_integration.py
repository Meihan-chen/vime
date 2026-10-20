"""Load optional backend components for the accelerator already selected.

Only the requested component is checked. Environment queries do not validate
training/transfer contracts or install vendor patches.
"""

from __future__ import annotations

import importlib
import os
from typing import TypeVar

from vime.utils import accelerator

_DEFAULT_MODULES = {"npu": "vime_plugins.ascend"}
_DEFAULT_COMPATIBLE = frozenset({"cuda", "musa", "supa"})
_Component = TypeVar("_Component")


def plugin_module() -> str | None:
    name = accelerator.describe_accelerator().name
    configured = os.environ.get("VIME_PLUGINS")
    module = _DEFAULT_MODULES.get(name) if configured is None else configured.strip() or None
    if module is None and name not in _DEFAULT_COMPATIBLE:
        raise RuntimeError(f"Accelerator {name!r} requires a backend module; configure VIME_PLUGINS")
    return module


def component(name: str, default: _Component) -> _Component:
    module_path = plugin_module()
    if module_path is None:
        return default
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise RuntimeError(f"Cannot import backend module {module_path}: {exc}") from exc
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise TypeError(f"Backend module {module_path} must provide {name!r}") from exc
