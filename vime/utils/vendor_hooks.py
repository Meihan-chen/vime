"""Callbacks that have to run at a specific point during import or startup.

Some vendor runtimes patch a library and must do so before that library is
imported anywhere, which no amount of lazy initialization can arrange after
the fact. Shared code names the point it has reached and passes whatever is
available there; whoever needs to act at that point registers for it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_HOOKS: dict[str, list[Callable[..., None]]] = {}


def register(phase: str, hook: Callable[..., None]) -> None:
    _HOOKS.setdefault(phase, []).append(hook)


def run(phase: str, *args: Any, **kwargs: Any) -> None:
    for hook in _HOOKS.get(phase, ()):
        hook(*args, **kwargs)
