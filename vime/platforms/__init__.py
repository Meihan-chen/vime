"""Accelerator platform discovery and narrow capability providers.

``VIME_PLATFORM`` is the production override.  ``VIME_TEST_DEVICE`` remains a
compatibility alias for the existing launch/test harness.  When neither is set,
NPU detection is lazy and failure-safe; CUDA is the explicit default.
"""

from __future__ import annotations

import os
from functools import cache

from .base import (
    CheckpointCapabilities,
    Platform,
    RayResourceSpec,
    TrainingBootstrap,
    VLLMLaunchPlatformOps,
    WeightTransferPlatformOps,
)
from .cuda import create_cuda_platform
from .npu import create_npu_platform, detect_npu

_PLATFORM_FACTORIES = {
    "cuda": create_cuda_platform,
    "npu": create_npu_platform,
}


@cache
def get_platform(name: str) -> Platform:
    normalized = name.strip().lower()
    try:
        factory = _PLATFORM_FACTORIES[normalized]
    except KeyError as exc:
        available = ", ".join(_PLATFORM_FACTORIES)
        raise ValueError(f"Unknown Vime platform {name!r}; registered platforms: {available}") from exc
    return factory()


@cache
def _resolve_platform(override: str | None, test_override: str | None) -> Platform:
    selected = override or test_override
    if selected:
        return get_platform(selected)

    try:
        if detect_npu():
            return get_platform("npu")
    except Exception:  # noqa: BLE001 - a failed detector must not break imports
        pass
    return get_platform("cuda")


def current_platform() -> Platform:
    """Resolve the active platform without probing hardware at module import."""
    raw_override = os.environ.get("VIME_PLATFORM")
    override = raw_override.strip().lower() if raw_override and raw_override.strip() else None
    raw_test_override = os.environ.get("VIME_TEST_DEVICE") if override is None else None
    test_override = raw_test_override.strip().lower() if raw_test_override and raw_test_override.strip() else None
    return _resolve_platform(override, test_override)


def reset_platform_cache() -> None:
    """Clear resolver/factory caches (primarily for tests and plugin registration)."""
    get_platform.cache_clear()
    _resolve_platform.cache_clear()


__all__ = [
    "CheckpointCapabilities",
    "Platform",
    "RayResourceSpec",
    "TrainingBootstrap",
    "VLLMLaunchPlatformOps",
    "WeightTransferPlatformOps",
    "current_platform",
    "get_platform",
    "reset_platform_cache",
]
