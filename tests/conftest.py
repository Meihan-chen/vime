"""Make the CPU test suite independent of the runner's accelerator.

``vime/backends/megatron_utils/__init__.py`` runs the vendor bootstrap at
module import time, so on a host with Ascend devices visible, importing
anything under that package pulls in ``megatron_adaptor`` and the real
Megatron modules -- through whatever stubs a test installed, and whether or
not the test has anything to do with NPU. Pinning the platform keeps these
tests asserting the same thing on every runner. Tests that exercise platform
selection set ``VIME_PLATFORM`` themselves and are unaffected.
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    # Set before any test module is imported, since the bootstrap this guards
    # against runs at import time rather than inside a fixture.
    os.environ.setdefault("VIME_PLATFORM", "cuda")
