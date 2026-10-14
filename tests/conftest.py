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
    # Both of these are set before any test module is imported, since what they
    # guard against runs at import time rather than inside a fixture.
    os.environ.setdefault("VIME_PLATFORM", "cuda")

    # Several modules install Megatron stubs and then import something under
    # vime.backends.megatron_utils, whose package body asks vendors to install
    # their patches. A vendor doing that needs the real Megatron and fails
    # against the stubs. Marking the bootstrap as already done keeps the vendor
    # support package out of this session; the tests that do exercise it select
    # the accelerator themselves.
    from vime.utils import accelerator

    accelerator._NPU_BOOTSTRAP_CHECKED = True
