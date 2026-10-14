"""Keep the CPU suite independent of the runner's accelerator.

Several modules install Megatron stubs and then import something under
``vime.backends.megatron_utils``, whose package body asks vendors to install
their patches. A vendor doing that needs the real Megatron and fails against
the stubs, so the support package stays out of this session. The tests that do
exercise it register what they need themselves.
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    # Set before any test module is imported, since the load this suppresses
    # happens at import time rather than inside a fixture.
    from vime.utils import accelerator

    accelerator._NPU_BOOTSTRAP_CHECKED = True
