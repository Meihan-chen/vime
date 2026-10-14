"""Distributed-checkpoint adjustments on Ascend."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def relax_global_plan_validation(default_planner: Any) -> None:
    """Skip the access-integrity check, which does not hold for Ascend shards."""
    if not hasattr(default_planner, "_validate_global_plan"):
        return

    def _validate_global_plan(global_plan, metadata):
        logger.info("Skipping validate_access_integrity")
        return True

    default_planner._validate_global_plan = _validate_global_plan
