"""Patches that have to be installed at a specific point during startup.

Each function here is registered against one phase of the shared vendor-hook
seam; the phase name says when it runs, not what it patches.
"""

from __future__ import annotations

import importlib
import logging
import sys
from typing import Any

from vime.utils import accelerator

logger = logging.getLogger(__name__)

def _install_safe_empty_cache() -> None:
    """Keep the allocator guard MindSpeed and TMS callers rely on."""
    torch = importlib.import_module("torch")
    original_empty_cache = torch.npu.empty_cache
    if not getattr(original_empty_cache, "_vime_safe_empty_cache", False):

        def _safe_empty_cache(_original=original_empty_cache) -> None:
            try:
                _original()
            except RuntimeError:
                pass

        _safe_empty_cache._vime_safe_empty_cache = True
        torch.npu.empty_cache = _safe_empty_cache

    # Some shared dependencies still call the CUDA spelling after torch_npu's
    # compatibility patching. Scope that alias to NPU-bootstrapped jobs.
    torch.cuda.empty_cache = torch.npu.empty_cache


def before_megatron_import() -> None:
    # MegatronAdaptor refuses to install its patches twice, and this module can
    # be re-imported within one process (a test that resets sys.modules, for
    # instance), so the guard has to live where the patches do rather than in a
    # module-level flag here.
    if "megatron_adaptor" in sys.modules:
        return

    importlib.import_module("torch_npu")
    selected = accelerator.get_accelerator()
    if selected.name != "npu":
        raise RuntimeError(f"NPU bootstrap cannot use an already selected {selected.name!r} accelerator")
    _install_safe_empty_cache()
    # MegatronAdaptor installs its pre-patches here; the attention override goes
    # on top of them.
    importlib.import_module("megatron_adaptor")
    importlib.import_module("vime_plugins.ascend.attention_patch")


def after_megatron_args(args: Any) -> None:
    """Reapply the vendor patches against the fully resolved arguments."""
    features_manager = importlib.import_module("megatron_adaptor.features_manager.features_manager").FeaturesManager
    full_args = importlib.import_module("megatron_adaptor.utils.args_utils").get_full_args()
    for key, value in vars(args).items():
        setattr(full_args, key, value)
    features_manager.remove_patches()
    features_manager.apply_features_pre_patches(full_args)
    features_manager.apply_features_patches(full_args)
    # Reapplying can replace attention again, and importing the cached module
    # does not reinstall the override on its own.
    attention = importlib.import_module("vime_plugins.ascend.attention_patch")
    attention.DotProductAttention.forward = attention.npu_dot_product_attention_forward

def relax_global_plan_validation(default_planner: Any) -> None:
    """Skip the access-integrity check, which does not hold for Ascend shards."""
    if not hasattr(default_planner, "_validate_global_plan"):
        return

    def _validate_global_plan(global_plan, metadata):
        logger.info("Skipping validate_access_integrity")
        return True

    default_planner._validate_global_plan = _validate_global_plan
