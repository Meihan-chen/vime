"""Patches that have to be installed at a specific point during startup.

AscendMegatronIntegration calls these functions at fixed Megatron lifecycle stages.
"""

from __future__ import annotations

import importlib
from typing import Any

from vime.utils import accelerator


def _install_safe_empty_cache() -> None:
    """Route vendor CUDA cache calls to NPU without hiding allocator failures."""
    torch = importlib.import_module("torch")
    # The alias is required by vendor callers. Preserve all allocator/device
    # errors: there is no demonstrated safe exception that can be swallowed.
    torch.cuda.empty_cache = torch.npu.empty_cache


def before_megatron_import() -> None:
    importlib.import_module("torch_npu")
    selected = accelerator.describe_accelerator()
    if selected.name != "npu":
        raise RuntimeError(f"NPU bootstrap cannot use an already selected {selected.name!r} accelerator")
    _install_safe_empty_cache()
    # MegatronAdaptor installs its pre-patches here; the attention override goes
    # on top of them.
    importlib.import_module("megatron_adaptor")
    attention = importlib.import_module("vime_plugins.ascend.attention_patch")
    attention.DotProductAttention.forward = attention.npu_dot_product_attention_forward


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
