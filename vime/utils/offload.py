"""Accelerator memory-offload policy.

Imports neither Ray nor an accelerator, so the offload environment can be
constructed and asserted without a device or a live cluster.
"""

from __future__ import annotations

import os


def cuda_offload_env_vars(env_vars: dict, library_dir: str | None = None) -> dict:
    """Return ``env_vars`` plus the torch_memory_saver preload contract.

    All three variables are required for TMS to take over the allocator in the
    actor process, so a missing library raises rather than letting training
    start without it.
    """
    if library_dir is None:
        import torch_memory_saver

        library_dir = os.path.dirname(os.path.dirname(torch_memory_saver.__file__))

    for path in [
        "torch_memory_saver_hook_mode_preload_cu13.abi3.so",
        "torch_memory_saver_hook_mode_preload_cu12.abi3.so",
        "torch_memory_saver_hook_mode_preload.abi3.so",
    ]:
        dynlib_path = os.path.join(library_dir, path)
        if os.path.exists(dynlib_path):
            break
    else:
        raise FileNotFoundError(
            "Cannot find torch_memory_saver dynamic library. Please make sure torch_memory_saver is properly installed."
        )

    env_vars = dict(env_vars)
    env_vars["LD_PRELOAD"] = dynlib_path
    env_vars["TMS_INIT_ENABLE"] = "1"
    env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"
    return env_vars
