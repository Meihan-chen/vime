"""Launch inputs only. command_utils owns the single Ray/job lifecycle."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass


@dataclass(frozen=True)
class LaunchInputs:
    name: str
    ray_args: str
    start_env: dict[str, str]


def shell_exports(env: dict[str, str]) -> str:
    for key in env:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
            raise ValueError(f"Invalid environment variable name: {key!r}")
    return "".join(f"export {key}={shlex.quote(str(value))} && " for key, value in env.items())


def launch_inputs(num_devices: int) -> LaunchInputs:
    from vime.utils import accelerator
    from vime.utils.device_env import current, for_child

    device = accelerator.describe_accelerator()
    env = current()
    if device.ray_uses_native_gpu_resource:
        flags = f"--num-gpus {num_devices}"
    else:
        flags = "--num-gpus 0 --resources " + shlex.quote(json.dumps({device.ray_resource_name: num_devices}))
    return LaunchInputs(device.name, flags, for_child(env.launch_env()))
