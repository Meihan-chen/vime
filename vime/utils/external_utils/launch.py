"""Launch policy for the vime test and example harness.

Used only by the launch utilities in ``command_utils``, not by vime core. A
``Platform`` describes one accelerator for the purpose of starting a job: how
Ray should be told to advertise its devices, and the environment the job needs
exported before ``ray start`` so the raylet and the actors it spawns inherit
it. Both are read off the seams that already answer those questions for the
running process, so adding an accelerator needs no entry here.

Imports stay stdlib-only at module level so this is unit-testable in
isolation; the ``exec_command`` calls live in ``command_utils``.
"""

import json
import shlex
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Platform:
    name: str
    ray_args: str  # ray-start resource flags, "{n}"-templated with the device count
    env: dict = field(default_factory=dict)  # device env (into runtime_env + raylet)

    def ray_start_args(self, num_devices: int) -> str:
        return self.ray_args.format(n=num_devices)


def launch_platform() -> Platform | None:
    """Describe the selected accelerator, or ``None`` if it needs no override.

    A ``None`` result means the default launch path applies unchanged, which is
    also what an accelerator with no launch environment of its own gets.
    """
    from vime.utils import accelerator
    from vime.utils.device_env import current as current_device_env

    env = current_device_env().launch_env()
    if not env:
        return None

    if accelerator.ray_uses_native_gpu_resource():
        ray_args = "--num-gpus {n}"
    else:
        # Ray resolves a worker's visible-devices variable from the resource
        # name, so the job has to advertise the same name the process asks for.
        # Braces are doubled because ray_args is a format template.
        ray_args = "--num-gpus 0 --resources '{{" + json.dumps(accelerator.ray_resource_name()) + ": {n}}}'"

    return Platform(name=accelerator.device_type(), ray_args=ray_args, env=env)


def launch_commands(
    platform: Platform,
    train_args: str,
    num_devices: int,
    megatron_model_type: str | None,
    repo_base_dir: str,
    train_script: str = "train.py",
    extra_env: dict | None = None,
    external_ray: bool = False,
    master_addr: str = "127.0.0.1",
) -> list:
    """Ordered shell commands to launch a training job on `platform` (pure; caller execs)."""
    extra_env = extra_env or {}
    all_env = {**platform.env, **extra_env}
    all_env["VIME_ACCELERATOR"] = platform.name
    cmds: list = []
    cmds.append(
        "pkill -9 -f '[v]llm serve|VLL[M]::'; sleep 3; "
        + ("" if external_ray else "ray stop --force; pkill -9 ray; ")
        + "pkill -9 vime; sleep 3; true; "
    )
    if not external_ray:
        # Export the device env BEFORE `ray start` so the raylet and the actors it spawns
        # (e.g. vLLM engines) inherit it; a job's runtime_env does not reach those actors.
        exports = "".join(f"export {k}={shlex.quote(str(v))} && " for k, v in all_env.items())
        cmds.append(
            f"{exports}export PYTHONUNBUFFERED=1 && ray start --head --node-ip-address {master_addr} "
            f"{platform.ray_start_args(num_devices)} --disable-usage-stats"
        )

    runtime_env = {"env_vars": {"no_proxy": f"127.0.0.1,{master_addr}", "MASTER_ADDR": master_addr, **all_env}}
    src = f'source "{repo_base_dir}/scripts/models/{megatron_model_type}.sh" && ' if megatron_model_type else ""
    model_args = "${MODEL_ARGS[@]}" if megatron_model_type else ""
    cmds.append(
        f"export no_proxy=127.0.0.1 && export PYTHONUNBUFFERED=1 && {src}"
        f'ray job submit --address="http://127.0.0.1:8265" '
        f"--runtime-env-json={shlex.quote(json.dumps(runtime_env))} "
        f"-- python3 {train_script} {model_args} {train_args}"
    )
    return cmds
