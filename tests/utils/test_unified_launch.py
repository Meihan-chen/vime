"""Run the real launcher with command boundaries replaced, preserving M10 lifecycle."""

import json
import shlex
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vime.utils import accelerator, platform_integration
from vime.utils.external_utils import command_utils as commands
from vime.utils.external_utils.launch import launch_inputs, shell_exports


@pytest.fixture
def launch_case(monkeypatch):
    monkeypatch.setattr(accelerator, "_ACCELERATOR", None)
    monkeypatch.setattr(platform_integration, "_selected", None)
    monkeypatch.delenv("VIME_PLUGINS", raising=False)
    monkeypatch.setenv("VIME_ACCELERATOR", "cuda")
    monkeypatch.setenv("VIME_SCRIPT_EXTERNAL_RAY", "0")
    monkeypatch.setenv("VIME_SCRIPT_ENABLE_RAY_SUBMIT", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setattr(torch.version, "hip", None)
    calls = []
    monkeypatch.setattr(commands, "exec_command", lambda cmd, **kwargs: calls.append(cmd))
    monkeypatch.setattr(commands, "check_has_nvlink", lambda: True)
    monkeypatch.setattr(commands.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    return calls


def submitted_env(calls):
    tokens = shlex.split(next(c for c in calls if "ray job submit" in c))
    return json.loads(next(t.split("=", 1)[1] for t in tokens if t.startswith("--runtime-env-json=")))["env_vars"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "backend,resource", [("cuda", "--num-gpus 4"), ("npu", "--num-gpus 0 --resources '{\"NPU\": 4}'")]
)
def test_shared_start_callback_and_submit(launch_case, monkeypatch, backend, resource):
    monkeypatch.setenv("VIME_ACCELERATOR", backend)
    if backend == "npu":
        monkeypatch.setattr(commands, "check_has_nvlink", lambda: pytest.fail("NPU must not probe NVIDIA"))
    commands.execute_train(
        "--num-rollout 1",
        4,
        None,
        before_ray_job_submit=lambda: launch_case.append("callback"),
        extra_env_vars={"LABEL": "a'b $value"},
    )
    assert len(launch_case) == 3
    assert resource in launch_case[0]
    assert launch_case[1] == "callback"
    assert "-- python3 train.py" in launch_case[2]
    env = submitted_env(launch_case)
    assert env["VIME_ACCELERATOR"] == backend
    assert env["LABEL"] == "a'b $value"
    assert not any("pkill" in c or "ray stop" in c for c in launch_case)
    if backend == "cuda":
        assert env["NCCL_NVLS_ENABLE"] == "1"
        assert env["PYTHONPATH"] == "/root/Megatron-LM/"
    else:
        assert "NCCL_NVLS_ENABLE" not in env
        assert env["HCCL_CONNECT_TIMEOUT"] == "7200"


@pytest.mark.unit
@pytest.mark.parametrize("external,running", [(False, True), (True, False)])
def test_reuse_preserves_detached_engines(launch_case, monkeypatch, external, running):
    monkeypatch.setenv("VIME_ACCELERATOR", "npu")
    monkeypatch.setenv("VIME_SCRIPT_EXTERNAL_RAY", str(int(external)))

    def status(*args, **kwargs):
        assert not external
        assert args[0] == ["ray", "status"]
        return SimpleNamespace(returncode=0 if running else 1)

    monkeypatch.setattr(commands.subprocess, "run", status)
    commands.execute_train("--recover", 8, None)
    assert len(launch_case) == 1
    assert "ray job submit" in launch_case[0]


@pytest.mark.unit
def test_submit_switch_and_configuration_precedence(launch_case, monkeypatch):
    monkeypatch.setenv("VIME_ACCELERATOR", "npu")
    monkeypatch.setenv("VIME_SCRIPT_ENABLE_RAY_SUBMIT", "0")
    commands.execute_train("", 2, None, before_ray_job_submit=lambda: launch_case.append("callback"))
    assert len(launch_case) == 2
    monkeypatch.setenv("VIME_SCRIPT_ENABLE_RAY_SUBMIT", "1")
    commands.execute_train(
        "",
        2,
        None,
        extra_env_vars={"HCCL_CONNECT_TIMEOUT": "12"},
        config=commands.ExecuteTrainConfig(extra_env_vars='{"HCCL_CONNECT_TIMEOUT": "13"}'),
    )
    assert "HCCL_CONNECT_TIMEOUT=13" in launch_case[2]
    assert submitted_env(launch_case)["HCCL_CONNECT_TIMEOUT"] == "13"


@pytest.mark.unit
@pytest.mark.parametrize("key,value", [("VIME_ACCELERATOR", "npu"), ("VIME_PLUGINS", "another")])
def test_conflicting_child_support_rejected_before_start(launch_case, key, value):
    with pytest.raises(ValueError, match="must match"):
        commands.execute_train("", 1, None, extra_env_vars={key: value})
    assert launch_case == []


@pytest.mark.unit
def test_custom_resource_with_empty_environment_still_uses_common_lifecycle(launch_case, monkeypatch):
    from vime.backends.megatron_integration import MegatronIntegration
    from vime.backends.vllm_weight_transfer import WeightTransfer
    from vime.utils.device_env import DeviceEnv

    class EmptyEnv(DeviceEnv):
        def launch_job_env(self):
            return {}

    class Third(accelerator.CUDAAccelerator):
        name = "third"
        ray_resource_name = "THIRD"
        ray_uses_native_gpu_resource = False

    monkeypatch.setattr(accelerator, "_REGISTRY", accelerator._REGISTRY.copy())
    accelerator.register_accelerator("third", Third, lambda: False)
    module = ModuleType("third_support")
    module.build_integration = lambda: platform_integration.PlatformIntegration(
        EmptyEnv(), MegatronIntegration(), WeightTransfer()
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("VIME_ACCELERATOR", "third")
    monkeypatch.setenv("VIME_PLUGINS", module.__name__)
    inputs = launch_inputs(2)
    assert inputs.ray_args == "--num-gpus 0 --resources '{\"THIRD\": 2}'"
    commands.execute_train("", 2, None)
    assert "THIRD" in launch_case[0]
    assert submitted_env(launch_case)["VIME_ACCELERATOR"] == "third"


@pytest.mark.unit
def test_rocm_direct_launch_keeps_main_behavior(launch_case, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test")
    commands.execute_train("--num-rollout 1", 4, None)
    assert "python3 train.py" in launch_case[1]
    assert "ray job submit" not in launch_case[1]
    assert f"{commands.repo_base_dir}:/root/Megatron-LM/" in launch_case[1]


@pytest.mark.unit
def test_shell_environment_is_quoted_and_keys_validated():
    assert shell_exports({"LABEL": "x'y $z"}) == "export LABEL='x'\"'\"'y $z' && "
    with pytest.raises(ValueError, match="Invalid environment"):
        shell_exports({"BAD;KEY": "value"})


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["cuda", "musa", "supa", "npu", "third"])
def test_explicit_checkpoint_conversion_does_not_depend_on_hardware(launch_case, monkeypatch, tmp_path, backend):
    monkeypatch.setenv("VIME_ACCELERATOR", backend)
    monkeypatch.setattr(accelerator, "describe_accelerator", lambda: pytest.fail("Conversion is a job choice"))
    commands.convert_checkpoint("test-model", "test-model", 4, dir_dst=str(tmp_path))
    assert len(launch_case) == 1
    assert "torchrun --nproc-per-node 4" in launch_case[0]
    assert "tools/convert_hf_to_torch_dist.py" in launch_case[0]
    assert f"--save {tmp_path}/test-model_torch_dist" in launch_case[0]
