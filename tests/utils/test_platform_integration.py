"""Backend components remain lazy and child selection follows the accelerator."""

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from vime.backends.megatron_integration import current as training
from vime.utils import accelerator, device_env


@pytest.fixture(autouse=True)
def isolated_integration(monkeypatch):
    monkeypatch.setattr(accelerator, "_ACCELERATOR", None)
    monkeypatch.setenv("VIME_ACCELERATOR", "cuda")
    monkeypatch.delenv("VIME_PLUGINS", raising=False)


def install_module(monkeypatch, **components):
    module = types.ModuleType("test_device_integration")
    vars(module).update(components)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("VIME_PLUGINS", module.__name__)
    return module


@pytest.mark.unit
def test_environment_only_module_does_not_require_training_or_transfer(monkeypatch):
    env = device_env.DeviceEnv()
    install_module(monkeypatch, env=env)
    assert device_env.current() is env
    assert device_env.current() is env
    with pytest.raises(TypeError, match="must provide 'megatron'"):
        training()


@pytest.mark.unit
def test_missing_module_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("VIME_PLUGINS", "nonexistent_device_integration")
    with pytest.raises(RuntimeError, match="nonexistent_device_integration"):
        device_env.current()


@pytest.mark.unit
def test_empty_override_cannot_silently_use_cuda_for_npu(monkeypatch):
    monkeypatch.setenv("VIME_ACCELERATOR", "npu")
    monkeypatch.setenv("VIME_PLUGINS", "")
    with pytest.raises(RuntimeError, match="requires a backend module"):
        device_env.current()


@pytest.mark.unit
def test_unknown_device_needs_an_explicit_backend_module(monkeypatch):
    monkeypatch.setattr(accelerator, "describe_accelerator", lambda: types.SimpleNamespace(name="third"))
    with pytest.raises(RuntimeError, match="requires a backend module"):
        device_env.current()
    env = device_env.DeviceEnv()
    install_module(monkeypatch, env=env)
    assert device_env.current() is env


@pytest.mark.unit
@pytest.mark.parametrize("requested", [None, "auto"])
def test_implicit_gpu_environment_is_unchanged(monkeypatch, requested):
    if requested is None:
        monkeypatch.delenv("VIME_ACCELERATOR")
    else:
        monkeypatch.setenv("VIME_ACCELERATOR", requested)
    monkeypatch.setattr(accelerator, "describe_accelerator", lambda: types.SimpleNamespace(name="cuda"))
    assert device_env.current().launch_env() == {}
    assert device_env.for_child({"KEEP": "1"}) == {"KEEP": "1"}


@pytest.mark.unit
@pytest.mark.parametrize("backend,module", [("cuda", None), ("npu", "vime_plugins.ascend")])
def test_child_environment_propagates_selection_and_rejects_conflicts(monkeypatch, backend, module):
    monkeypatch.setenv("VIME_ACCELERATOR", backend)
    env = device_env.for_child({"KEEP": "1"})
    assert env["VIME_ACCELERATOR"] == backend
    assert env.get("VIME_PLUGINS") == module
    assert env["KEEP"] == "1"
    for key in ("VIME_ACCELERATOR", "VIME_PLUGINS"):
        with pytest.raises(ValueError, match="must match"):
            device_env.for_child({key: "wrong"})


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["cuda", "npu"])
def test_fresh_process_description_and_assembly_do_not_import_vendor_runtime(backend):
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "VIME_ACCELERATOR": backend, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "PYTHONPATH": str(root)}
    env.pop("VIME_PLUGINS", None)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["ASCEND_RT_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc, sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch_npu', 'megatron', 'megatron_adaptor', 'vllm', 'vllm_ascend', 'mindspeed'}:
            raise AssertionError('unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, BlockRuntime())
from vime.utils import accelerator
from vime.backends.vllm_weight_transfer import current as transfer
from vime.utils.device_env import current, for_child
backend = accelerator.describe_accelerator()
assert current() is current()
assert for_child(current().launch_env())['VIME_ACCELERATOR'] == backend.name
assert accelerator.initialize_accelerator(require_device=False) is None
import vime.backends.vllm_utils
if backend.name == 'cuda':
    import vime.backends.megatron_utils
if backend.name == 'npu':
    assert 'hccl' == transfer().backend('distributed')
try:
    accelerator.get_accelerator()
except RuntimeError:
    pass
else:
    raise AssertionError('CPU execution must fail')
""",
        ],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
