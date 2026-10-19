"""Exercise real platform integration assembly without importing vendor runtimes."""

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from vime.backends.megatron_integration import MegatronIntegration
from vime.backends.vllm_weight_transfer import WeightTransfer
from vime.utils import accelerator, device_env, platform_integration


@pytest.fixture(autouse=True)
def isolated_integration(monkeypatch):
    monkeypatch.setattr(accelerator, "_ACCELERATOR", None)
    monkeypatch.setattr(platform_integration, "_selected", None)
    monkeypatch.setattr(platform_integration, "_building", False)
    monkeypatch.setenv("VIME_ACCELERATOR", "cuda")
    monkeypatch.delenv("VIME_PLUGINS", raising=False)


def install_factory(monkeypatch, factory):
    module = types.ModuleType("test_device_integration")
    module.build_integration = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("VIME_PLUGINS", module.__name__)


def default_integration():
    return platform_integration.PlatformIntegration(device_env.DeviceEnv(), MegatronIntegration(), WeightTransfer())


@pytest.mark.unit
def test_factory_runs_once_and_all_consumers_share_components(monkeypatch):
    calls = []
    chosen = default_integration()
    install_factory(monkeypatch, lambda: calls.append("build") or chosen)
    selected = platform_integration.current_integration()
    assert platform_integration.current_integration() is selected
    from vime.backends.megatron_integration import current as training
    from vime.backends.vllm_weight_transfer import current as transfer

    assert training() is chosen.megatron
    assert transfer() is chosen.weight_transfer
    assert device_env.current() is selected.env
    assert device_env.current() is device_env.current()
    assert device_env.current().policy is chosen.env
    assert calls == ["build"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad", [None, {}, platform_integration.PlatformIntegration(None, MegatronIntegration(), WeightTransfer())]
)
def test_invalid_factory_result_is_never_published(monkeypatch, bad):
    install_factory(monkeypatch, lambda: bad)
    with pytest.raises(TypeError):
        platform_integration.current_integration()
    assert platform_integration._selected is None
    install_factory(monkeypatch, default_integration)
    assert isinstance(platform_integration.current_integration(), platform_integration.PlatformIntegration)


@pytest.mark.unit
def test_stale_lifecycle_signature_rejected_before_patch_install(monkeypatch):
    class Stale(MegatronIntegration):
        def after_args(self):
            pytest.fail("validation must not run patches")

    install_factory(
        monkeypatch,
        lambda: platform_integration.PlatformIntegration(device_env.DeviceEnv(), Stale(), WeightTransfer()),
    )
    with pytest.raises(TypeError):
        platform_integration.current_integration()
    assert platform_integration._selected is None


@pytest.mark.unit
def test_recursive_factory_is_rejected_and_can_retry(monkeypatch):
    install_factory(monkeypatch, platform_integration.current_integration)
    with pytest.raises(RuntimeError, match="Recursive"):
        platform_integration.current_integration()
    install_factory(monkeypatch, default_integration)
    assert platform_integration.current_integration()


@pytest.mark.unit
def test_missing_factory_and_module_are_clear_errors(monkeypatch):
    install_factory(monkeypatch, None)
    with pytest.raises(TypeError, match="build_integration"):
        platform_integration.current_integration()
    monkeypatch.setenv("VIME_PLUGINS", "nonexistent_device_integration")
    with pytest.raises(RuntimeError, match="nonexistent_device_integration"):
        platform_integration.current_integration()


@pytest.mark.unit
def test_empty_override_requires_explicit_default_compatibility(monkeypatch):
    monkeypatch.setenv("VIME_ACCELERATOR", "npu")
    monkeypatch.setenv("VIME_PLUGINS", "")
    assert accelerator.describe_accelerator().name == "npu"
    with pytest.raises(RuntimeError, match="complete platform integration"):
        platform_integration.current_integration()


@pytest.mark.unit
def test_successful_assembly_cannot_switch_module(monkeypatch):
    platform_integration.current_integration()
    monkeypatch.setenv("VIME_PLUGINS", "another_integration")
    with pytest.raises(RuntimeError, match="cannot change"):
        platform_integration.current_integration()


@pytest.mark.unit
def test_unknown_platform_has_no_silent_cuda_fallback(monkeypatch):
    monkeypatch.setattr(accelerator, "describe_accelerator", lambda: types.SimpleNamespace(name="third"))
    with pytest.raises(RuntimeError, match="complete platform integration"):
        platform_integration.current_integration()
    install_factory(monkeypatch, default_integration)
    assert platform_integration.current_integration().weight_transfer.backend("distributed") == "nccl"


@pytest.mark.unit
@pytest.mark.parametrize("backend,module", [("cuda", ""), ("npu", "vime_plugins.ascend")])
def test_child_env_pins_platform_and_factory(monkeypatch, backend, module):
    monkeypatch.setenv("VIME_ACCELERATOR", backend)
    env = device_env.current().engine_actor_env(types.SimpleNamespace(colocate=False), {})
    assert env["VIME_ACCELERATOR"] == backend
    assert env["VIME_PLUGINS"] == module
    with pytest.raises(ValueError, match="Child VIME_ACCELERATOR"):
        device_env.current().engine_actor_env(types.SimpleNamespace(colocate=False), {"VIME_ACCELERATOR": "wrong"})


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
from vime.utils.platform_integration import current_integration
from vime.utils.device_env import current
backend = accelerator.describe_accelerator()
assert current_integration() is current_integration()
assert current().launch_env()['VIME_ACCELERATOR'] == backend.name
assert accelerator.initialize_accelerator(require_device=False) is None
import vime.backends.vllm_utils
if backend.name == 'cuda':
    import vime.backends.megatron_utils
if backend.name == 'npu':
    assert 'hccl' == current_integration().weight_transfer.backend('distributed')
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
