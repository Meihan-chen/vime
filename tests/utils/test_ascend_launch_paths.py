"""NPU launch search paths must be overridable, not baked into the source.

The reference image layout stays the default, so an unset environment keeps
the historical strings; a site that installs CANN or the Megatron forks
elsewhere overrides the roots instead of editing the plugin.
"""

from __future__ import annotations

import importlib
import importlib.resources

import pytest

MODULE = "vime_plugins.ascend.device_env"

REFERENCE_PYTHONPATH = (
    "/root/Megatron-LM:/root/vime:/root/MegatronAdaptor:/root/TransformerEngineNPU:"
    "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:"
    "/usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/torch_npu_bridge"
)


@pytest.fixture
def device_env(monkeypatch):
    for name in (
        "ASCEND_HOME",
        "ASCEND_TOOLKIT_HOME",
        "VIME_REPO_ROOT",
        "VIME_NPU_PYTHONPATH",
        "VIME_NPU_LD_LIBRARY_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    return importlib.reload(importlib.import_module(MODULE))


@pytest.mark.unit
def test_defaults_match_the_reference_image_layout(device_env):
    assert device_env._launch_pythonpath() == REFERENCE_PYTHONPATH
    assert device_env._launch_ld_library_path().startswith("/usr/local/Ascend/driver/lib64:")
    assert device_env._launch_ld_library_path().endswith("/usr/local/Ascend/cann/aarch64-linux/devlib")


@pytest.mark.unit
def test_toolkit_home_is_honoured(device_env, monkeypatch):
    # CANN points ASCEND_TOOLKIT_HOME at a versioned directory on real images.
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", "/opt/cann-9.1.0")

    assert "/opt/cann-9.1.0/python/site-packages" in device_env._launch_pythonpath()
    assert "/opt/cann-9.1.0/lib64" in device_env._launch_ld_library_path()


@pytest.mark.unit
def test_repo_root_is_honoured(device_env, monkeypatch):
    monkeypatch.setenv("VIME_REPO_ROOT", "/workspace")

    pythonpath = device_env._launch_pythonpath()

    assert pythonpath.startswith("/workspace/Megatron-LM:/workspace/vime:")
    assert "/root/" not in pythonpath


@pytest.mark.unit
@pytest.mark.parametrize(
    "variable,accessor",
    [
        ("VIME_NPU_PYTHONPATH", "_launch_pythonpath"),
        ("VIME_NPU_LD_LIBRARY_PATH", "_launch_ld_library_path"),
    ],
)
def test_full_override_wins(device_env, monkeypatch, variable, accessor):
    monkeypatch.setenv(variable, "/only/this")

    assert getattr(device_env, accessor)() == "/only/this"


@pytest.mark.unit
def test_source_carries_no_hardcoded_repo_paths():
    # The literals must live in the two default helpers only.
    source = importlib.resources.files("vime_plugins.ascend").joinpath("device_env.py").read_text()
    assert '"/root/Megatron-LM:/root/vime:' not in source
