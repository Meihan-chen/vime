"""NPU launch search paths must be overridable, not baked into the source.

The reference image layout stays the default, so an unset environment keeps
the historical strings; a site that installs CANN or the Megatron forks
elsewhere overrides the roots instead of patching launch.py.
"""

from __future__ import annotations

import importlib
import importlib.resources

import pytest

MODULE = "vime.utils.external_utils.launch"

REFERENCE_PYTHONPATH = (
    "/root/Megatron-LM:/root/vime:/root/MegatronAdaptor:/root/TransformerEngineNPU:"
    "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:"
    "/usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/torch_npu_bridge"
)


@pytest.fixture
def launch(monkeypatch):
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
def test_defaults_match_the_reference_image_layout(launch):
    assert launch._npu_pythonpath() == REFERENCE_PYTHONPATH
    assert launch._npu_ld_library_path().startswith("/usr/local/Ascend/driver/lib64:")
    assert launch._npu_ld_library_path().endswith("/usr/local/Ascend/cann/aarch64-linux/devlib")


@pytest.mark.unit
def test_toolkit_home_is_honoured(launch, monkeypatch):
    # CANN points ASCEND_TOOLKIT_HOME at a versioned directory on real images.
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", "/opt/cann-9.1.0")

    assert "/opt/cann-9.1.0/python/site-packages" in launch._npu_pythonpath()
    assert "/opt/cann-9.1.0/lib64" in launch._npu_ld_library_path()


@pytest.mark.unit
def test_repo_root_is_honoured(launch, monkeypatch):
    monkeypatch.setenv("VIME_REPO_ROOT", "/workspace")

    pythonpath = launch._npu_pythonpath()

    assert pythonpath.startswith("/workspace/Megatron-LM:/workspace/vime:")
    assert "/root/" not in pythonpath


@pytest.mark.unit
@pytest.mark.parametrize(
    "variable,accessor",
    [
        ("VIME_NPU_PYTHONPATH", "_npu_pythonpath"),
        ("VIME_NPU_LD_LIBRARY_PATH", "_npu_ld_library_path"),
    ],
)
def test_full_override_wins(launch, monkeypatch, variable, accessor):
    monkeypatch.setenv(variable, "/only/this")

    assert getattr(launch, accessor)() == "/only/this"


@pytest.mark.unit
def test_source_carries_no_hardcoded_repo_paths():
    # The literals must live in the two default helpers only.
    source = importlib.resources.files("vime.utils.external_utils").joinpath("launch.py").read_text()
    assert '"/root/Megatron-LM:/root/vime:' not in source
