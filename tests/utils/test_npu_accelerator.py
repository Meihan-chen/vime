"""CPU contracts for NPU selection and pre-Megatron bootstrap ordering."""

from types import SimpleNamespace

import pytest
import torch

from vime.utils import accelerator
from vime.utils.accelerator.npu import NPUAccelerator
from vime_plugins.ascend import patches


@pytest.fixture(autouse=True)
def npu_runtime(monkeypatch):
    monkeypatch.setenv("VIME_ACCELERATOR", "npu")
    monkeypatch.delenv("MUSA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("MUSA_PATCH_PATH", raising=False)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(accelerator, "_REGISTRY", {})
    monkeypatch.setattr(accelerator, "_ACCELERATOR", None)
    monkeypatch.setattr(accelerator, "_NPU_BOOTSTRAP_CHECKED", True)
    monkeypatch.setattr(accelerator, "_cuda_available", lambda: False)
    monkeypatch.setattr(accelerator, "is_musa_available", lambda: False)
    monkeypatch.setattr(accelerator, "is_supa_available", lambda: False)
    monkeypatch.setattr(accelerator, "_has_npu_device", lambda: True)
    monkeypatch.setattr("vime.utils.accelerator.npu.has_visible_device", lambda: True)
    calls = []
    fake_npu = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 1,
        device_count=lambda: 2,
        set_device=lambda index: calls.append(("set_device", index)),
        synchronize=lambda: calls.append("synchronize"),
        empty_cache=lambda: calls.append("empty_cache"),
        ipc_collect=lambda: calls.append("ipc_collect"),
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    yield calls


@pytest.mark.unit
def test_explicit_npu_preserves_the_runtime_surface(monkeypatch, npu_runtime):
    # Explicit selection stays NPU even when a vendor library aliases CUDA.
    monkeypatch.setattr(accelerator, "_cuda_available", lambda: True)

    assert isinstance(accelerator.initialize_accelerator(), NPUAccelerator)
    assert accelerator.device_type() == "npu"
    assert accelerator.process_group_backend() == "hccl"
    assert accelerator.process_group_backend("gloo") == "gloo"
    assert accelerator.is_accelerator_backend("cpu:gloo,npu:hccl")
    assert not accelerator.is_accelerator_backend("gloo")
    assert accelerator.distributed_device_id() is None
    assert accelerator.visible_devices_env_key() == "ASCEND_RT_VISIBLE_DEVICES"
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,7")
    assert accelerator.resolve_visible_device_id("7") == 1
    accelerator.set_device(1)
    accelerator.synchronize()
    accelerator.ipc_collect()
    accelerator.empty_cache()
    assert npu_runtime == [("set_device", 1), "synchronize", "ipc_collect", "empty_cache"]


@pytest.mark.unit
def test_npu_allocator_remains_owned_by_existing_runtime_hooks(monkeypatch):
    monkeypatch.setenv("VIME_ENABLE_EXPANDABLE_SEGMENTS", "1")

    assert NPUAccelerator().set_allocator_expandable_segments() is False


@pytest.mark.unit
def test_npu_is_not_selected_without_a_visible_device(monkeypatch):
    from vime.utils.accelerator import npu as npu_backend

    monkeypatch.setattr(npu_backend, "has_visible_device", lambda: False)

    assert npu_backend.has_visible_device() is False
    assert npu_backend.is_npu_available() is False


@pytest.mark.unit
def test_explicit_cuda_request_wins_over_an_available_npu(monkeypatch):
    monkeypatch.setenv("VIME_ACCELERATOR", "cuda")
    monkeypatch.setattr(accelerator, "_cuda_available", lambda: True)
    monkeypatch.setattr(accelerator.CUDAAccelerator, "is_available", lambda self: True)

    assert accelerator.initialize_accelerator().name == "cuda"


@pytest.mark.unit
def test_an_unknown_explicit_request_names_what_is_registered(monkeypatch):
    monkeypatch.setenv("VIME_ACCELERATOR", "not-registered")

    with pytest.raises(ValueError, match="Unknown accelerator"):
        accelerator.get_accelerator()


@pytest.mark.unit
def test_safe_empty_cache_wraps_the_original_once(monkeypatch):
    calls = []

    def original_empty_cache():
        calls.append("original")
        raise RuntimeError("allocator is between offload states")

    fake_torch = SimpleNamespace(
        npu=SimpleNamespace(empty_cache=original_empty_cache),
        cuda=SimpleNamespace(empty_cache=lambda: None),
    )
    monkeypatch.setattr(patches.importlib, "import_module", lambda name: fake_torch if name == "torch" else None)

    patches._install_safe_empty_cache()
    wrapped = fake_torch.npu.empty_cache
    wrapped()
    patches._install_safe_empty_cache()

    assert calls == ["original"]
    assert fake_torch.npu.empty_cache is wrapped
    assert fake_torch.cuda.empty_cache is wrapped


@pytest.mark.unit
def test_bootstrap_selects_npu_before_adaptor_and_attention(monkeypatch):
    events = []
    monkeypatch.setitem(__import__("sys").modules, "torch_npu", SimpleNamespace())
    monkeypatch.delitem(__import__("sys").modules, "megatron_adaptor", raising=False)
    monkeypatch.setattr(patches, "_install_safe_empty_cache", lambda: events.append("empty_cache_guard"))
    original_import = patches.importlib.import_module

    def import_module(name, *args, **kwargs):
        if name in {"megatron_adaptor", "vime_plugins.ascend.attention_patch"}:
            assert accelerator.get_accelerator().name == "npu"
            events.append(name)
            return SimpleNamespace(DotProductAttention=SimpleNamespace(), npu_dot_product_attention_forward=object())
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(patches.importlib, "import_module", import_module)

    patches.before_megatron_import()

    assert events == [
        "empty_cache_guard",
        "megatron_adaptor",
        "vime_plugins.ascend.attention_patch",
    ]


@pytest.mark.unit
def test_support_runs_bootstrap_once_and_rejects_partial_retry(monkeypatch):
    from vime_plugins.ascend.training import NpuMegatronSupport

    calls = []
    monkeypatch.setattr(patches, "before_megatron_import", lambda: calls.append("install"))
    support = NpuMegatronSupport()
    support.before_import()
    support.before_import()
    assert calls == ["install"]

    def partial_failure():
        calls.append("partial")
        raise RuntimeError("partial patch")

    monkeypatch.setattr(patches, "before_megatron_import", partial_failure)
    failed = NpuMegatronSupport()
    with pytest.raises(RuntimeError, match="partial patch"):
        failed.before_import()
    with pytest.raises(RuntimeError, match="restart"):
        failed.before_import()
    assert calls == ["install", "partial"]


@pytest.mark.unit
def test_bootstrap_rejects_preselected_cuda_without_replacing_it(monkeypatch):
    selected = accelerator.CUDAAccelerator()
    monkeypatch.setattr(accelerator, "_ACCELERATOR", selected)
    monkeypatch.setitem(__import__("sys").modules, "torch_npu", SimpleNamespace())
    monkeypatch.delitem(__import__("sys").modules, "megatron_adaptor", raising=False)

    with pytest.raises(RuntimeError, match="already selected 'cuda'"):
        patches.before_megatron_import()

    assert accelerator._ACCELERATOR is selected


@pytest.mark.unit
def test_reapplying_patches_passes_typed_args_and_restores_attention(monkeypatch):
    events = []
    full_args = SimpleNamespace(adaptor_default=True)
    typed_config = {"weight_nz_mode": 0}
    args = SimpleNamespace(vllm_additional_config=typed_config, tensor_model_parallel_size=4)
    original_forward = object()
    vime_forward = object()
    attention_class = SimpleNamespace(forward=vime_forward)

    def apply_features(config):
        assert config is full_args
        assert config.vllm_additional_config is typed_config
        assert config.tensor_model_parallel_size == 4
        assert config.adaptor_default
        attention_class.forward = original_forward
        events.append("features")

    modules = {
        "megatron_adaptor.features_manager.features_manager": SimpleNamespace(
            FeaturesManager=SimpleNamespace(
                remove_patches=lambda: events.append("remove"),
                apply_features_pre_patches=lambda config: events.append(("pre", config)),
                apply_features_patches=apply_features,
            )
        ),
        "megatron_adaptor.utils.args_utils": SimpleNamespace(get_full_args=lambda: full_args),
        "vime_plugins.ascend.attention_patch": SimpleNamespace(
            DotProductAttention=attention_class,
            npu_dot_product_attention_forward=vime_forward,
        ),
    }
    original_import = patches.importlib.import_module

    def import_module(name, *args, **kwargs):
        if name in modules:
            return modules[name]
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(patches.importlib, "import_module", import_module)

    patches.after_megatron_args(args)

    assert events == ["remove", ("pre", full_args), "features"]
    assert attention_class.forward is vime_forward
    assert args.vllm_additional_config is typed_config


@pytest.mark.unit
def test_auto_selection_preserves_cuda_priority_and_ignores_ascend_hints(monkeypatch):
    monkeypatch.delenv("VIME_ACCELERATOR")
    monkeypatch.setenv("ASCEND_HOME_PATH", "/usr/local/Ascend")
    monkeypatch.setattr(accelerator, "_cuda_available", lambda: True)
    monkeypatch.setattr(accelerator.CUDAAccelerator, "is_available", lambda self: True)
    monkeypatch.setattr(
        accelerator, "_bootstrap_torch_npu_if_needed", lambda: pytest.fail("CUDA must not bootstrap NPU")
    )
    assert accelerator.initialize_accelerator().name == "cuda"


@pytest.mark.unit
def test_auto_selection_can_use_npu_when_no_existing_backend_is_available(monkeypatch):
    monkeypatch.delenv("VIME_ACCELERATOR")
    assert accelerator.initialize_accelerator().name == "npu"
