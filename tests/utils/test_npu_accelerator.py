"""CPU contracts for NPU selection and pre-Megatron bootstrap ordering."""

import os
from types import SimpleNamespace

import pytest
import torch

from vime.platforms import current_platform, get_platform, reset_platform_cache
from vime.platforms import npu
from vime.platforms.npu import NPUAccelerator
from vime.utils import accelerator


@pytest.fixture(autouse=True)
def npu_runtime(monkeypatch):
    reset_platform_cache()
    monkeypatch.setenv("VIME_PLATFORM", "npu")
    monkeypatch.delenv("VIME_ACCELERATOR", raising=False)
    monkeypatch.delenv("MUSA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("MUSA_PATCH_PATH", raising=False)
    monkeypatch.setattr(accelerator, "_REGISTRY", {})
    monkeypatch.setattr(accelerator, "_ACCELERATOR", None)
    monkeypatch.setattr(accelerator, "_cuda_available", lambda: False)
    monkeypatch.setattr(accelerator, "is_musa_available", lambda: False)
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
    reset_platform_cache()


def test_platform_registers_npu_before_main_auto_selection(monkeypatch, npu_runtime):
    # MindSpeed may also make CUDA's availability probe return true.
    monkeypatch.setattr(accelerator, "_cuda_available", lambda: True)
    assert current_platform().is_npu
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


def test_npu_allocator_remains_owned_by_existing_runtime_hooks(monkeypatch):
    monkeypatch.setenv("VIME_ENABLE_EXPANDABLE_SEGMENTS", "1")
    assert NPUAccelerator().set_allocator_expandable_segments() is False


def test_main_npu_override_selects_matching_platform(monkeypatch):
    monkeypatch.delenv("VIME_PLATFORM")
    monkeypatch.setenv("VIME_ACCELERATOR", "npu")
    assert current_platform().is_npu
    assert accelerator.get_accelerator().name == "npu"


@pytest.mark.parametrize("platform,backend", [("npu", "cuda"), ("cuda", "npu"), ("npu", "musa")])
def test_conflicting_overrides_fail_before_bootstrap(monkeypatch, platform, backend):
    monkeypatch.setenv("VIME_PLATFORM", platform)
    monkeypatch.setenv("VIME_ACCELERATOR", backend)
    with pytest.raises(ValueError, match="Conflicting VIME_PLATFORM"):
        current_platform()


def test_registered_npu_does_not_override_explicit_cuda_platform(monkeypatch):
    get_platform("npu")
    monkeypatch.setenv("VIME_PLATFORM", "cuda")
    monkeypatch.setattr(accelerator, "_cuda_available", lambda: True)
    monkeypatch.setattr(accelerator.CUDAAccelerator, "is_available", lambda self: True)
    assert current_platform().name == "cuda"
    assert accelerator.initialize_accelerator().name == "cuda"


@pytest.mark.parametrize("gdn_enabled", [False, True])
def test_bootstrap_selects_npu_before_adaptor_and_attention(monkeypatch, gdn_enabled):
    if gdn_enabled:
        monkeypatch.setenv("FLA_NPU_OPP_PATH", "/installed/fla")
    else:
        monkeypatch.delenv("FLA_NPU_OPP_PATH", raising=False)
    events = []
    bootstrap = current_platform().megatron
    monkeypatch.setattr(npu, "_ensure_torch_npu", lambda: events.append("torch_npu"))
    monkeypatch.setattr(npu, "_install_safe_empty_cache", lambda: events.append("empty_cache_guard"))
    monkeypatch.setattr(npu, "_prioritize_fla_npu_opp", lambda: events.append("fla_priority") if gdn_enabled else None)
    original_import = npu.importlib.import_module

    def import_module(name, *args, **kwargs):
        if name == "fla_npu":
            assert gdn_enabled
            events.append(name)
            return SimpleNamespace()
        if name in {"megatron_adaptor", "vime.backends.megatron_utils.npu_attention_patch"}:
            assert accelerator.get_accelerator().name == "npu"
            events.append(name)
            bootstrap.bootstrap()  # Recursive imports must not repeat initialization.
            return SimpleNamespace()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(npu.importlib, "import_module", import_module)
    bootstrap.bootstrap()
    bootstrap.bootstrap()
    assert events == (["fla_npu"] if gdn_enabled else []) + [
        "torch_npu",
        "empty_cache_guard",
        "megatron_adaptor",
        "vime.backends.megatron_utils.npu_attention_patch",
    ] + (["fla_priority"] if gdn_enabled else [])


def test_gdn_opp_priority_keeps_other_vendors(monkeypatch):
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", "/serving:/installed/opp:/installed/opp/vendors/fla_npu_transformer")
    monkeypatch.delenv("FLA_NPU_OPP_PATH", raising=False)
    npu._prioritize_fla_npu_opp()
    assert os.environ["ASCEND_CUSTOM_OPP_PATH"].startswith("/serving:")
    monkeypatch.setenv("FLA_NPU_OPP_PATH", "/installed/opp/vendors/fla_npu_transformer")
    monkeypatch.setenv("FLA_NPU_OP_API_LIB", "/installed/opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so")
    npu._prioritize_fla_npu_opp()
    npu._prioritize_fla_npu_opp()
    assert os.environ["ASCEND_CUSTOM_OPP_PATH"] == "/installed/opp:/installed/opp/vendors/fla_npu_transformer:/serving"


@pytest.fixture
def fla_serving_env(monkeypatch, tmp_path):
    vendor = tmp_path / "fla" / "opp" / "vendors" / "fla_npu_transformer"
    lib_dir = vendor / "op_api" / "lib"
    lib_dir.mkdir(parents=True)
    package = tmp_path / "vllm_ascend"
    serving = package / "_cann_ops_custom" / "vendors" / "custom_transformer"
    serving.mkdir(parents=True)
    monkeypatch.setattr(npu, "find_spec", lambda name: SimpleNamespace(origin=str(package / "__init__.py")))
    monkeypatch.setenv("FLA_NPU_OPP_PATH", str(vendor))
    monkeypatch.setenv("FLA_NPU_OP_API_LIB", str(lib_dir / "libcust_opapi.so"))
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", f"{vendor.parent.parent}:{vendor}:/other/opp:{serving}")
    monkeypatch.setenv("ASCEND_OPP_PATH", "/cann/opp")
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{lib_dir}:/cann/lib:/driver/lib")
    monkeypatch.setenv("LD_PRELOAD", f"{lib_dir}/libcust_opapi.so /other/memory_saver.so")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    return vendor, serving


@pytest.mark.parametrize("colocate", [False, True])
def test_rollout_isolates_fla_before_actor_start_without_changing_training(monkeypatch, fla_serving_env, colocate):
    vendor, serving = fla_serving_env
    parent_env = dict(os.environ)
    args = SimpleNamespace(colocate=colocate, offload_train=colocate, train_backend="megatron")
    overrides = {"KEEP": "1"}
    platform = current_platform()
    train_env = {**parent_env, **platform.ray.train_runtime_env(args, overrides)}
    rollout_env = platform.ray.rollout_runtime_env(args, overrides)
    effective = {**parent_env, **rollout_env}
    assert effective["ASCEND_CUSTOM_OPP_PATH"] == f"{serving}:/other/opp"
    assert effective["ASCEND_OPP_PATH"] == "/cann/opp"
    assert effective["LD_LIBRARY_PATH"] == "/cann/lib:/driver/lib"
    assert effective["LD_PRELOAD"] == "/other/memory_saver.so"
    assert effective["FLA_NPU_OPP_PATH"] == effective["FLA_NPU_OP_API_LIB"] == ""
    assert effective["OMP_NUM_THREADS"] == "1"
    assert effective["KEEP"] == "1"
    for key in ("ASCEND_CUSTOM_OPP_PATH", "FLA_NPU_OPP_PATH", "FLA_NPU_OP_API_LIB", "LD_LIBRARY_PATH"):
        assert train_env[key] == parent_env[key]
    assert os.environ == parent_env
    assert overrides == {"KEEP": "1"}
    child_env = platform.vllm.subprocess_env(effective, visible_devices="4,5", colocate=colocate)
    assert child_env["ASCEND_CUSTOM_OPP_PATH"] == effective["ASCEND_CUSTOM_OPP_PATH"]
    assert child_env["FLA_NPU_OPP_PATH"] == child_env["FLA_NPU_OP_API_LIB"] == ""
    assert child_env["ASCEND_RT_VISIBLE_DEVICES"] == "4,5"
    assert child_env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_fla_isolation_preserves_other_vendors_under_shared_opp_root(monkeypatch, fla_serving_env):
    vendor, serving = fla_serving_env
    other = vendor.parent / "other_vendor"
    other.mkdir()
    alias = vendor.parent / "fla_alias"
    alias.symlink_to(vendor, target_is_directory=True)
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", f"{vendor.parent.parent}:{alias}:{other}")
    env = current_platform().ray.rollout_runtime_env(SimpleNamespace(colocate=False))
    assert env["ASCEND_CUSTOM_OPP_PATH"] == f"{serving}:{other}"


def test_fla_isolation_accepts_resolved_vendor_without_loaded_api(monkeypatch, fla_serving_env):
    vendor, serving = fla_serving_env
    monkeypatch.delenv("FLA_NPU_OP_API_LIB")
    # The launcher can pass a vendor directory or the OPP root containing it.
    monkeypatch.setenv("FLA_NPU_OPP_PATH", str(vendor.parent.parent))
    env = current_platform().vllm.subprocess_env({}, visible_devices="0", colocate=False)
    assert env["ASCEND_CUSTOM_OPP_PATH"] == f"{serving}:/other/opp"
    assert env["FLA_NPU_OP_API_LIB"] == ""


def test_fla_isolation_fails_clearly_when_serving_package_is_missing(monkeypatch, fla_serving_env):
    monkeypatch.setattr(npu, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="installed vllm_ascend"):
        current_platform().ray.rollout_runtime_env(SimpleNamespace(colocate=False))


def test_no_fla_job_keeps_existing_launch_environment(monkeypatch):
    monkeypatch.delenv("FLA_NPU_OPP_PATH", raising=False)
    monkeypatch.delenv("FLA_NPU_OP_API_LIB", raising=False)

    def unexpected_lookup(name):
        raise AssertionError(f"non-FLA jobs must not probe {name}")

    monkeypatch.setattr(npu, "find_spec", unexpected_lookup)
    env = {"ASCEND_CUSTOM_OPP_PATH": "/other/opp", "LD_LIBRARY_PATH": "/cann/lib", "OMP_NUM_THREADS": "8"}
    actual = current_platform().ray.rollout_runtime_env(SimpleNamespace(colocate=False), env)
    assert all(actual[key] == value for key, value in env.items())
    assert "FLA_NPU_OPP_PATH" not in actual
    assert "FLA_NPU_OP_API_LIB" not in actual


@pytest.mark.parametrize("colocate", [False, True])
@pytest.mark.parametrize("worker_method", [None, "fork", "spawn"])
def test_npu_serving_uses_spawn_without_changing_parent_env(monkeypatch, colocate, worker_method):
    monkeypatch.delenv("FLA_NPU_OPP_PATH", raising=False)
    monkeypatch.delenv("FLA_NPU_OP_API_LIB", raising=False)
    base_env = {"KEEP": "1"}
    if worker_method is not None:
        base_env["VLLM_WORKER_MULTIPROC_METHOD"] = worker_method
    parent_env = dict(os.environ)

    env = current_platform().vllm.subprocess_env(base_env, visible_devices="4,5", colocate=colocate)

    assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert env["KEEP"] == "1"
    assert base_env.get("VLLM_WORKER_MULTIPROC_METHOD") == worker_method
    assert os.environ == parent_env


def test_bootstrap_rejects_preselected_cuda_without_replacing_it(monkeypatch):
    selected = accelerator.CUDAAccelerator()
    monkeypatch.setattr(accelerator, "_ACCELERATOR", selected)
    monkeypatch.setattr(npu, "_ensure_torch_npu", lambda: None)
    bootstrap = current_platform().megatron
    with pytest.raises(RuntimeError, match="already selected 'cuda'"):
        bootstrap.bootstrap()
    assert accelerator._ACCELERATOR is selected
    assert not bootstrap._bootstrapping
    assert not bootstrap._bootstrapped


def test_repatch_passes_typed_args_and_restores_attention(monkeypatch):
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
        "vime.backends.megatron_utils.npu_attention_patch": SimpleNamespace(
            DotProductAttention=attention_class,
            npu_dot_product_attention_forward=vime_forward,
        ),
    }
    bootstrap = current_platform().megatron
    original_import = npu.importlib.import_module

    def import_module(name, *args, **kwargs):
        if name in modules:
            return modules[name]
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(npu.importlib, "import_module", import_module)
    bootstrap.repatch(args)
    assert events == ["remove", ("pre", full_args), "features"]
    assert attention_class.forward is vime_forward
    assert args.vllm_additional_config is typed_config
