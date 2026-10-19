"""Vendor lifecycle and failure contracts with only external dependencies replaced."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vime_plugins.ascend import patches
from vime_plugins.ascend.training import NpuMegatronSupport


@pytest.mark.unit
def test_cached_adaptor_does_not_skip_our_attention_patch(monkeypatch):
    original = object()
    override = object()
    attention = SimpleNamespace(
        DotProductAttention=SimpleNamespace(forward=original), npu_dot_product_attention_forward=override
    )
    modules = {"torch_npu": object(), "megatron_adaptor": object(), "vime_plugins.ascend.attention_patch": attention}
    calls = []
    monkeypatch.setattr(patches.accelerator, "describe_accelerator", lambda: SimpleNamespace(name="npu"))
    monkeypatch.setattr(patches, "_install_safe_empty_cache", lambda: calls.append("cache"))
    monkeypatch.setattr(patches.importlib, "import_module", lambda name: calls.append(name) or modules[name])
    monkeypatch.setitem(sys.modules, "megatron_adaptor", modules["megatron_adaptor"])
    patches.before_megatron_import()
    assert attention.DotProductAttention.forward is override
    assert calls == ["torch_npu", "cache", "megatron_adaptor", "vime_plugins.ascend.attention_patch"]


@pytest.mark.unit
@pytest.mark.parametrize("flag", ["disable_param_buffers_cpu_backup", "disable_grad_buffers_cpu_backup"])
def test_unsupported_buffer_flags_fail_before_patch_mutation(monkeypatch, flag):
    monkeypatch.setattr(patches, "after_megatron_args", lambda _: pytest.fail("must reject first"))
    with pytest.raises(ValueError, match="unsupported"):
        NpuMegatronSupport().after_args(SimpleNamespace(**{flag: True}))


@pytest.mark.unit
def test_argument_patch_failure_requires_a_new_process(monkeypatch):
    def fail(_):
        raise LookupError("partial mutation")

    monkeypatch.setattr(patches, "after_megatron_args", fail)
    instance = NpuMegatronSupport()
    with pytest.raises(LookupError, match="partial mutation"):
        instance.after_args(SimpleNamespace())
    with pytest.raises(RuntimeError, match="restart"):
        instance.after_args(SimpleNamespace())


@pytest.mark.unit
def test_checkpoint_validation_stays_on_and_escape_hatch_is_explicit(monkeypatch, caplog):
    def original(plan, metadata):
        return False

    planner = SimpleNamespace(_validate_global_plan=original)
    monkeypatch.delenv("VIME_NPU_SKIP_CHECKPOINT_VALIDATION", raising=False)
    NpuMegatronSupport().after_checkpoint_setup(planner)
    assert planner._validate_global_plan is original
    assert planner._validate_global_plan({}, {}) is False
    monkeypatch.setenv("VIME_NPU_SKIP_CHECKPOINT_VALIDATION", "1")
    NpuMegatronSupport().after_checkpoint_setup(planner)
    assert planner._validate_global_plan({}, {}) is True
    assert "disables checkpoint shard integrity" in caplog.text
    with pytest.raises(RuntimeError, match="target .* missing"):
        NpuMegatronSupport().after_checkpoint_setup(SimpleNamespace())


@pytest.mark.unit
@pytest.mark.parametrize(
    "training,packed,keep", [(False, False, 1.0), (True, False, 0.8), (False, True, 1.0), (True, True, 0.8)]
)
def test_attention_eval_disables_dropout_and_preserves_layout(monkeypatch, training, packed, keep):
    calls = []
    modules = {
        "torch_npu": {"npu_fusion_attention": lambda *a, **kw: calls.append((a, kw)) or ("output",)},
        "megatron.core.packed_seq_params": {"PackedSeqParams": type("PackedSeqParams", (), {})},
        "megatron.core.transformer.enums": {"AttnMaskType": SimpleNamespace(no_mask="no-mask")},
        "megatron.core.transformer.dot_product_attention": {
            "DotProductAttention": type("DotProductAttention", (), {})
        },
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[2] / "vime_plugins/ascend/attention_patch.py"
    spec = importlib.util.spec_from_file_location("_attention_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instance = SimpleNamespace(
        config=SimpleNamespace(), softmax_scale=0.5, attention_dropout=SimpleNamespace(p=0.2), training=training
    )
    x = torch.zeros((4, 2, 3) if packed else (4, 1, 2, 3))
    params = SimpleNamespace(cu_seqlens_q=[0, 2, 4], cu_seqlens_kv=[0, 2, 4]) if packed else None
    assert module.npu_dot_product_attention_forward(instance, x, x, x, None, packed_seq_params=params) == "output"
    args, kwargs = calls[0]
    assert kwargs["keep_prob"] == keep
    assert args[4] == ("TND" if packed else "SBH")
    assert kwargs["actual_seq_qlen"] == ([0, 2, 4] if packed else None)


@pytest.mark.unit
def test_empty_visible_mask_and_manager_only_host_do_not_bootstrap(monkeypatch):
    from vime.utils.accelerator import npu

    monkeypatch.setattr(npu.os.path, "exists", lambda _: True)
    monkeypatch.setattr(npu, "glob", lambda _: [])
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    assert npu.has_visible_device() is False
    monkeypatch.setattr(npu, "glob", lambda _: ["/dev/davinci0"])
    for mask in ("", "-1"):
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", mask)
        assert npu.has_visible_device() is False


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["cuda", "npu"])
@pytest.mark.parametrize("options", [{"colocate": True}, {"offload_train": True}, {"advantage_estimator": "ppo"}])
def test_real_argument_validation_keeps_supported_offload_policy(monkeypatch, backend, options):
    from vime.utils import accelerator, support

    monkeypatch.setattr(accelerator, "_ACCELERATOR", None)
    monkeypatch.setattr(support, "_selected", None)
    monkeypatch.setenv("VIME_ACCELERATOR", backend)
    monkeypatch.delenv("VIME_PLUGINS", raising=False)
    path = Path(__file__).resolve().parents[1] / "test_megatron_argument_validation.py"
    spec = importlib.util.spec_from_file_location("_argument_contract_helpers", path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    module = helpers.load_vime_arguments_module(monkeypatch)
    args, _ = module.vime_validate_args(helpers.make_vime_validate_args(**options))
    assert args.offload_train is True
    for field in ("disable_param_buffers_cpu_backup", "disable_grad_buffers_cpu_backup"):
        assert getattr(args, field, False) is (backend == "cuda")
    if backend == "npu":
        calls = []
        monkeypatch.setattr(patches, "after_megatron_args", calls.append)
        support.current_support().megatron.after_args(args)
        assert calls == [args]
