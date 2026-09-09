"""Real NPU GDN contracts. Run separately from CPU suites that stub Megatron.

Opt in with VIME_RUN_NPU_GDN_TESTS=1 after sourcing CANN's set_env.sh.
This is a small operator/model test, not the 16-NPU Qwen3.5 E2E.
"""

import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vime.utils.external_utils.launch import get_fla_npu_runtime_env

NUM_GPUS = 1
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def runtime():
    if os.environ.get("VIME_RUN_NPU_GDN_TESTS") != "1":
        pytest.skip("requires explicit opt-in and the frozen NPU vendor environment")
    # The E2E launcher propagates this environment before starting Ray workers.
    os.environ.update(get_fla_npu_runtime_env())
    from vime.platforms import current_platform

    assert current_platform().is_npu
    import vime.backends.megatron_utils  # noqa: F401 - bootstrap before Megatron/model imports
    from vime_plugins.models import qwen3_5

    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(123)
    return qwen3_5


def _compare(actual, expected, *, tol=8e-3, cosine=0.999):
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    similarity = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
    assert similarity >= cosine, similarity.item()


def _recurrent(q, k, v, g, beta, boundaries, normalize=True):
    # Independent FP32 delta-rule reference; no patched Megatron/FLA fallback.
    if normalize:
        q = (q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)).to(q.dtype)
        k = (k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)).to(k.dtype)
    q, k, v, g, beta = (tensor.float() for tensor in (q, k, v, g, beta))
    outputs = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        state = q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
        for t in range(start, end):
            state = state * g[:, t].exp()[..., None, None]
            residual = (v[:, t] - (k[:, t, :, :, None] * state).sum(-2)) * beta[:, t, :, None]
            state = state + k[:, t, :, :, None] * residual[..., None, :]
            outputs.append((q[:, t, :, :, None] * state).sum(-2) * q.shape[-1] ** -0.5)
    return torch.stack(outputs, dim=1)


def test_provider_survives_repatch_and_resolves_packaged_opp(runtime):
    from vime.backends.megatron_utils import npu_attention_patch
    from vime.platforms import current_platform

    fn = runtime.get_chunk_gated_delta_rule("fla")
    assert fn.__module__ == "megatron.core.ssm.chunk_gated_delta_rule"
    assert runtime.ShortConvolution is npu_attention_patch.ShortConvolution
    assert runtime.FusedRMSNormGated is npu_attention_patch.FusedRMSNormGated
    current_platform().megatron.repatch(SimpleNamespace())
    assert runtime.get_chunk_gated_delta_rule("fla") is fn
    assert Path(os.environ["FLA_NPU_OP_API_LIB"]).is_file()
    with pytest.raises(ValueError, match="requires backend 'fla'"):
        runtime.get_chunk_gated_delta_rule("flashqla")


def test_packed_convolution_forward_backward_and_boundaries(runtime):
    # The existing NPU convolution requires channels divisible by 256 (35B uses 8192).
    conv = runtime.ShortConvolution(256, 4).to(device="npu", dtype=torch.bfloat16)
    x = torch.randn(1, 128, 256, dtype=torch.bfloat16, device="npu", requires_grad=True)
    boundaries = [0, 48, 128]
    cu = torch.tensor(boundaries, dtype=torch.int32, device="npu")
    out, state = conv(x, cu_seqlens=cu)
    assert state is None
    # Accumulate the reference weight gradient in FP32 across packed samples.
    ref_x = x.detach().float().cpu().requires_grad_()
    ref_w = conv.weight.detach().float().cpu().requires_grad_()
    ref = torch.cat(
        [
            F.silu(F.conv1d(ref_x[:, a:b].float().transpose(1, 2), ref_w.float(), padding=3, groups=256)[..., : b - a])
            .transpose(1, 2)
            .to(x.dtype)
            for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)
        ],
        dim=1,
    )
    _compare(out, ref, tol=5e-3)
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad.cpu())
    _compare(x.grad, ref_x.grad)
    _compare(conv.weight.grad, ref_w.grad, tol=2e-2)
    changed = x.detach().clone()
    changed[:, :48] += 10
    changed_out, _ = conv(changed, cu_seqlens=cu)
    torch.testing.assert_close(out[:, 48:], changed_out[:, 48:], atol=0, rtol=0)
    assert conv.weight.shape == (256, 1, 4)


def test_norm_forward_and_all_gradients(runtime):
    norm = runtime.FusedRMSNormGated(128, dtype=torch.bfloat16, device="npu")
    x, z = [torch.randn(96, 128, device="npu", dtype=torch.bfloat16, requires_grad=True) for _ in range(2)]
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
    ref_x, ref_z, ref_w = [t.detach().float().cpu().requires_grad_() for t in (x, z, norm.weight)]
    ref = F.rms_norm(ref_x, (128,), ref_w, eps=1e-6) * F.silu(ref_z)
    out = norm(x, z)
    _compare(out, ref, tol=2e-2)
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad.float().cpu())
    for actual, expected in ((x.grad, ref_x.grad), (z.grad, ref_z.grad), (norm.weight.grad, ref_w.grad)):
        _compare(actual, expected, tol=2e-2)
    assert list(norm.state_dict()) == ["weight"]


@pytest.mark.parametrize("normalize", [False, True])
def test_packed_gdn_against_recurrent_forward_and_backward(runtime, normalize):
    kernel = runtime.get_chunk_gated_delta_rule("fla")
    shape = (1, 128, 4, 128)
    q, k, v = [torch.randn(shape, device="npu", dtype=torch.bfloat16) for _ in range(3)]
    # Keep the unnormalized recurrence stable. Tiny-norm epsilon is tested separately.
    q, k = q * 0.05, k * 0.05
    g = -torch.rand(shape[:-1], device="npu", dtype=torch.float32)
    beta = torch.rand(shape[:-1], device="npu", dtype=torch.bfloat16)
    inputs = [t.requires_grad_() for t in (q, k, v, g, beta)]
    refs = [t.detach().float().cpu().requires_grad_() for t in inputs]
    boundaries = [0, 48, 128]
    cu = torch.tensor(boundaries, dtype=torch.int32, device="npu")
    out, state = kernel(q, k, v, g=g, beta=beta, cu_seqlens=cu, use_qk_l2norm_in_kernel=normalize)
    assert state is None
    ref = _recurrent(*refs, boundaries, normalize=normalize)
    _compare(out, ref, tol=5e-3)
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad.float().cpu())
    for index, (actual, expected) in enumerate(zip(inputs, refs, strict=True)):
        _compare(actual.grad, expected.grad, tol=2e-2 if index >= 3 else 8e-3, cosine=0.99 if index >= 3 else 0.999)
    # A different first sequence cannot alter the second sequence's recurrent state.
    changed_v = v.detach().clone()
    changed_v[:, :48] += 10
    changed, _ = kernel(q, k, changed_v, g=g, beta=beta, cu_seqlens=cu, use_qk_l2norm_in_kernel=normalize)
    torch.testing.assert_close(out[:, 48:], changed[:, 48:], atol=0, rtol=0)


def test_l2norm_small_norm_formula_and_backward(runtime):
    from megatron.core.ssm.triton.l2norm import l2norm

    # FP32 isolates the epsilon formula/derivative from BF16 saved-y rounding.
    x = torch.full((1, 16, 4, 128), 1e-4, device="npu", requires_grad=True)
    ref_x = x.detach().cpu().requires_grad_()
    out = l2norm(x, eps=1e-6)
    ref = ref_x * torch.rsqrt(ref_x.square().sum(-1, keepdim=True) + 1e-6)
    _compare(out, ref, tol=1e-5)
    out.sum().backward()
    ref.sum().backward()
    _compare(x.grad, ref_x.grad, tol=1e-4)


def test_vime_gdn_parameters_native_roundtrip_and_backward(runtime):
    from vime.backends.megatron_utils.hf_to_megatron.qwen3_5 import qwen3_5_hf_tensor
    from vime.backends.megatron_utils.megatron_to_hf.qwen3_5 import convert_qwen3_5_to_hf

    config = SimpleNamespace(
        hidden_size=32, linear_num_value_heads=32, linear_num_key_heads=16,
        linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
        hidden_act="silu", rms_norm_eps=1e-6, dtype=torch.bfloat16,
    )
    model = runtime.Qwen3_5GatedDeltaNet(config, 0).to(device="npu", dtype=torch.bfloat16)
    expected_shapes = {
        "conv1d.weight": (8192, 1, 4), "norm.weight": (128,),
        "in_proj_qkv.weight": (8192, 32), "in_proj_z.weight": (4096, 32),
        "in_proj_a.weight": (32, 32), "in_proj_b.weight": (32, 32),
        "A_log": (32,), "dt_bias": (32,), "out_proj.weight": (32, 4096),
    }
    assert {name: tuple(t.shape) for name, t in model.state_dict().items()} == expected_shapes
    args = SimpleNamespace(kv_channels=128, hidden_size=32, num_attention_heads=2, num_query_groups=2)
    hf_tensors = {}
    for name, param in model.state_dict().items():
        hf_tensors.update(
            convert_qwen3_5_to_hf(args, f"module.module.decoder.layers.0.self_attention.linear_attn.{name}", param)
        )
    reader = SimpleNamespace(get_tensor=hf_tensors.__getitem__)
    restored = {
        name: qwen3_5_hf_tensor(f"decoder.layers.0.self_attention.linear_attn.{name}", reader, config)
        for name in expected_shapes
    }
    clone = copy.deepcopy(model)
    clone.load_state_dict(restored, strict=True)
    x = torch.randn(1, 128, 32, device="npu", dtype=torch.bfloat16, requires_grad=True)
    cu = torch.tensor([0, 48, 128], device="npu", dtype=torch.int32)
    out = model(x, cu_seqlens=cu)
    torch.testing.assert_close(out, clone(x, cu_seqlens=cu), atol=0, rtol=0)
    out.float().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
