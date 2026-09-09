import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from torch import Tensor

try:
    from einops import rearrange
except ImportError:
    rearrange = None


def npu_dot_product_attention_forward(
    self,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor,
    attn_mask_type: AttnMaskType = None,
    attention_bias: Tensor = None,
    packed_seq_params: PackedSeqParams | None = None,
):
    assert attention_bias is None, "Attention bias is not supported for DotProductAttention."

    if packed_seq_params is None:
        n_head = query.shape[2]
    else:
        n_head = query.shape[1]

    sparse_mode = getattr(self.config, "sparse_mode", 0)
    if attn_mask_type == AttnMaskType.no_mask:
        sparse_mode = 0

    scale = self.softmax_scale

    pre_tockens = getattr(self.config, "pre_tockens", 65536)
    next_tockens = getattr(self.config, "next_tockens", 0)

    if packed_seq_params is not None:
        if isinstance(packed_seq_params.cu_seqlens_q, list):
            actual_seq_qlen = packed_seq_params.cu_seqlens_q
            actual_seq_kvlen = packed_seq_params.cu_seqlens_kv
        else:
            actual_seq_qlen = packed_seq_params.cu_seqlens_q.tolist()
            actual_seq_kvlen = packed_seq_params.cu_seqlens_kv.tolist()
        shape_order = "TND"
    else:
        actual_seq_qlen = None
        actual_seq_kvlen = None
        if rearrange is not None:
            query, key, value = [rearrange(x, "s b h d -> s b (h d)") for x in [query, key, value]]
        else:
            query = query.reshape(query.shape[0], query.shape[1], -1)
            key = key.reshape(key.shape[0], key.shape[1], -1)
            value = value.reshape(value.shape[0], value.shape[1], -1)
        shape_order = "SBH"

    output = torch_npu.npu_fusion_attention(
        query,
        key,
        value,
        n_head,
        shape_order,
        pse=None,
        padding_mask=None,
        atten_mask=attention_mask,
        scale=scale,
        pre_tockens=pre_tockens,
        next_tockens=next_tockens,
        keep_prob=1 - self.attention_dropout.p,
        inner_precise=0,
        sparse_mode=sparse_mode,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )[0]

    return output


from megatron.core.transformer.dot_product_attention import DotProductAttention

DotProductAttention.forward = npu_dot_product_attention_forward


# Qwen3.5 training interfaces; keep the public model and saved parameter layout.
def get_chunk_gated_delta_rule(backend: str):
    if backend != "fla":
        raise ValueError(f"Qwen3.5 NPU GDN requires backend 'fla', got {backend!r}")
    # Bind directly to the existing NPU implementation, not Adaptor's dummy FLA namespace.
    from megatron.core.ssm.chunk_gated_delta_rule import chunk_gated_delta_rule

    return chunk_gated_delta_rule


class ShortConvolution(nn.Conv1d):
    """Training-only FLA interface with HF's [channels, 1, kernel] weight."""

    def __init__(self, hidden_size, kernel_size, bias=False):
        super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=bias)

    def forward(self, x, cu_seqlens=None):
        from megatron.core.ssm.triton.causal_conv1d import causal_conv1d

        return causal_conv1d(
            x=x,
            # The NPU kernel uses [kernel, channels]; keep the saved Parameter
            # in HF/FLA's [channels, 1, kernel] layout and transform only its view.
            weight=self.weight.squeeze(1).t().contiguous(),
            bias=self.bias,
            activation="silu",
            cu_seqlens=cu_seqlens,
        )


class FusedRMSNormGated(nn.Module):
    """FLA's norm-before-SiLU-gate semantics, with FP32 intermediates on NPU.

    The interface name is retained; this implementation uses torch autograd,
    not a CUDA fused kernel. The weight is multiplicative, not layernorm-1p.
    """

    def __init__(self, hidden_size, eps=1e-6, activation="silu", device=None, dtype=None):
        super().__init__()
        if activation not in ("silu", "swish"):
            raise ValueError(f"Unsupported NPU GDN norm activation: {activation!r}")
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x, z):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float() * F.silu(z.float())).to(x.dtype)
