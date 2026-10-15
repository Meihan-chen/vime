"""The fc1 shard-dimension correction must not touch a layout that is right.

It exists for grouped-MoE implementations that report dim 1 for weights they
store [out, in]. Transformer Engine reports 0 for the same weights, so a
reported 0 has to pass through untouched on every platform.
"""

from __future__ import annotations

import pytest

FC1_NAMES = (
    "decoder.layers.0.mlp.linear_fc1.weight",
    "decoder.layers.0.mlp.linear_fc1.bias",
)

OTHER_NAMES = (
    "decoder.layers.0.mlp.linear_fc2.weight",
    "decoder.layers.0.self_attention.linear_qkv.weight",
)


def _normalizers():
    # Both modules import megatron at module level, and other tests in the same
    # session replace it with a stub, so this cannot be a module-level import.
    try:
        from vime.backends.megatron_utils.hf_to_megatron.common import _normalized_partition_dim
        from vime.backends.megatron_utils.update_weight.common import _normalized_fc1_partition_dim
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on session order
        pytest.skip(f"megatron is shadowed by a stub in this session: {exc}")

    return _normalized_partition_dim, _normalized_fc1_partition_dim


@pytest.mark.unit
@pytest.mark.parametrize("name", FC1_NAMES)
def test_a_correctly_reported_dimension_is_left_alone(name):
    by_name, by_dim = _normalizers()

    assert by_name(name, 0) == 0
    assert by_dim(0) == 0


@pytest.mark.unit
@pytest.mark.parametrize("name", FC1_NAMES)
def test_a_contradictory_dimension_is_corrected(name):
    by_name, by_dim = _normalizers()

    assert by_name(name, 1) == 0
    assert by_dim(1) == 0


@pytest.mark.unit
@pytest.mark.parametrize("name", OTHER_NAMES)
def test_other_parameters_are_untouched(name):
    by_name, _ = _normalizers()

    for partition_dim in (0, 1):
        assert by_name(name, partition_dim) == partition_dim
