"""CPU regression tests of the shipped TE-NPU patch, without applying it."""

import ast
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

NUM_GPUS = 0


@pytest.fixture(scope="module")
def local_cu_seqlens():
    # Extract the added, self-contained helper. Never import or modify the
    # installed TE-NPU dependency; this also runs in CPU CI without torch-npu.
    patch = (Path(__file__).resolve().parents[1] / "docker/npu_patch/transformer-engine-npu.patch").read_text()
    lines = patch.splitlines()
    start = lines.index("+def _local_thd_cu_seqlens(cu_seqlens, local_tokens, cp_size):")
    added = []
    for line in lines[start:]:
        if not line.startswith("+"):
            break
        added.append(line[1:])
    namespace = {"torch": torch}
    exec(compile(ast.parse("\n".join(added)), "<te-npu-cp-patch>", "exec"), namespace)
    return namespace["_local_thd_cu_seqlens"]


@pytest.mark.unit
@pytest.mark.parametrize(("local_tokens", "cp_size"), [(896, 2), (2816, 2), (8192, 4)])
def test_build_384_global_lengths_match_local_tensor(local_cu_seqlens, local_tokens, cp_size):
    assert local_cu_seqlens([0, local_tokens * cp_size], local_tokens, cp_size) == [0, local_tokens]


@pytest.mark.unit
@pytest.mark.parametrize("as_tensor", [False, True])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_packed_sequence_boundaries_and_half_chunks(local_cu_seqlens, cp_size, as_tensor):
    # Three sequences of different lengths, including a final padding segment.
    lengths = [0, 4 * cp_size, 12 * cp_size, 16 * cp_size]
    source = torch.tensor(lengths, dtype=torch.int32) if as_tensor else lengths.copy()
    local = local_cu_seqlens(source, 16, cp_size)
    assert local == [0, 4, 12, 16]
    # Both halves selected by causal ring attention have the expected T length.
    assert [x // 2 for x in local] == [0, 2, 6, 8]
    assert (source.tolist() if as_tensor else source) == lengths


@pytest.mark.unit
@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_already_local_metadata_is_not_divided_again(local_cu_seqlens, cp_size):
    assert local_cu_seqlens([0, 4, 12, 16], 16, cp_size) == [0, 4, 12, 16]


@pytest.mark.unit
@pytest.mark.parametrize(("cp_size", "qkv_format"), [(2, "thd"), (4, "thd"), (2, "sbhd")])
def test_ring_forward_uses_local_lengths_for_indices_and_saved_config(local_cu_seqlens, cp_size, qkv_format):
    patch = (Path(__file__).resolve().parents[1] / "docker/npu_patch/transformer-engine-npu.patch").read_text()
    # Execute the forward hunk's new-side statements only, not the dependency.
    # This verifies the helper is wired into both indexing and the lengths saved
    # in cp_config for forward/backward, while leaving SBHD behavior unchanged.
    hunk = patch.rsplit("@@", 1)[1].splitlines()[1:]
    source = textwrap.dedent("\n".join(line[1:] for line in hunk if line.startswith((" ", "+"))))
    index_inputs = []

    def compute_qkv_index(lengths):
        index_inputs.append(lengths)
        return "query_indices", "kv_indices"

    config = SimpleNamespace(is_eod_reset=True, cp_size=cp_size)
    namespace = {
        "cp_config": config,
        "qkv_format": qkv_format,
        "cu_seqlens_q": [0, 4 * cp_size, 16 * cp_size],
        "cu_seqlens_kv": [0, 4 * cp_size, 16 * cp_size],
        "max_seqlen_q": 12 * cp_size,
        "max_seqlen_kv": 12 * cp_size,
        "q": torch.empty(16, 2, 4),
        "k": torch.empty(16, 2, 4),
        "_local_thd_cu_seqlens": local_cu_seqlens,
        "compute_qkv_index": compute_qkv_index,
    }
    exec(compile(ast.parse(source), "<te-npu-ring-forward-hunk>", "exec"), namespace)
    expected = [0, 4, 16] if qkv_format == "thd" else [0, 4 * cp_size, 16 * cp_size]
    assert config.actual_seq_qlen == expected
    assert config.actual_seq_kvlen == expected
    assert index_inputs == ([tuple(expected)] if qkv_format == "thd" else [])


@pytest.mark.unit
@pytest.mark.parametrize("lengths", [[], [2, 8], [0, 12, 8], [0, 3, 8], [0, 2, 8], [0, 10]])
def test_incompatible_unpadded_metadata_is_rejected(local_cu_seqlens, lengths):
    with pytest.raises(ValueError, match="THD"):
        local_cu_seqlens(lengths, 4, 2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
