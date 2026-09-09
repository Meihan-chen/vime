"""CPU contracts for the S7 checkpoint test modes and patch ordering."""

import ast
import importlib.util
import shlex
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def qwen30(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    (tmp_path / "models").mkdir()
    spec = importlib.util.spec_from_file_location("qwen30_npu_case", REPO / "tests/test_qwen3_30B_A3B_npu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_30b_keeps_hf_path(qwen30, monkeypatch):
    commands = []
    launches = []
    monkeypatch.setattr(qwen30.U, "exec_command", commands.append)
    monkeypatch.setattr(qwen30.U, "execute_train", lambda **kwargs: launches.append(kwargs))
    assert qwen30.prepare() is None
    qwen30.execute()
    assert not any("torch.distributed.run" in cmd or "rm -rf" in cmd for cmd in commands)
    args = launches[0]["train_args"]
    assert f"--ref-load {shlex.quote(qwen30.MODEL_DIR)} " in args
    assert "--colocate " in args
    assert "--tensor-model-parallel-size 4 " in args
    assert "--expert-model-parallel-size 8 " in args
    assert "weight_nz_mode" in args


def test_qwen35_native_paths_parallelism_and_packaged_opp(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    spec = importlib.util.spec_from_file_location("qwen35_npu_case", REPO / "tests/test_qwen3.5_35B_A3B_npu.py")
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    launches = []
    monkeypatch.setattr(case.U, "execute_train", lambda **kwargs: launches.append(kwargs))
    opp_env = {"ASCEND_CUSTOM_OPP_PATH": "/installed/fla:/other/vendor", "FLA_NPU_OPP_PATH": "/installed/fla"}
    monkeypatch.setattr(case, "get_fla_npu_runtime_env", lambda: opp_env)
    case.execute()
    assert case.MODEL_DIR == f"{tmp_path}/models/Qwen/Qwen3.5-35B-A3B"
    assert case.DATASET_DIR == f"{tmp_path}/datasets/dapo-math-17k"
    launch = launches[0]
    args = launch["train_args"]
    for flag in ("hf-checkpoint", "load", "ref-load"):
        assert f"--{flag} {shlex.quote(case.MODEL_DIR)} " in args
    for flag in (
        "--tensor-model-parallel-size 2 ", "--sequence-parallel ",
        "--expert-model-parallel-size 8 ", "--expert-tensor-parallel-size 1 ",
        "--actor-num-gpus-per-node 8 ", "--rollout-num-gpus 8 ",
        "--rollout-num-gpus-per-engine 2 ", "--num-rollout 2 ",
    ):
        assert flag in args
    assert "--colocate" not in args
    assert "bridge" not in args
    assert launch["num_gpus_per_node"] == 16
    assert launch["extra_env_vars"]["ASCEND_CUSTOM_OPP_PATH"] == "/installed/fla:/other/vendor"
    assert launch["extra_env_vars"]["FLA_NPU_OPP_PATH"] == "/installed/fla"
    script = (REPO / "scripts/run-qwen3.5-35B-A3B-npu.sh").read_text()
    assert "opp/vendors/fla_npu_transformer" not in script


def test_torch_dist_mode_uses_new_output_and_ref_load(qwen30, monkeypatch, tmp_path):
    commands = []
    launches = []
    existing = tmp_path / "models/Qwen3-30B-A3B_torch_dist"
    existing.mkdir()
    sentinel = existing / "keep"
    sentinel.write_text("existing checkpoint")

    def execute(command):
        commands.append(command)
        if "torch.distributed.run" in command:
            tokens = shlex.split(command)
            target = Path(tokens[tokens.index("--save") + 1])
            (target / "latest_checkpointed_iteration.txt").write_text("release")
            (target / ".metadata").write_bytes(b"test fixture")

    monkeypatch.setattr(qwen30.U, "exec_command", execute)
    monkeypatch.setattr(qwen30.U, "execute_train", lambda **kwargs: launches.append(kwargs))
    checkpoint = qwen30.prepare(torch_dist_ref_load=True)
    qwen30.execute(checkpoint)
    assert Path(checkpoint) != existing
    assert sentinel.read_text() == "existing checkpoint"
    assert not any("rm -rf" in command for command in commands)
    conversion = next(command for command in commands if "torch.distributed.run" in command)
    assert "VIME_PLATFORM=npu" in conversion
    assert "--nproc-per-node 8 " in conversion
    args = launches[0]["train_args"]
    assert f"--ref-load {shlex.quote(checkpoint)} " in args
    assert "--load " not in args
    assert "--colocate " in args
    assert "weight_nz_mode" in args


def test_converter_bootstraps_before_first_megatron_import():
    tree = ast.parse((REPO / "tools/convert_hf_to_torch_dist.py").read_text())
    first_megatron = next(
        node.lineno
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("megatron.")
    )
    bootstrap = next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        and any(alias.name == "vime.backends.megatron_utils" for alias in node.names)
    )
    assert bootstrap < first_megatron
    assert "vime.utils.common" not in ast.unparse(tree)


def test_common_megatron_patch_is_snapshotted_before_npu_patch():
    entries = [
        line.split("|")
        for line in (REPO / "docker/npu_patch/series.conf").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    megatron = [entry for entry in entries if entry[0] == "/root/Megatron-LM"]
    assert megatron == [
        ["/root/Megatron-LM", "megatron-common.patch", "docker/patch/latest/megatron.patch"],
        ["/root/Megatron-LM", "megatron.patch", "docker/npu_patch/megatron.patch"],
    ]
    dockerfile = (REPO / "docker/Dockerfile.npu").read_text()
    assert "COPY docker/patch/latest/megatron.patch /opt/npu_patch/megatron-common.patch" in dockerfile
    assert "/opt/vime_patch/megatron.patch" not in dockerfile


def _megatron_patch_additions(path, patch_path="docker/npu_patch/megatron.patch"):
    patch = (REPO / patch_path).read_text()
    section = patch.split(f"diff --git a/{path} b/{path}\n", 1)[1].split("diff --git ", 1)[0]
    return "\n".join(line[1:] for line in section.splitlines() if line.startswith("+") and not line.startswith("+++"))


@pytest.mark.parametrize("normalize", [False, True])
def test_gdn_calls_local_l2norm_signature(normalize):
    norm_calls, chunk_calls = [], []

    def norm_apply(x, eps, output_dtype):
        norm_calls.append((x, eps, output_dtype))
        return x

    def chunk_apply(*args):
        chunk_calls.append(args)
        return "output", "state"

    namespace = {
        "torch": SimpleNamespace(float32="float32"),
        "L2NormFunction": SimpleNamespace(apply=norm_apply),
        "ChunkGatedDeltaRuleFunction": SimpleNamespace(apply=chunk_apply),
    }
    for path, name in (
        ("megatron/core/ssm/triton/l2norm.py", "l2norm"),
        ("megatron/core/ssm/chunk_gated_delta_rule.py", "chunk_gated_delta_rule"),
    ):
        tree = ast.parse(_megatron_patch_additions(path))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        function.decorator_list = []
        exec("from __future__ import annotations\n" + ast.unparse(function), namespace)

    q, k, v = (SimpleNamespace(shape=(1, 8, 2, 16), dtype="bfloat16") for _ in range(3))
    beta = SimpleNamespace(shape=(1, 8, 2))
    result = namespace["chunk_gated_delta_rule"](q, k, v, None, beta, use_qk_l2norm_in_kernel=normalize)
    assert result == ("output", "state")
    assert norm_calls == ([(q, 1e-6, None), (k, 1e-6, None)] if normalize else [])
    assert len(chunk_calls) == 1
    assert chunk_calls[0][:6] == (q, k, v, None, beta, 0.25)
    assert chunk_calls[0][9] is normalize


def test_npu_patch_keeps_public_transformer_layer():
    patch = (REPO / "docker/npu_patch/megatron.patch").read_text()
    assert "diff --git a/megatron/core/transformer/transformer_layer.py " not in patch


def test_post_layernorm_flags_remain_dataclass_generated():
    additions = _megatron_patch_additions(
        "megatron/core/transformer/transformer_config.py", "docker/patch/latest/megatron.patch"
    )
    fields = ast.parse(textwrap.dedent(additions)).body
    defaults = {node.target.id: ast.literal_eval(node.value) for node in fields if isinstance(node, ast.AnnAssign)}
    npu_patch = (REPO / "docker/npu_patch/megatron.patch").read_text()
    for name in ("post_self_attn_layernorm", "post_mlp_layernorm"):
        assert defaults[name] is False
        assert f"--{name.replace('_', '-')}" not in npu_patch
