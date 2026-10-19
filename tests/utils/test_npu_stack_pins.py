"""A new patch series must not be applied to a different vendor source base."""

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("_check_npu_stack", ROOT / "tools/check_npu_stack.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


@pytest.mark.unit
def test_manifest_validates_real_git_head_without_importing_runtime(tmp_path):
    repo = tmp_path / "vendor"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "--no-verify",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    manifest = tmp_path / "versions.json"

    def write(commit):
        manifest.write_text(json.dumps({"sources": {"vendor": {"path": "/vendor", "commit": commit}}}))

    write(head)
    checker.check_stack(manifest, tmp_path)
    write("0" * 40)
    with pytest.raises(RuntimeError, match="rebuild the image"):
        checker.check_stack(manifest, tmp_path)
    write("moving-branch")
    with pytest.raises(ValueError, match="immutable"):
        checker.check_stack(manifest, tmp_path)


@pytest.mark.unit
def test_runtime_reconciliation_checks_pins_before_reverting(tmp_path):
    source = tmp_path / "source"
    state = tmp_path / "state"
    (source / "tools").mkdir(parents=True)
    (source / "docker/npu_patch").mkdir(parents=True)
    state.mkdir()
    shutil.copy(ROOT / "tools/check_npu_stack.py", source / "tools/check_npu_stack.py")
    (source / "docker/npu_patch/versions.json").write_text(
        json.dumps({"sources": {"bad": {"path": str(tmp_path / "missing"), "commit": "0" * 40}}})
    )
    old = state / "series.conf"
    old.write_text("must not be interpreted as a patch series\n")
    env = {
        **os.environ,
        "BUILDKITE_COMMIT": "",
        "VIME_DIR": str(source),
        "VIME_NPU_PATCH_SOURCE_ROOT": str(source),
        "VIME_NPU_PATCH_STATE_DIR": str(state),
    }
    result = subprocess.run(
        ["bash", str(ROOT / ".buildkite/scripts/update-npu-environment.sh")], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "rebuild the image" in result.stderr
    assert "Reverting" not in result.stdout
    assert old.read_text() == "must not be interpreted as a patch series\n"
