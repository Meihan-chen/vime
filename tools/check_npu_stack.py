"""Check source pins before applying NPU patches; requires only Python and Git."""

import argparse
import json
import re
import subprocess
from pathlib import Path


def check_stack(manifest: Path, root: Path = Path("/"), only: list[str] | None = None) -> None:
    sources = json.loads(manifest.read_text())["sources"]
    names = list(sources) if only is None else only
    errors = []
    for name in names:
        if name not in sources:
            raise ValueError(f"Unknown stack component: {name}")
        spec = sources[name]
        expected = spec["commit"]
        if re.fullmatch(r"[0-9a-f]{40}", expected) is None:
            raise ValueError(f"{name}: manifest needs a full immutable Git SHA")
        path = root / spec["path"].lstrip("/")
        result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
        actual = result.stdout.strip() if result.returncode == 0 else "missing Git checkout"
        if actual != expected:
            errors.append(f"{name}: expected {expected}, found {actual} at {path}")
    if errors:
        raise RuntimeError("NPU stack does not match this patch series; rebuild the image.\n" + "\n".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "docker/npu_patch/versions.json"
    )
    parser.add_argument(
        "--root", type=Path, default=Path("/"), help="Prefix for checking an isolated filesystem snapshot"
    )
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    check_stack(args.manifest, args.root, args.only)
    print("NPU source pins match")


if __name__ == "__main__":
    main()
