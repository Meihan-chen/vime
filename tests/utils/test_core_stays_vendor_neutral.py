"""The core tree must not name a specific accelerator vendor in its code.

Vendor-specific values belong in an accelerator backend or in vime_plugins/,
reached through a seam that defaults to the behaviour the GPU path already has.
Grepping for one module name does not catch this: a second platform registry
under a different name passes that check while carrying vendor paths inline.

Comments and docstrings are not checked. Naming a vendor to explain why a
generic mechanism exists is how the surrounding code already documents itself.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# The accelerator package carries the backends themselves, as it already does
# for the others, so it is the one place in the core tree allowed to name them.
EXEMPT_TREES = (Path("vime/utils/accelerator"),)

# Enumerating every accelerator Ray supports is not a preference for one of
# them; this list mirrors Ray's own and predates any of this.
EXEMPT_FILES = {
    Path("vime/ray/utils.py"): "exhaustive list of Ray's noset variables",
}

VENDOR_WORDS = frozenset({"ascend", "cann", "hccl", "npu", "musa", "supa", "maca"})


def _vendor_word(text: str) -> str | None:
    """Match whole words, so an identifier like ASCEND_RT_VISIBLE_DEVICES is
    caught while an ordinary word such as "Cannot" is not."""
    for word in re.findall(r"[A-Za-z]+", text):
        if word.lower() in VENDOR_WORDS:
            return word
    return None


def _code_strings_and_names(source: str, *, binding_file=False):
    """Yield (lineno, text) for code, skipping comments and docstrings."""
    tree = ast.parse(source)
    if binding_file:
        tree.body = [
            node
            for node in tree.body
            if not (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id in {"_DEFAULT_MODULES", "_DEFAULT_COMPATIBLE"}
                    for target in node.targets
                )
            )
        ]
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(first := node.body[0], ast.Expr):
                if isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                    docstrings.add(id(first.value))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            yield node.lineno, node.value
        elif isinstance(node, ast.Name):
            yield node.lineno, node.id
        elif isinstance(node, ast.Attribute):
            yield node.lineno, node.attr
        elif isinstance(node, ast.alias):
            yield getattr(node, "lineno", 0), node.name + " " + (node.asname or "")


def _core_sources() -> list[Path]:
    sources = []
    for path in sorted((REPO_ROOT / "vime").rglob("*.py")):
        relative = path.relative_to(REPO_ROOT)
        if any(relative.is_relative_to(tree) for tree in EXEMPT_TREES):
            continue
        if relative in EXEMPT_FILES:
            continue
        sources.append(path)
    return sources


@pytest.mark.unit
def test_core_tree_code_names_no_vendor():
    offenders: list[str] = []
    for path in _core_sources():
        for lineno, text in _code_strings_and_names(
            path.read_text(encoding="utf-8"), binding_file=path.relative_to(REPO_ROOT) == Path("vime/utils/support.py")
        ):
            if word := _vendor_word(text):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {word!r} in {text!r}")

    assert not offenders, "vendor-specific tokens in core code:\n" + "\n".join(offenders)


@pytest.mark.unit
def test_the_gate_catches_an_inline_vendor_value(tmp_path):
    # Without this, a gate that silently matches nothing would look like a pass.
    planted = "PATHS = {'npu': '/usr/local/Ascend/ascend-toolkit/latest'}\n"

    found = [text for _, text in _code_strings_and_names(planted) if _vendor_word(text)]

    assert found
