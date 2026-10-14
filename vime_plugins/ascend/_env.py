"""CANN search paths shared by the Ascend registrations."""

from __future__ import annotations

import os


def cann_site_packages() -> str | None:
    candidates: list[str] = []
    for env_key in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH"):
        base = os.environ.get(env_key)
        if not base:
            continue
        candidates.extend(
            [
                os.path.join(base, "python", "site-packages"),
                os.path.normpath(os.path.join(base, "..", "python", "site-packages")),
            ]
        )
    candidates.append("/usr/local/Ascend/ascend-toolkit/latest/python/site-packages")
    for path in candidates:
        if os.path.isdir(os.path.join(path, "acl")):
            return path
    return None


def prepend_pythonpath(env: dict[str, str], *paths: str) -> None:
    existing = env.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    existing_parts = {part for part in existing.split(os.pathsep) if part}
    prefix_parts = [path for path in paths if path and path not in existing_parts]
    if prefix_parts:
        env["PYTHONPATH"] = os.pathsep.join([*prefix_parts, existing] if existing else prefix_parts)
