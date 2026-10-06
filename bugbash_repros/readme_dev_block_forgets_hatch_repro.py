"""Repro: README.md 'Development' block omits `hatch` from the install step.

README.md L115-L118 (upstream main @ 724de8a31, 2026-10-06):

    uv venv --python 3.12 && source .venv/bin/activate
    uv pip install -e ".[all,dev]"
    hatch run test && hatch run lint   # pytest; ruff + mypy

A new contributor who follows the block literally hits
`FileNotFoundError: 'hatch'` because `hatch` (the task runner, PyPI
`hatch`) is not in `[dev]`, not in `[all]`, not in `[project.dependencies]`,
not in `[build-system.requires]` (that line declares `hatchling`, the
build backend — a different package).

`.github/workflows/test-lint.yml:204` runs
`pip install --no-cache-dir hatch` BEFORE `hatch run test` in CI, so the
upstream path acknowledges the dependency — the README just forgets to.

Fix on branch is one line in README.md: add
`uv pip install hatch` between the `-e ".[all,dev]"` install and the
`hatch run test` invocation.

This repro simulates the README path in a scratch venv and shows the
exact `FileNotFoundError` a new contributor sees, without needing a full
strands-robots install (`[all]` would try to pull torch, CUDA toolchains,
etc. — the point is that even the subset `[dev]` on its own doesn't give
you `hatch`).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
README = REPO_ROOT / "README.md"


def _collect_transitive_extras(opt: dict[str, list[str]], root: str) -> list[str]:
    """Flatten extras of the form `strands-robots[X]` recursively."""
    seen: set[str] = set()
    out: list[str] = []

    def walk(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        for dep in opt.get(name, []):
            if dep.startswith("strands-robots["):
                inner = dep.split("[", 1)[1].rstrip("]")
                for sub in inner.split(","):
                    walk(sub.strip())
            else:
                out.append(dep)

    walk(root)
    return out


def check_hatch_declared() -> None:
    with PYPROJECT.open("rb") as f:
        data = tomllib.load(f)

    base = data["project"]["dependencies"]
    opt = data["project"]["optional-dependencies"]
    build_req = data.get("build-system", {}).get("requires", [])

    all_deps = list(base)
    for e in ("all", "dev"):
        all_deps.extend(_collect_transitive_extras(opt, e))

    hatch_hits = [d for d in all_deps if d.strip().lower().startswith("hatch")
                  and not d.strip().lower().startswith("hatchling")
                  and not d.strip().lower().startswith("hatch-vcs")
                  and not d.strip().lower().startswith("hatch_vcs")]

    print("--- pyproject.toml survey ---")
    print(f"base deps:            {len(base)}")
    print(f"[all,dev] transitive: {len(all_deps)}")
    print(f"build-system.requires: {build_req}")
    print(f"deps matching 'hatch' (excluding hatchling/hatch-vcs): {hatch_hits}")
    assert hatch_hits == [], (
        "If this assertion ever fails, the README's dev block will have become "
        "self-consistent and this defect will have been fixed. Update or remove."
    )
    print("CONFIRMED: `hatch` is NOT declared in [all,dev] -> following README L118 "
          "with the current project will not install the hatch task runner.")


def check_readme_line() -> None:
    text = README.read_text(encoding="utf-8")
    lines = text.splitlines()
    hits = [(i + 1, l) for i, l in enumerate(lines) if "hatch run test" in l]
    print()
    print("--- README.md survey ---")
    for ln, body in hits:
        print(f"L{ln}: {body!r}")
    # Fix on the branch should name an explicit `uv pip install hatch` step
    install_hint = any("install hatch" in l for l in lines)
    print(f"README names an explicit `install hatch` step: {install_hint}")


def repro_end_to_end() -> None:
    """Create a scratch venv, install the dev extras, show `hatch` is not on PATH."""
    print()
    print("--- end-to-end: scratch venv ---")
    env = {k: v for k, v in os.environ.items() if k != "SYSTEM_PROMPT"}
    tmp = Path(tempfile.mkdtemp(prefix="readme_hatch_repro_"))
    try:
        venv_dir = tmp / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)],
                       check=True, env=env, capture_output=True, text=True, timeout=120)
        venv_bin = venv_dir / "bin"
        pip = venv_bin / "pip"

        # Install the exact dev extras from pyproject.toml [dev]
        dev_deps = [
            "pytest>=6.0,<10.0.0",
            "pytest-cov>=4.0.0,<6.0.0",
            "ruff>=0.15.12,<0.16.0",
            "mypy>=1.0.0,<2.0.0",
            "pytest-timeout>=2.0.0,<3.0.0",
            "pytest-xdist>=3.0.0,<4.0.0",
        ]
        subprocess.run([str(pip), "install", "--quiet", *dev_deps],
                       check=True, env=env, capture_output=True, text=True, timeout=300)

        hatch_path = venv_bin / "hatch"
        print(f"after `pip install [dev extras]`, `{hatch_path}` exists? {hatch_path.exists()}")

        try:
            subprocess.run([str(hatch_path), "run", "test"],
                           check=True, env=env, capture_output=True, text=True, timeout=10)
        except FileNotFoundError as e:
            print(f"EXACT ERROR a new contributor sees:  {type(e).__name__}: {e}")
            return

        raise AssertionError("UNEXPECTED: `hatch` was callable from the dev-only venv.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    check_hatch_declared()
    check_readme_line()
    repro_end_to_end()
    print()
    print("Defect confirmed: README.md 'Development' block names `hatch run test` "
          "without naming `hatch` as a prerequisite install, and `hatch` is in no "
          "[dev]/[all] extra or base dependency. Fix on branch adds the missing "
          "`uv pip install hatch` line, matching .github/workflows/test-lint.yml:204.")
