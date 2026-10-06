"""Repro: README hero's `from strands import Agent` + PyPI `strands` squatter.

A new user reads README.md:50-57 and sees:

    from strands import Agent
    from strands_robots import Robot

A copy-paste reader who has not installed anything yet runs the script,
sees `ModuleNotFoundError: No module named 'strands'`, and does the
natural thing: `pip install strands`.

That installs an unrelated 2020 C++ Schrödinger-eigenvalue solver by
Toon Baeyens (https://github.com/twist-numerical/strands) --- metadata
below -- not the Strands Agent SDK whose distribution name is
`strands-agents` (import name `strands`). Build usually fails on hosts
without CMake+GSL+Eigen, so the signal to the user is "something is
wrong with strands' install" rather than "wrong package name on PyPI."

The project already carries an analogous warning for the identically-
shaped `nvidia-curobo` squatter at pyproject.toml:168-181, but no such
warning for `strands` appears in README, pyproject or docs.

Run:
    python bugbash_repros/readme_strands_pypi_squatter_repro.py

Expected after fix:
    README install section (or a short "On PyPI" note) names the
    distribution `strands-agents` and warns that `pip install strands`
    installs an unrelated package.
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path


def pypi_metadata(name: str) -> dict:
    with urllib.request.urlopen(
        f"https://pypi.org/pypi/{name}/json", timeout=20
    ) as f:
        return json.loads(f.read())["info"]


def main() -> int:
    rc = 0
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()

    # Readers reach the hero before the install section.
    hero_line = re.search(r"^from strands import Agent\s*$", readme, re.M)
    install_pos = readme.find("## Install")
    assert hero_line, "README hero line not found"
    assert install_pos > hero_line.end(), (
        "README install section is after the hero import, so a new "
        "user reading top-down sees `from strands import Agent` first."
    )

    strands = pypi_metadata("strands")
    strands_agents = pypi_metadata("strands-agents")

    print("README hero line (char offset", hero_line.start(), "):", hero_line.group(0))
    print(f"PyPI `strands` top version:         {strands['version']} by {strands['author']!r}")
    print(f"   summary:                         {strands['summary'][:90]}")
    print(f"   homepage:                        {strands['home_page']}")
    print(f"PyPI `strands-agents` top version:  {strands_agents['version']}")
    print(f"   homepage:                        {strands_agents.get('home_page') or strands_agents.get('project_url')}")

    # Distribution name mismatch is the trap.
    unrelated = "schr" in strands["summary"].lower() or "wave" in strands["summary"].lower()
    assert unrelated, (
        "PyPI `strands` 0.1.0 is a Schrödinger-equation solver, not an "
        "agent SDK -- the squatter trap exercised by the README hero."
    )

    squatter_cite = (
        Path(__file__).resolve().parents[1] / "pyproject.toml"
    ).read_text()
    assert "nvidia-curobo" in squatter_cite and "squatter" in squatter_cite, (
        "pyproject already documents the sibling nvidia-curobo squatter pattern"
    )
    pyproject_mentions_strands_squatter = re.search(
        r"pip install strands(?![-_\w])", squatter_cite
    ) is not None
    readme_mentions_strands_squatter = (
        "pypi.org/project/strands/" in readme
        or re.search(r"`strands`[^-_]", readme) is not None
    )
    if not pyproject_mentions_strands_squatter and not readme_mentions_strands_squatter:
        print()
        print("DEFECT: README's hero teaches `from strands import Agent` and the")
        print("Install section (L64-69) names the project PyPI distribution as")
        print("`strands-robots[sim-mujoco]`, but neither README nor pyproject.toml")
        print("warns that `pip install strands` installs an unrelated squatter.")
        print("The analogous nvidia-curobo warning lives at pyproject.toml:168-181.")
        rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
