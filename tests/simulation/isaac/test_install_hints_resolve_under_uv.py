"""The Isaac Sim install line has a uv spelling that resolves.

The repository's own tooling is uv (``uv.lock``, ``[tool.uv]``, hatch's
``installer = "uv"``), and the documented pip line does not resolve under it:

* without ``--index-strategy unsafe-best-match`` uv takes each package from the
  FIRST index that has it, so the ``isaacsim-*`` names PyPI also carries resolve
  against PyPI stubs (``no version of mujoco-usd-converter==0.1.0``);
* without ``--prerelease=allow`` it refuses ``isaacsim-core``'s exact pin
  ``tinyobjloader==2.0.0rc13`` (``if-necessary`` does not cover a transitive
  exact pin either).

Measured with ``uv 0.11 pip install --dry-run 'isaacsim[all,extscache]==6.0.*'``
on Python 3.12: only the line with both flags resolves (isaacsim 6.0.1.0, 166
packages). pip itself accepts an exact-pinned prerelease, so the pip line stays.
"""

from __future__ import annotations

import pathlib

from strands_robots.simulation.isaac import _install


def test_a_uv_install_line_is_declared_with_both_flags() -> None:
    line = _install.ISAAC_SIM_UV_INSTALL
    assert line.startswith("uv pip install 'isaacsim[all,extscache]")
    assert "--index-strategy unsafe-best-match" in line
    assert "--prerelease=allow" in line
    assert "--extra-index-url https://pypi.nvidia.com" in line


def test_the_uv_line_pins_the_same_release_as_the_pip_line() -> None:
    spec = _install.ISAAC_SIM_PIP_INSTALL.split("'")[1]
    assert f"'{spec}'" in _install.ISAAC_SIM_UV_INSTALL


def test_is_available_names_the_uv_route() -> None:
    assert _install.ISAAC_SIM_UV_INSTALL in _install.not_importable_reason()


def test_the_docs_carry_the_uv_line() -> None:
    doc = pathlib.Path(__file__).resolve().parents[3] / "docs" / "learn" / "simulation" / "isaac.md"
    assert _install.ISAAC_SIM_UV_INSTALL in doc.read_text(encoding="utf-8")
