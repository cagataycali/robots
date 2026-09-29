"""The install caveats say which Isaac Sim release they apply to.

Measured from the installed wheels' metadata (``isaacsim-kernel`` requires):

=========  ===================  ==============  =============
release    coverage             numpy           torch
=========  ===================  ==============  =============
6.0.1.0    ``==7.4.4``          ``==2.3.1``     ``==2.11.0``
6.1.0.0    ``>=7.4.4,<8``       ``==2.3.1``     ``==2.11.0``
=========  ===================  ==============  =============

The caveat named only coverage - which 6.1 no longer downgrades - and not the
numpy / torch exact pins both releases carry, which move an existing
environment's numpy (2.5.3 -> 2.3.1 measured). And the docs pinned ``6.0.*`` with
no word that 6.1.0.0, the newest wheel on pypi.nvidia.com, is also verified
(with the converter Physics-variant fix: add_robot and the GPU integration suite pass on an L40S).
"""

from __future__ import annotations

import pathlib

from strands_robots.simulation.isaac import _install

_DOC = pathlib.Path(__file__).resolve().parents[3] / "docs" / "learn" / "simulation" / "isaac.md"


def test_the_coverage_caveat_is_scoped_to_60() -> None:
    caveats = _install.ISAAC_SIM_PIP_CAVEATS
    assert "coverage" in caveats
    assert "6.0" in caveats.split("coverage")[0][-80:] + caveats.split("coverage")[1][:80]


def test_the_numpy_and_torch_pins_are_named() -> None:
    caveats = _install.ISAAC_SIM_PIP_CAVEATS
    assert "numpy==2.3.1" in caveats
    assert "torch==2.11.0" in caveats


def test_the_verified_releases_are_declared_and_documented() -> None:
    assert "6.0.1.0" in _install.ISAAC_SIM_VERIFIED_PIP_VERSIONS
    assert "6.1.0.0" in _install.ISAAC_SIM_VERIFIED_PIP_VERSIONS
    doc = _DOC.read_text(encoding="utf-8")
    for version in _install.ISAAC_SIM_VERIFIED_PIP_VERSIONS:
        assert version in doc
