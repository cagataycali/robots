"""The installed package tells a type checker that its annotations are real.

PEP 561 says a type checker only reads the inline annotations of an installed
package that carries a ``py.typed`` marker; without it every import from the
package is ``Any``. The package annotates its public surface (``SimEngine``,
``Policy``, ``HardwareDriver``) and checks most of its modules under ``mypy``
with ``disallow_untyped_defs``, yet shipped no marker, so a downstream project
running ``mypy --strict`` saw a subclass of ``SimEngine`` as a subclass of
``Any``: an override with a wrong return type or an incompatible signature went
unreported, and ``disallow_subclassing_any`` refused the subclass outright.

Two things have to hold for the marker to reach an install: it exists as a
resource of the package, and the wheel target does not leave it out.
"""

from __future__ import annotations

import fnmatch
import tomllib
from importlib import resources
from pathlib import Path
from typing import Any

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
MARKER = "py.typed"


class TestTypingMarkerIsAPackageResource:
    """``strands_robots/py.typed`` is present where a type checker looks for it."""

    def test_marker_exists_at_the_package_root(self) -> None:
        marker = resources.files("strands_robots").joinpath(MARKER)
        assert marker.is_file(), "strands_robots ships no py.typed, so type checkers read the package as Any"

    def test_marker_declares_the_whole_package_typed(self) -> None:
        # "partial\n" would declare a stub-only partial package (PEP 561), which
        # this package is not: the annotations are inline.
        content = resources.files("strands_robots").joinpath(MARKER).read_text(encoding="utf-8")
        assert "partial" not in content


class TestWheelTargetKeepsTheMarker:
    """The hatch wheel target packages the marker rather than filtering it out."""

    def _hatch_build(self) -> dict[str, Any]:
        config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        build: dict[str, Any] = config["tool"]["hatch"]["build"]
        return build

    def test_wheel_target_packages_the_package_directory(self) -> None:
        wheel = self._hatch_build()["targets"]["wheel"]
        assert "strands_robots" in wheel["packages"]

    def test_no_exclude_pattern_matches_the_marker(self) -> None:
        """No build or wheel exclude pattern drops the marker.

        No exclude exists today, so this guards a future config change. It is a
        config-level proxy for building a wheel and listing its contents.
        """
        build = self._hatch_build()
        wheel = build["targets"]["wheel"]
        patterns = [*build.get("exclude", []), *wheel.get("exclude", [])]
        path = f"strands_robots/{MARKER}"
        matching = [p for p in patterns if fnmatch.fnmatch(path, p.lstrip("/")) or fnmatch.fnmatch(MARKER, p)]
        assert not matching, f"the wheel exclude patterns {matching} drop {path}"
