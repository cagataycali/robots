"""Tests for Newton backend lazy-import behaviour.

Verifies that importing ``strands_robots.simulation.newton`` does NOT
trigger import of ``warp`` or ``newton`` (heavy GPU dependencies).
This is critical for keeping ``import strands_robots`` fast and
usable on machines without CUDA.
"""

from __future__ import annotations

import importlib
import sys


class TestLazyImport:
    """Importing the newton sub-package must not load warp or newton."""

    def test_import_newton_package_does_not_load_warp(self) -> None:
        """``import strands_robots.simulation.newton`` must be fast/light."""
        # Clear any cached imports so we get a clean test
        mods_to_clear = [k for k in sys.modules if k.startswith("strands_robots.simulation.newton")]
        for m in mods_to_clear:
            del sys.modules[m]

        # Import the package
        importlib.import_module("strands_robots.simulation.newton")

        # warp and newton must NOT have been imported
        assert "warp" not in sys.modules, "warp was eagerly imported"
        assert "newton" not in sys.modules, "newton was eagerly imported"

    def test_import_config_does_not_load_warp(self) -> None:
        """Config is a pure dataclass — no GPU deps."""
        mods_to_clear = [k for k in sys.modules if k.startswith("strands_robots.simulation.newton")]
        for m in mods_to_clear:
            del sys.modules[m]

        from strands_robots.simulation.newton.config import NewtonConfig  # noqa: F401

        assert "warp" not in sys.modules
        assert "newton" not in sys.modules

    def test_newton_config_available_directly(self) -> None:
        """NewtonConfig must be importable from the package __init__."""
        from strands_robots.simulation.newton import NewtonConfig

        config = NewtonConfig()
        assert config.solver == "mujoco"

    def test_factory_import_does_not_load_warp(self) -> None:
        """create_simulation() import must not trigger Newton deps."""
        mods_to_clear = [k for k in sys.modules if k.startswith("strands_robots.simulation.newton")]
        for m in mods_to_clear:
            del sys.modules[m]

        from strands_robots.simulation import list_backends  # noqa: F401

        # Factory knows about newton but hasn't imported it yet
        assert "warp" not in sys.modules
        assert "newton" not in sys.modules
