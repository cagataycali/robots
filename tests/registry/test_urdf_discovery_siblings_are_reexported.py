"""The URDF-side discovery siblings are re-exported from
``strands_robots.registry`` alongside their MJCF counterparts.

Four sibling pairs live in ``strands_robots.registry.discovery``:

    - is_discoverable             <-> is_urdf_discoverable
    - list_discoverable           <-> list_urdf_discoverable
    - discover_robot              <-> discover_urdf_path
    - descriptions_module         <-> urdf_descriptions_module

Each pair shares shape, docstring style, and purpose (cheap name->module
lookup, cheap sorted roster, or heavy resolution). Prior to this pin the
MJCF side was re-exported from ``strands_robots.registry.__init__`` but the
URDF side was not; a user following the pattern ``from strands_robots.registry
import list_discoverable, list_urdf_discoverable`` hit an ``ImportError`` on
the second name and had to deep-import from ``.discovery``. This test pins the
re-export (and the symmetric top-level promotion of the two cheap probes) so
the asymmetry cannot regress.

The two cheap probes (``is_urdf_discoverable``, ``list_urdf_discoverable``)
are also promoted to the top-level ``strands_robots`` package, mirroring the
promotion of ``is_discoverable`` / ``list_discoverable``. The two heavier ones
(``discover_urdf_path``, ``urdf_descriptions_module``) stay package-level only,
mirroring ``discover_robot`` / ``descriptions_module``.
"""

from __future__ import annotations

import pytest


SIBLING_PAIRS: tuple[tuple[str, str], ...] = (
    ("is_discoverable", "is_urdf_discoverable"),
    ("list_discoverable", "list_urdf_discoverable"),
    ("discover_robot", "discover_urdf_path"),
    ("descriptions_module", "urdf_descriptions_module"),
)


@pytest.mark.parametrize("mjcf,urdf", SIBLING_PAIRS)
def test_urdf_sibling_is_in_registry_all(mjcf: str, urdf: str) -> None:
    """Every URDF sibling of a re-exported MJCF discovery function is itself
    re-exported from ``strands_robots.registry``."""
    from strands_robots import registry

    assert mjcf in registry.__all__, (
        f"regression: {mjcf} is expected to be in strands_robots.registry.__all__"
    )
    assert urdf in registry.__all__, (
        f"{urdf} is missing from strands_robots.registry.__all__ while its "
        f"sibling {mjcf} is present; the re-exported discovery API is "
        f"asymmetric and users following the sibling pattern ImportError."
    )


@pytest.mark.parametrize("mjcf,urdf", SIBLING_PAIRS)
def test_urdf_sibling_import_from_registry_works(mjcf: str, urdf: str) -> None:
    """``from strands_robots.registry import <urdf_sibling>`` must not
    ImportError when the MJCF sibling import works."""
    import importlib

    reg = importlib.import_module("strands_robots.registry")
    assert hasattr(reg, mjcf), f"regression: strands_robots.registry.{mjcf} not accessible"
    assert hasattr(reg, urdf), (
        f"strands_robots.registry.{urdf} is not accessible even though its "
        f"sibling {mjcf} is; a user who writes `from strands_robots.registry "
        f"import {urdf}` will ImportError. Add {urdf} to the re-export list "
        f"in strands_robots/registry/__init__.py."
    )


@pytest.mark.parametrize("probe", ["is_urdf_discoverable", "list_urdf_discoverable"])
def test_cheap_urdf_probe_is_at_top_level(probe: str) -> None:
    """The two cheap URDF probes are promoted to the top-level
    ``strands_robots`` package, mirroring ``is_discoverable`` and
    ``list_discoverable``. The heavier two (``discover_urdf_path``,
    ``urdf_descriptions_module``) are not promoted, mirroring
    ``discover_robot`` and ``descriptions_module``."""
    import strands_robots

    assert hasattr(strands_robots, probe), (
        f"strands_robots.{probe} is expected at the top-level package to "
        f"mirror the top-level promotion of its MJCF sibling."
    )
    assert probe in strands_robots.__all__, (
        f"strands_robots.__all__ is expected to include {probe}."
    )


@pytest.mark.parametrize("heavy", ["discover_urdf_path", "urdf_descriptions_module"])
def test_heavy_urdf_resolver_is_registry_level_only(heavy: str) -> None:
    """The two heavier URDF entry points stay package-level (registry.<name>)
    only, matching the shape of ``discover_robot`` and ``descriptions_module``.
    This is the sibling-shape test, not a hard refusal of future promotion."""
    from strands_robots import registry

    assert hasattr(registry, heavy), (
        f"strands_robots.registry.{heavy} must be accessible from the "
        f"registry package even if not promoted to the top-level namespace."
    )


def test_urdf_siblings_resolve_to_the_discovery_module() -> None:
    """The re-exports are the same callables as the ones in
    ``strands_robots.registry.discovery`` -- there is no shim layer that could
    drift from the submodule's definitions."""
    from strands_robots import registry
    from strands_robots.registry import discovery

    for name in (
        "is_urdf_discoverable",
        "list_urdf_discoverable",
        "discover_urdf_path",
        "urdf_descriptions_module",
    ):
        assert getattr(registry, name) is getattr(discovery, name), (
            f"strands_robots.registry.{name} must be the same object as "
            f"strands_robots.registry.discovery.{name}; the re-export is "
            f"meant to be a façade, not a wrapper that can diverge."
        )
