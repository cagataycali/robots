"""Repro: strands_robots.registry's URDF discovery siblings are missing from the
public re-export list, while their MJCF counterparts are not.

Reads a user intuition that `is_discoverable` <-> `is_urdf_discoverable`,
`list_discoverable` <-> `list_urdf_discoverable`, `discover_robot` <->
`discover_urdf_path`, `descriptions_module` <-> `urdf_descriptions_module`
are sibling pairs on the same module (`strands_robots.registry.discovery`).
Three of the four MJCF-side names are re-exported from
`strands_robots.registry.__init__` (`__all__`); two of the four are promoted
all the way to the top-level `strands_robots` package. Zero of the four URDF
siblings are re-exported at either layer.

Expected (both of the following imports work, and both report the same thing):

    from strands_robots.registry import is_discoverable, list_discoverable
    from strands_robots.registry import is_urdf_discoverable, list_urdf_discoverable

Actual (second line ImportErrors on main; the symbol only exists under the
deep sub-module path `strands_robots.registry.discovery`).

Run::

    python3 bugbash_repros/registry_urdf_discovery_not_reexported_repro.py
"""

from __future__ import annotations

import importlib


# ---- Known sibling pairs (MJCF <-> URDF), both in registry/discovery.py ----
SIBLING_PAIRS: list[tuple[str, str]] = [
    ("is_discoverable", "is_urdf_discoverable"),
    ("list_discoverable", "list_urdf_discoverable"),
    ("discover_robot", "discover_urdf_path"),
    ("descriptions_module", "urdf_descriptions_module"),
]


def _is_in(name: str, module_path: str) -> bool:
    mod = importlib.import_module(module_path)
    all_list = getattr(mod, "__all__", None)
    if all_list is None:
        return hasattr(mod, name)
    return name in all_list


def _can_import_from(module_path: str, name: str) -> bool:
    try:
        importlib.import_module(module_path)
        exec(f"from {module_path} import {name}", {})
        return True
    except ImportError:
        return False


def main() -> int:
    print("=== Sibling re-export audit for strands_robots.registry URDF side ===\n")
    print(f"{'MJCF':<28} {'URDF':<28} {'MJCF@reg':<10} {'URDF@reg':<10} {'MJCF@top':<10} {'URDF@top':<10}")
    print("-" * 100)
    asymmetries: list[tuple[str, str]] = []
    for mjcf, urdf in SIBLING_PAIRS:
        mjcf_reg = _can_import_from("strands_robots.registry", mjcf)
        urdf_reg = _can_import_from("strands_robots.registry", urdf)
        mjcf_top = _can_import_from("strands_robots", mjcf)
        urdf_top = _can_import_from("strands_robots", urdf)
        print(f"{mjcf:<28} {urdf:<28} {str(mjcf_reg):<10} {str(urdf_reg):<10} {str(mjcf_top):<10} {str(urdf_top):<10}")
        if mjcf_reg and not urdf_reg:
            asymmetries.append((mjcf, urdf))

    print()
    print(f"Asymmetric sibling pairs (MJCF re-exported, URDF not): {len(asymmetries)}")
    for mjcf, urdf in asymmetries:
        print(f"  - {mjcf} is in strands_robots.registry.__all__ but {urdf} is not")

    print()
    print("--- Attempted sibling imports a user would try ---")
    try:
        from strands_robots.registry import list_urdf_discoverable  # noqa: F401
        print("from strands_robots.registry import list_urdf_discoverable: OK")
    except ImportError as e:
        print(f"from strands_robots.registry import list_urdf_discoverable: ImportError: {e}")

    try:
        from strands_robots.registry import is_urdf_discoverable  # noqa: F401
        print("from strands_robots.registry import is_urdf_discoverable: OK")
    except ImportError as e:
        print(f"from strands_robots.registry import is_urdf_discoverable: ImportError: {e}")

    print()
    print("--- Internal code works around the omission ---")
    print("  strands_robots/simulation/newton/simulation.py:41:")
    print("    from strands_robots.registry.discovery import discover_urdf_path, list_urdf_discoverable")
    print("  scripts/build_urdf_registry.py:42:")
    print("    from strands_robots.registry.discovery import list_urdf_only, urdf_descriptions_module")
    print()
    print("--- docs/reference/api/registry.md 'Discovery through robot_descriptions' ---")
    print("  Lists 4 members: is_discoverable, list_discoverable, discover_robot, descriptions_module")
    print("  Lists 0 URDF siblings even though they live in the same submodule and are documented in docstrings.")

    return 0 if not asymmetries else 1


if __name__ == "__main__":
    raise SystemExit(main())
