#!/usr/bin/env python3
"""Repro: README promises `150+ robots across 8 categories` but the function
that answers the categorical split (``list_robots_by_category``) is missing
from the top-level ``strands_robots`` namespace, while its sibling
``list_robots`` is promoted.

Three discovery siblings share the same registry module and the same
_LAZY_IMPORTS shape (list_robots, list_discoverable, list_urdf_only); the
fourth (list_robots_by_category) is omitted. The AttributeError the user
hits is bare (no ``difflib.get_close_matches`` 'did you mean' hint),
unlike register_robot / create_policy / create_trainer / add_camera / set_gripper
refusals in the same codebase.

Expected on fix: list_robots_by_category reachable top-level, same lazy
shape as its three siblings, same ``__all__`` membership.

Target: v0.5.3 - rotation readme_quickstart.
"""

from __future__ import annotations

import importlib
import sys


def main() -> int:
    import strands_robots

    # ------------------------------------------------------------------
    # (1) README promise -> top-level reach
    # ------------------------------------------------------------------
    print("--- README promise: '150+ robots across 8 categories' ---")
    print("README.md line (hero → What you get table row 1):")
    print("  '150+ robots across 8 categories - arms, bimanual, hands and")
    print("   grippers, humanoids, mobile bases, mobile manipulators, aerial, expressive'")
    print()
    print("Natural user reach (matching every docs/start/*.md import):")
    print("    from strands_robots import list_robots_by_category")
    try:
        from strands_robots import list_robots_by_category  # noqa: F401
        reach_ok = True
    except ImportError as exc:
        reach_ok = False
        print(f"  ImportError: {exc}")
    print(f"  reachable: {reach_ok}")

    # Direct attribute access (what list_robots-style users try first)
    print()
    print("    strands_robots.list_robots_by_category")
    try:
        _ = strands_robots.list_robots_by_category
        attr_ok = True
    except AttributeError as exc:
        attr_ok = False
        print(f"  AttributeError: {exc}")
    print(f"  reachable: {attr_ok}")

    # ------------------------------------------------------------------
    # (2) Sibling asymmetry
    # ------------------------------------------------------------------
    print()
    print("--- Sibling-shape asymmetry ---")
    siblings = [
        "list_robots",
        "list_robots_by_category",
        "list_discoverable",
        "list_urdf_only",
    ]
    print(f"{'name':<28} {'in registry.__all__':<22} {'in top-level __all__':<22} {'top-level reach':<16}")
    from strands_robots import registry as _reg
    registry_all = set(getattr(_reg, "__all__", ()))
    top_all = set(getattr(strands_robots, "__all__", ()))
    rows = []
    for name in siblings:
        in_reg = name in registry_all
        in_top = name in top_all
        try:
            getattr(strands_robots, name)
            top_reach = True
        except AttributeError:
            top_reach = False
        rows.append((name, in_reg, in_top, top_reach))
        print(f"{name:<28} {str(in_reg):<22} {str(in_top):<22} {str(top_reach):<16}")

    # ------------------------------------------------------------------
    # (3) Same-callable identity (facade must not drift once added)
    # ------------------------------------------------------------------
    print()
    print("--- Same-callable identity check ---")
    from strands_robots.registry import list_robots_by_category as _lrbc_reg
    from strands_robots.assets import list_robots_by_category as _lrbc_assets
    print(f"  registry.list_robots_by_category is assets.list_robots_by_category: "
          f"{_lrbc_reg is _lrbc_assets}")

    # ------------------------------------------------------------------
    # (4) AttributeError has no 'did you mean' hint
    # ------------------------------------------------------------------
    print()
    print("--- 'Did you mean' hint on AttributeError? ---")
    try:
        strands_robots.list_robots_by_category
    except AttributeError as exc:
        txt = str(exc)
        has_hint = "did you mean" in txt.lower() or " →" in txt or "->" in txt
        print(f"  AttributeError text: {txt!s}")
        print(f"  'did you mean' hint present: {has_hint}")
        print(f"  (sibling discovery/factory refusals in-repo use difflib.get_close_matches)")

    # ------------------------------------------------------------------
    # (5) Did we succeed via the deep path?
    # ------------------------------------------------------------------
    print()
    print("--- Deep-import workaround still works ---")
    from strands_robots.registry import list_robots_by_category
    by_cat = list_robots_by_category()
    print(f"  registry.list_robots_by_category() returns {len(by_cat)} categories: "
          f"{sorted(by_cat.keys())}")
    print(f"  total robots across categories: {sum(len(v) for v in by_cat.values())}")

    # Exit non-zero when top-level reach fails (bug present)
    bug_present = (not reach_ok) or (not attr_ok)
    print()
    print(f"BUG_PRESENT: {bug_present}")
    return 1 if bug_present else 0


if __name__ == "__main__":
    sys.exit(main())
