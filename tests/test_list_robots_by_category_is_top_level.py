"""Pin test: ``strands_robots.list_robots_by_category`` is top-level reachable.

The README's own prose ("150+ robots across 8 categories") promises the
categorical split; the function that answers it must share the same
promotion shape as its three ``registry`` siblings.
"""

from __future__ import annotations

import strands_robots


def test_list_robots_by_category_is_top_level_import():
    """Must be importable the same way list_robots is."""
    from strands_robots import list_robots_by_category  # noqa: F401
    from strands_robots import list_robots  # noqa: F401

    # The function must come from the registry façade (not a shim).
    from strands_robots.registry import (
        list_robots_by_category as _registry_lrbc,
    )

    assert strands_robots.list_robots_by_category is _registry_lrbc, (
        "top-level strands_robots.list_robots_by_category must be the SAME "
        "callable as strands_robots.registry.list_robots_by_category; a wrapper "
        "would drift between the two layers."
    )


def test_list_robots_by_category_is_in_dunder_all():
    """`__all__` controls `from strands_robots import *` and type-checker surface."""
    assert "list_robots_by_category" in strands_robots.__all__, (
        "list_robots_by_category must be in strands_robots.__all__ next to its "
        "three registry siblings (list_robots, list_discoverable, list_urdf_only)."
    )


def test_four_registry_discovery_siblings_share_top_level_promotion_shape():
    """The four read-API siblings all live in registry.__all__ and must all
    be promoted top-level - any one missing is an invitation to re-emerge."""
    from strands_robots import registry as _registry

    siblings = (
        "list_robots",
        "list_robots_by_category",
        "list_discoverable",
        "list_urdf_only",
    )
    for name in siblings:
        assert name in _registry.__all__, (
            f"{name} must be in strands_robots.registry.__all__ (source-of-truth façade)"
        )
        assert name in strands_robots.__all__, (
            f"{name} must be in strands_robots.__all__ next to its {len(siblings)-1} "
            f"siblings - sibling-shape asymmetry is a UX cliff."
        )
        # Attribute access (triggers lazy __getattr__ for the real callable)
        attr = getattr(strands_robots, name)
        assert callable(attr), f"strands_robots.{name} must be callable"


def test_list_robots_by_category_returns_eight_categories_matching_readme():
    """The README's '150+ robots across 8 categories' claim must be
    observable through the function the user imports."""
    by_cat = strands_robots.list_robots_by_category()
    assert isinstance(by_cat, dict)
    # The README names 8 categories.
    assert len(by_cat) == 8, (
        f"README prose: 'across 8 categories' - got {len(by_cat)}: {sorted(by_cat)}"
    )
    # The README's 150+ claim: cross-category total >= 150.
    total = sum(len(v) for v in by_cat.values())
    assert total >= 150, (
        f"README prose: '150+ robots' - list_robots_by_category sums to {total}"
    )
