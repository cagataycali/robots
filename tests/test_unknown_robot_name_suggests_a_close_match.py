"""``Robot("so1000")`` suggests the registered names it is closest to.

Measured on a fresh install, the refusal read ``Unknown robot
'so1000' (resolved to 'so1000'). Pass a registered name (see ``list_robots()``)``
- the resolved spelling repeated the input, no candidate was offered although
``add_robot(data_config="so1000")`` already answers "Did you mean: so100, so101",
and ``list_robots()`` was not qualified with the module that exports it.
"""

from __future__ import annotations

import pytest

from strands_robots import Robot


def test_typo_gets_the_nearest_registered_names(monkeypatch: pytest.MonkeyPatch) -> None:
    # The candidates come from the live registry, which merges the caller's own
    # ``register_robot`` entries: registering one robot named ``so1002`` makes
    # the real answer "so100, so1002, so101", so an exact list pinned against
    # the merged registry turns red on any box that used that public API. Pin
    # the message against a fixed registry; the two cells below keep exercising
    # the real one, where they do not depend on which names it holds.
    monkeypatch.setattr(
        "strands_robots.robot.list_robots",
        lambda: [{"name": name} for name in ("so100", "so101", "panda")],
    )
    with pytest.raises(ValueError, match=r"Did you mean: so100, so101\?") as info:
        Robot("so1000")
    msg = str(info.value)
    assert "(resolved to" not in msg, "resolved spelling equals the input; repeating it is noise"
    assert "strands_robots.list_robots()" in msg


def test_resolved_spelling_is_shown_only_when_it_differs() -> None:
    with pytest.raises(ValueError, match=r"'SO-100x' \(resolved to 'so_100x'\)"):
        Robot("SO-100x")


def test_no_near_match_still_points_at_the_catalog() -> None:
    with pytest.raises(ValueError) as info:
        Robot("zzzz")
    msg = str(info.value)
    assert "Did you mean" not in msg
    assert "strands_robots.list_robots()" in msg and "urdf_path=" in msg
