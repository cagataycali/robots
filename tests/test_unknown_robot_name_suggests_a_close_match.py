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


@pytest.mark.parametrize(
    ("overlay", "names_the_overlay"),
    [
        (None, False),
        (b'{"robots": {}}', False),
        (b'{"robots": {"my_scout": {"category": "arm"},,}}', True),
        (b"\xff\xfe not utf-8", True),
    ],
    ids=["absent", "valid", "trailing-comma", "not-utf8"],
)
def test_a_broken_user_overlay_is_named_in_the_refusal(
    tmp_path, monkeypatch: pytest.MonkeyPatch, overlay: bytes | None, names_the_overlay: bool
) -> None:
    # A one-character typo in user_robots.json hides every register_robot entry;
    # the refusal for the now-unknown name must point at the file, not only at
    # list_robots(). An absent or valid overlay adds nothing to the message.
    monkeypatch.setenv("STRANDS_BASE_DIR", str(tmp_path))
    path = tmp_path / "user_robots.json"
    if overlay is not None:
        path.write_bytes(overlay)
    with pytest.raises(ValueError, match="Unknown robot") as info:
        Robot("my_scout")
    msg = str(info.value)
    assert (str(path) in msg and "not valid JSON" in msg) is names_the_overlay
    assert "strands_robots.list_robots()" in msg
