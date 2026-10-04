"""The requires_lerobot_from_source refusal fires before the generic listing.

Covers the fix for ``Robot("rebot_b601", mode="real")``: when the registry
entry carries ``hardware.requires_lerobot_from_source: true`` and lerobot's
draccus registry does not know the type (because the running lerobot is a
stable PyPI release and the type is only on main), the error must name the
registry entry and the from-source install command rather than fall through
to the generic ``Known lerobot robot types`` listing that by construction
excludes the robot the user asked for.

Sibling of
``tests/test_native_driver_refusal_name_fires_before_generic_listing.py`` and
``tests/test_other_lerobot_kind_refusal_names_the_teleoperator_kind.py``
(both pre-existing): the three refusals are the ordered chain in
``hardware_robot._initialize_lerobot_config``'s KeyError handler.
"""
from __future__ import annotations

import pytest

from strands_robots.hardware_robot import _requires_lerobot_from_source_refusal
from strands_robots.registry.robots import (
    find_requires_lerobot_from_source,
    get_robot,
)


REGISTERED_FROM_SOURCE_TYPES: tuple[tuple[str, str], ...] = (
    # (lerobot_type, registry_entry_name)
    ("rebot_b601_follower", "rebot_b601"),
    ("bi_rebot_b601_follower", "bi_rebot_b601"),
)


@pytest.mark.parametrize("lerobot_type,entry_name", REGISTERED_FROM_SOURCE_TYPES)
def test_registry_entry_still_declares_the_flag(
    lerobot_type: str, entry_name: str
) -> None:
    """The two entries this test is written for still carry the flag.

    If a future refactor renames the entry or drops the flag, this test fails
    loudly rather than silently stop covering the behaviour.
    """
    info = get_robot(entry_name)
    assert info is not None, f"registry entry {entry_name!r} vanished"
    hw = info.get("hardware")
    assert isinstance(hw, dict), f"{entry_name!r} has no hardware block"
    assert hw.get("lerobot_type") == lerobot_type, (
        f"{entry_name!r} no longer declares lerobot_type={lerobot_type!r}"
    )
    assert hw.get("requires_lerobot_from_source") is True, (
        f"{entry_name!r} no longer carries requires_lerobot_from_source=True"
    )


@pytest.mark.parametrize("lerobot_type,entry_name", REGISTERED_FROM_SOURCE_TYPES)
def test_find_requires_lerobot_from_source_names_the_entry(
    lerobot_type: str, entry_name: str
) -> None:
    """The registry lookup resolves the lerobot type to the curated name."""
    assert find_requires_lerobot_from_source(lerobot_type) == entry_name


def test_find_requires_lerobot_from_source_returns_none_for_an_unflagged_type() -> None:
    """A lerobot type without the flag (e.g. so101_follower) returns None.

    so101_follower is declared under ``so101`` with ``lerobot_type`` but
    no ``requires_lerobot_from_source`` -- the finder must not shadow it.
    """
    assert find_requires_lerobot_from_source("so101_follower") is None


def test_find_requires_lerobot_from_source_returns_none_for_an_unknown_type() -> None:
    """A lerobot type no registry entry claims returns None."""
    assert find_requires_lerobot_from_source("nonexistent_lerobot_type") is None


@pytest.mark.parametrize("lerobot_type,entry_name", REGISTERED_FROM_SOURCE_TYPES)
def test_refusal_names_the_type_entry_install_command_and_doc_page(
    lerobot_type: str, entry_name: str
) -> None:
    """The refusal mentions every piece the user needs to recover."""
    msg = _requires_lerobot_from_source_refusal(lerobot_type)
    assert msg is not None
    # The lerobot type the user hit (as a repr, matching the generic listing's shape).
    assert repr(lerobot_type) in msg
    # The curated registry entry name (as a repr).
    assert repr(entry_name) in msg
    # The install command (anchored enough that a copy-paste works).
    assert "pip install" in msg
    assert "git+https://github.com/huggingface/lerobot" in msg
    # The docs page that walks through usage.
    assert f"docs/robots/{entry_name}.md" in msg


def test_refusal_returns_none_for_a_type_without_the_flag() -> None:
    """A type with no from-source requirement leaves the generic listing in place.

    The chain contract: a sibling refusal returns None to decline, so the
    generic ``Known lerobot robot types`` listing survives for cases where
    it IS the right answer.
    """
    assert _requires_lerobot_from_source_refusal("so101_follower") is None


def test_refusal_returns_none_for_an_unknown_type() -> None:
    """An unknown type reaches the generic listing (the right answer there)."""
    assert _requires_lerobot_from_source_refusal("nonexistent_lerobot_type") is None
