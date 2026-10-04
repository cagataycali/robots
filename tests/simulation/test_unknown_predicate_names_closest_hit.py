"""Pin: the three sibling ``Unknown predicate`` refusal sites each name
the closest registry hit, mirroring ``Robot('<typo>')`` at
``strands_robots.robot:211``.

Covers:
- :func:`make_predicate` (predicates.py raise-site #1)
- :func:`predicate_kind` (predicates.py raise-site #2)
- :func:`predicate_reads_robot_base` (predicates.py raise-site #3)

See harness bugbash #predicate-no-didyoumean.
"""
from __future__ import annotations

import pytest

from strands_robots.simulation.predicates import (
    PREDICATE_REGISTRY,
    make_predicate,
    predicate_kind,
    predicate_reads_robot_base,
)


# Each (typo, expected_hint) pair is a realistic end-user miss on a name
# that already lives in PREDICATE_REGISTRY — one letter off, missing
# underscore, missing suffix, or a plausible synonym.
TYPO_CASES = [
    ("joint_abovve", "joint_above"),
    ("basetipped", "base_tipped"),
    ("graspd", "grasped"),
    ("distance_less", "distance_less_than"),
    ("contact_any_body", "contact_any"),
]


@pytest.mark.parametrize("typo,expected", TYPO_CASES)
def test_make_predicate_names_closest_hit(typo: str, expected: str) -> None:
    with pytest.raises(ValueError) as ei:
        make_predicate(typo)
    msg = str(ei.value)
    assert "Did you mean" in msg, msg
    assert f"'{expected}'" in msg, msg
    # The valid-set dump stays — a caller who typed something off-registry
    # still gets the full enumeration.
    assert "Valid: [" in msg, msg


@pytest.mark.parametrize("typo,expected", TYPO_CASES)
def test_predicate_kind_names_closest_hit(typo: str, expected: str) -> None:
    with pytest.raises(ValueError) as ei:
        predicate_kind(typo)
    msg = str(ei.value)
    assert "Did you mean" in msg, msg
    assert f"'{expected}'" in msg, msg


@pytest.mark.parametrize("typo,expected", TYPO_CASES)
def test_predicate_reads_robot_base_names_closest_hit(
    typo: str, expected: str
) -> None:
    with pytest.raises(ValueError) as ei:
        predicate_reads_robot_base(typo)
    msg = str(ei.value)
    assert "Did you mean" in msg, msg
    assert f"'{expected}'" in msg, msg


def test_off_registry_typo_omits_the_hint() -> None:
    """A string nothing in the registry resembles must NOT be padded with
    a near-random first-letter neighbour (cutoff=0.6 is the gate). The
    valid-set dump still fires so the caller has the full menu."""
    with pytest.raises(ValueError) as ei:
        make_predicate("xxxxxxxxxxxxxxxx")
    msg = str(ei.value)
    assert "Did you mean" not in msg, msg
    assert "Valid: [" in msg, msg


def test_known_predicate_still_builds() -> None:
    """The hint path is refusal-only; a registered name still returns a
    callable. (Sanity against the helper leaking into the happy path.)"""
    assert "body_above_z" in PREDICATE_REGISTRY
    cb = make_predicate("body_above_z", body="cube", z=0.1)
    assert callable(cb)
