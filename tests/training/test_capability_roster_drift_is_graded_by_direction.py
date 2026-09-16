# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A written lerobot roster is graded by the direction it drifted, not for equality.

``strands_robots.training.lerobot`` writes down which lerobot policy types carry
a capability (the ``*_POLICY_TYPES_FALLBACK`` snapshots, plus the ``validate()``
paragraph of ``docs/training/overview.md``), and those rosters are graded against
the set derived from the INSTALLED lerobot's config registry.

Graded for equality that cannot hold, because the manifest admits a RANGE and
lerobot moves a capability inside it: lerobot's #4425 added
``use_relative_actions`` to ``VLAJEPAConfig`` after v0.6.1, so the declared floor
accepts four policy types for relative actions and the next release accepts five.
Four types fail on the newer lerobot; naming the fifth fails on the floor, which
is the version the lockfile resolves - so an equality grader is red on one end of
the range whatever the roster says, and the roster cannot be repaired by editing
it.

:func:`~tests.training._lerobot_capability_range.roster_problem` splits the two
directions, and this file pins the split. The asymmetry is the whole point: a
roster naming a type the registry has no such field on makes an OFFLINE gate
accept a run lerobot will not honour, which is wrong at every version; a roster
lagging a lerobot newer than the floor is drift a frozen roster cannot avoid, and
becomes a failure the moment the floor is raised onto that release.
"""

from __future__ import annotations

import warnings

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

from strands_robots.training.lerobot import (
    _RELATIVE_ACTION_POLICY_TYPES_FALLBACK,
    _policy_registry,
    _policy_supports_relative_actions,
)
from tests.training import _lerobot_capability_range as capability_range
from tests.training._lerobot_capability_range import (
    declared_lerobot_floor,
    installed_lerobot_version,
    roster_problem,
)

_FLOOR = "0.6.1"
_ABOVE_FLOOR = "0.6.2"

#: ``(case, written, accepted, installed, failure owed, report owed)``. The
#: roster stands still and the installed lerobot moves, so each row differs from
#: its neighbour in one of the two things that decide the verdict. Exactly one
#: row is the tolerated case, and it is the only one that must be reported.
_CASES = (
    ("exact at the floor", {"pi0", "groot"}, {"pi0", "groot"}, _FLOOR, False, False),
    ("exact above the floor", {"pi0", "groot"}, {"pi0", "groot"}, _ABOVE_FLOOR, False, False),
    ("roster lags at the floor", {"pi0"}, {"pi0", "vla_jepa"}, _FLOOR, True, False),
    ("roster lags above the floor", {"pi0"}, {"pi0", "vla_jepa"}, _ABOVE_FLOOR, False, True),
    ("roster over-reports at the floor", {"pi0", "vla_jepa"}, {"pi0"}, _FLOOR, True, False),
    ("roster over-reports above the floor", {"pi0", "vla_jepa"}, {"pi0"}, _ABOVE_FLOOR, True, False),
    ("both directions above the floor", {"pi0", "act"}, {"pi0", "vla_jepa"}, _ABOVE_FLOOR, True, False),
)


@pytest.fixture
def at_version(monkeypatch: pytest.MonkeyPatch):
    """Report a chosen installed lerobot, against the manifest's real floor."""

    def _install(version: str) -> None:
        monkeypatch.setattr(capability_range, "installed_lerobot_version", lambda: Version(version))
        monkeypatch.setattr(capability_range, "declared_lerobot_floor", lambda: Version(_FLOOR))

    return _install


@pytest.mark.parametrize(
    ("case", "written", "accepted", "installed", "fails", "reports"), _CASES, ids=[c[0] for c in _CASES]
)
def test_only_a_lag_above_the_declared_floor_is_tolerated(
    case: str,
    written: set[str],
    accepted: set[str],
    installed: str,
    fails: bool,
    reports: bool,
    at_version,
) -> None:
    """Over-reporting always fails; lagging fails only up to the declared floor."""
    at_version(installed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        problem = roster_problem("a roster", written, accepted, written_label="roster-only", accepted_label="live-only")
    assert bool(problem) is fails, f"{case}: expected a failure={fails}, got {problem!r}"
    assert bool(caught) is reports, f"{case}: expected a report={reports}, got {[str(w.message) for w in caught]}"


def test_a_tolerated_lag_reports_both_versions_and_the_types(at_version) -> None:
    """The lag is silent to the suite but not to the reader: it must be actionable."""
    at_version(_ABOVE_FLOOR)
    with pytest.warns(UserWarning) as caught:
        assert (
            roster_problem(
                "a roster", {"pi0"}, {"pi0", "vla_jepa"}, written_label="roster-only", accepted_label="live-only"
            )
            is None
        )
    message = str(caught[0].message)
    for token in ("vla_jepa", _ABOVE_FLOOR, _FLOOR):
        assert token in message, f"the report must name {token!r}: {message}"


def test_the_over_report_message_names_the_types_and_the_remedy(at_version) -> None:
    """The direction that is always wrong must say which types to drop."""
    at_version(_ABOVE_FLOOR)
    problem = roster_problem(
        "a roster", {"pi0", "act"}, {"pi0"}, written_label="roster-only", accepted_label="live-only"
    )
    assert problem is not None
    assert "act" in problem and "drop" in problem, problem


def test_the_graded_floor_is_the_lowest_release_the_manifest_admits() -> None:
    """The floor is read from the manifest, so raising it re-grades the rosters."""
    manifest_floor = declared_lerobot_floor()
    specifier = _declared_lerobot_specifier()
    assert manifest_floor in specifier, f"{manifest_floor} is not admitted by {specifier}"
    below = Version(f"{manifest_floor.major}.{manifest_floor.minor}.{max(manifest_floor.micro - 1, 0)}")
    assert below not in specifier, f"{below} is below the floor yet admitted by {specifier}"


def _declared_lerobot_specifier():
    """The ``lerobot`` specifier the ``[lerobot]`` extra declares."""
    import tomllib

    manifest = tomllib.loads(capability_range._PYPROJECT.read_text(encoding="utf-8"))
    extra = manifest["project"]["optional-dependencies"]["lerobot"]
    return next(r for r in map(Requirement, extra) if r.name == "lerobot").specifier


def test_the_shipped_relative_action_roster_over_reports_on_no_installed_lerobot() -> None:
    """The direction that is version-independent, pinned against the real install.

    Whatever lerobot is installed, every type the shipped roster names must
    really expose ``use_relative_actions`` - otherwise an offline
    ``validate()`` accepts ``relative_actions`` for a policy lerobot has no
    field for.
    """
    registry = _policy_registry()
    if registry is None:
        pytest.skip("lerobot not installed; the live registry is what grades the roster")
    accepted = {ptype for ptype in registry if _policy_supports_relative_actions(ptype)}
    over_reported = sorted(set(_RELATIVE_ACTION_POLICY_TYPES_FALLBACK) - accepted)
    assert not over_reported, (
        f"_RELATIVE_ACTION_POLICY_TYPES_FALLBACK names types lerobot "
        f"{installed_lerobot_version()} has no use_relative_actions field on: {over_reported}"
    )
