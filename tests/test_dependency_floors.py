"""Repo hygiene: pin minimum security-floor versions and cross-site consistency.

These tests guard the dependency constraints in ``pyproject.toml`` so a routine
``uv lock --upgrade`` or constraint-cleanup PR cannot silently lower a floor
that was raised for a CVE, or let the two pin sites
(``[project.dependencies]`` / ``[project.optional-dependencies].dev`` /
``[tool.hatch.envs.default]``) drift out of sync.

Pinned floors (raise, never lower without an explicit security review):

- Pillow >= 10.3.0 -- carries CVE-2024-28219 buffer-overflow fix.
  Raised in PR #153 R1 from the previous >=8.0.0 floor.

  Why only Pillow here? Floors are only added when a CVE recurs in a *direct*
  dep that downstream `pip install` consumers can resolve below the fixed
  version. Transitive bumps (cryptography, urllib3, gitpython,
  python-multipart, etc.) are mitigated by `uv.lock` regeneration -- they
  have no direct constraint to pin.

Cross-site consistency: every dep that appears in more than one pin block must
have identical floor + ceiling everywhere it appears, otherwise a future
Dependabot PR can update one site and forget the other -- the exact pattern
that motivated the floor sweep in PR #153. The check is property-style
(scans every dep that appears in >=2 sites), so adding a new dual-site dep
does not require updating an allowlist.

Add an entry to ``_SECURITY_FLOORS`` whenever a floor is raised for a CVE
fix.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


# --------------------------------------------------------------------------- #
# Constraint parsing
# --------------------------------------------------------------------------- #

# Match e.g. ``Pillow>=10.3.0,<13.0.0``. Permissive on whitespace; case-insensitive
# on the package name (``Pillow`` vs ``pillow``).
_CONSTRAINT_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9._-]+)\s*"
    r"(?P<spec>(?:[<>=!~]=?\s*[A-Za-z0-9._*+-]+(?:\s*,\s*[<>=!~]=?\s*[A-Za-z0-9._*+-]+)*)?)\s*$"
)


def _parse_constraint(entry: str) -> tuple[str, str]:
    """Return ``(canonical_name_lower, normalised_spec)``."""
    m = _CONSTRAINT_RE.match(entry.split(";", 1)[0])  # drop env markers
    if m is None:
        raise ValueError(f"Could not parse constraint: {entry!r}")
    name = m.group("name").lower().replace("_", "-")
    spec = re.sub(r"\s+", "", m.group("spec"))
    return name, spec


def _floor_of(spec: str) -> str | None:
    """Extract the ``>=`` lower bound from a spec like ``>=10.3.0,<13.0.0``.

    Only ``>=`` is recognised. ``~=`` and ``==`` floors are intentionally not
    accepted: ``_SECURITY_FLOORS`` entries must use ``>=`` so the pin clearly
    states the minimum-acceptable version. If a future dep needs a different
    operator, broaden this helper at the same time as the floor is added.
    """
    m = re.search(r">=\s*([A-Za-z0-9._*+-]+)", spec)
    return m.group(1) if m else None


def _parse_version(v: str) -> Version:
    """Parse a version string using PEP 440 semantics.

    Uses ``packaging.version.Version`` for correct handling of pre-releases,
    post-releases, dev-releases, and local versions. This ensures that a
    floor like ``>=10.3.0`` correctly rejects ``10.3.0rc1`` (which is
    *before* the release and may not contain the CVE fix).
    """
    return Version(v)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


@pytest.fixture(scope="module")
def constraint_index(pyproject: dict) -> dict[str, dict[str, str]]:
    """Map ``dep_name -> {site: spec}`` across every pin site in pyproject.

    Sites tracked:
      - ``project.dependencies``
      - ``project.optional-dependencies.<extra>`` for every declared extra
      - ``tool.hatch.envs.default.dependencies``

    Self-references with extras (``"strands-robots[mesh]"``) have no spec and
    are intentionally skipped by ``_parse_constraint`` raising ``ValueError``;
    a real parse failure on a typo'd spec re-raises so the test surfaces the
    bug instead of silently dropping the entry.
    """
    index: dict[str, dict[str, str]] = {}

    def absorb(site: str, entries: list[str]) -> None:
        for raw in entries:
            try:
                name, spec = _parse_constraint(raw)
            except ValueError:
                # Self-reference like "strands-robots[mesh]" has no spec;
                # let any other parse failure propagate so a typo doesn't
                # silently disappear.
                if raw.lstrip().startswith("strands-robots["):
                    continue
                raise
            index.setdefault(name, {})[site] = spec

    project = pyproject.get("project", {})
    absorb("project.dependencies", project.get("dependencies", []) or [])
    for extra, entries in (project.get("optional-dependencies", {}) or {}).items():
        absorb(f"project.optional-dependencies.{extra}", entries or [])

    hatch_default = (
        pyproject.get("tool", {}).get("hatch", {}).get("envs", {}).get("default", {}).get("dependencies", []) or []
    )
    absorb("tool.hatch.envs.default.dependencies", hatch_default)

    return index


# --------------------------------------------------------------------------- #
# Security floors -- raise, never silently lower
# --------------------------------------------------------------------------- #

# (canonical_name, minimum_floor, rationale)
_SECURITY_FLOORS: list[tuple[str, str, str]] = [
    (
        "pillow",
        "10.3.0",
        "CVE-2024-28219 buffer-overflow fix; raised in PR #153 R1.",
    ),
]


@pytest.mark.parametrize(("name", "min_floor", "rationale"), _SECURITY_FLOORS)
def test_security_floor_not_lowered(
    constraint_index: dict[str, dict[str, str]],
    name: str,
    min_floor: str,
    rationale: str,
) -> None:
    """Every pin site that declares ``name`` must enforce ``>=min_floor``.

    Lowering a security floor requires deleting the entry from
    ``_SECURITY_FLOORS`` with a reviewer-visible explanation, so the change
    surfaces in PR review rather than slipping through a silent ``uv lock``.
    """
    sites = constraint_index.get(name)
    assert sites, (
        f"{name} not found in any pyproject pin site -- floor pin is dead. "
        f"Either add it back or remove from _SECURITY_FLOORS. ({rationale})"
    )

    for site, spec in sites.items():
        floor = _floor_of(spec)
        assert floor is not None, (
            f"{name} in {site} has no >= floor (spec={spec!r}); cannot enforce CVE floor. {rationale}"
        )
        assert _parse_version(floor) >= _parse_version(min_floor), (
            f"{name} floor regression in {site}: spec={spec!r} "
            f"declares >={floor} but security minimum is >={min_floor}. "
            f"{rationale}"
        )


# --------------------------------------------------------------------------- #
# Multi-site consistency -- pinned in multiple blocks must match exactly
# --------------------------------------------------------------------------- #


def test_multi_site_pins_consistent(constraint_index: dict[str, dict[str, str]]) -> None:
    """Every dep declared in 2+ pin sites must carry an identical spec.

    Catches the foot-gun where a Dependabot PR updates
    ``[project.optional-dependencies].dev`` but forgets
    ``[tool.hatch.envs.default]`` (or vice versa), letting the hatch env
    silently resolve a different version than the wheel install.

    Property-style: scans every dep automatically. Adding a new dual-site
    dep does not require updating an allowlist; the test catches drift on
    every dep that lives in 2+ sites today and any added tomorrow.
    """
    drifted = {
        name: sites for name, sites in constraint_index.items() if len(sites) >= 2 and len(set(sites.values())) > 1
    }
    assert not drifted, (
        f"Multi-site pins disagree -- update every site to the same constraint when bumping. Drift: {drifted}"
    )


# --------------------------------------------------------------------------- #
# PEP 440 pre-release correctness (regression pin for R5 fix)
# --------------------------------------------------------------------------- #


def test_parse_version_rejects_prerelease_below_release() -> None:
    """Verify that _parse_version correctly identifies pre-releases as below their release.

    Regression pin: before R5, ``_version_tuple("10.3.0rc1")`` returned ``(10, 3, 0)``
    which compared equal to ``_version_tuple("10.3.0")``, silently accepting a
    pre-release floor that might not carry the CVE fix. With ``packaging.version.Version``,
    ``10.3.0rc1 < 10.3.0`` is correctly enforced.
    """
    assert _parse_version("10.3.0rc1") < _parse_version("10.3.0")
    assert _parse_version("10.3.0") >= _parse_version("10.3.0")
    assert _parse_version("10.3") == _parse_version("10.3.0")
    assert _parse_version("10.4.0") > _parse_version("10.3.0")
    assert _parse_version("10.3.0.post1") > _parse_version("10.3.0")
    assert _parse_version("10.3.0.dev1") < _parse_version("10.3.0")
