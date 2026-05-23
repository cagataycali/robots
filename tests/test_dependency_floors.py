"""Repo hygiene: pin minimum security-floor versions and cross-site consistency.

These tests guard the dependency constraints in ``pyproject.toml`` so a routine
``uv lock --upgrade`` or constraint-cleanup PR cannot silently lower a floor
that was raised for a CVE, or let the two pin sites
(``[project.dependencies]`` / ``[project.optional-dependencies].dev`` /
``[tool.hatch.envs.default]``) drift out of sync.

Pinned floors (raise, never lower without an explicit security review):

- Pillow >= 10.3.0 -- carries CVE-2024-28219 buffer-overflow fix.
  Raised in PR #153 R1 from the previous >=8.0.0 floor.

Cross-site consistency: every dep that appears in more than one pin block must
have identical floor + ceiling everywhere it appears, otherwise a future
Dependabot PR can update one site and forget the other -- the exact pattern
that motivated the floor sweep in PR #153.

Add an entry to ``_SECURITY_FLOORS`` whenever a floor is raised for a CVE
fix. Add an entry to ``_DUAL_SITE_DEPS`` whenever a new dep is declared in
both the runtime/dev block and the hatch env block.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

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
    """Extract the ``>=`` lower bound from a spec like ``>=10.3.0,<13.0.0``."""
    m = re.search(r">=\s*([A-Za-z0-9._*+-]+)", spec)
    return m.group(1) if m else None


def _version_tuple(v: str) -> tuple[int, ...]:
    """Best-effort numeric tuple for floor comparison."""
    parts = re.split(r"[.\-+]", v)
    out: list[int] = []
    for p in parts:
        m = re.match(r"^(\d+)", p)
        if m:
            out.append(int(m.group(1)))
        else:
            break
    return tuple(out) or (0,)


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
    """
    index: dict[str, dict[str, str]] = {}

    def absorb(site: str, entries: list[str]) -> None:
        for raw in entries:
            try:
                name, spec = _parse_constraint(raw)
            except ValueError:
                continue
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
        assert _version_tuple(floor) >= _version_tuple(min_floor), (
            f"{name} floor regression in {site}: spec={spec!r} "
            f"declares >={floor} but security minimum is >={min_floor}. "
            f"{rationale}"
        )


# --------------------------------------------------------------------------- #
# Two-site consistency -- pinned in multiple blocks must match exactly
# --------------------------------------------------------------------------- #

# Deps that are intentionally pinned in BOTH the runtime/dev block AND the
# ``[tool.hatch.envs.default]`` block. Any drift between sites is a bug.
_DUAL_SITE_DEPS = ("pillow", "pytest", "pytest-cov")


@pytest.mark.parametrize("name", _DUAL_SITE_DEPS)
def test_pin_site_consistency(constraint_index: dict[str, dict[str, str]], name: str) -> None:
    """Same dep in multiple pin sites must carry identical specs.

    Catches the foot-gun where a Dependabot PR updates
    ``[project.optional-dependencies].dev`` but forgets
    ``[tool.hatch.envs.default]`` (or vice versa), letting the hatch env
    silently resolve a different version than the wheel install.
    """
    sites = constraint_index.get(name, {})
    assert len(sites) >= 2, (
        f"{name} expected in >= 2 pin sites, found {len(sites)}: "
        f"{sorted(sites)}. Either add the missing site or remove "
        f"{name!r} from _DUAL_SITE_DEPS."
    )
    distinct = set(sites.values())
    assert len(distinct) == 1, (
        f"{name} pin sites disagree: {sites}. Update every site to the same constraint when bumping."
    )
