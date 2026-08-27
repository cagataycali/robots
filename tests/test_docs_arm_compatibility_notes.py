"""Repo hygiene: the arms page's capability claims agree with the registry.

``docs/robots/arms.md`` ends in a *Compatibility notes* block whose bullets are
set-membership claims about ``registry/robots.json``: which arms have no sim
asset, and which have a real-hardware path. A reader plans hardware work from
them, so a stale name there is a wrong answer rather than a cosmetic one.

Nothing graded those bullets, and all three claims had gone stale:

* The sim-asset exception named ``hope_jr`` and ``omx``. ``rebot_b601`` also
  declares no ``asset`` block, so it was the third exception and unlisted.
* The real-hardware bullet named ``panda``, ``so100`` and ``ur5e``. Only
  ``so100`` was right - neither ``panda`` nor ``ur5e`` declares a ``hardware``
  block at all, so ``Robot("panda", mode="real")`` refuses with
  ``Unsupported robot type: 'panda'``. The claim mattered because it is the
  opposite of the truth: a reader was told the Franka arms already drive real
  hardware through LeRobot, when LeRobot registers no Franka type.
* "The rest are simulation-only" then misdescribed the nine arms that do have a
  path (``dynamixel_2r``, ``hope_jr``, ``koch``, ``omx``, ``openarm``,
  ``rebot_b601``, ``so101``, ``vx300s``, ``wx250s``).

``tests/test_docs_robot_catalog_coverage.py`` grades the same page and could not
see any of it: its four guards pin catalog-table *membership*, the robot *counts*
and the *Aliases* column, and a capability claim in the prose block is none of
those. This file grades the prose.

Both routes are read from the repo's own sources of truth rather than restated
here, so an arm that gains a driver fails the bullet that should have named it:

* LeRobot: the entry declares ``hardware.lerobot_type``.
  :func:`test_every_declared_lerobot_type_is_one_lerobot_registers` is the
  premise that makes declaring one mean the path works.
* Native: :func:`strands_robots.drivers.registry.get_native_driver_class`
  answers for the name.

Deliberately out of scope: the block's last bullet, about what the ``joints``
count includes. Of the 59 registry robots whose asset compiles here, 50 declare
a ``joints`` value other than the asset's actuator count, so that field is a
loose informational number whose contract needs deciding before it can be
graded - a different question from which arms drive hardware.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ROBOTS_JSON = REPO_ROOT / "strands_robots" / "registry" / "robots.json"
ARMS_PAGE = REPO_ROOT / "docs" / "robots" / "arms.md"

#: Pattern identifying the bullet each claim lives on, matched against the
#: whitespace-normalised bullet so a re-wrap cannot stop a rule applying. Each
#: keys on the claim's *subject* rather than on one phrasing, so it still finds
#: the bullet after a reword: ``sim asset`` matched the shipped
#: "have no MuJoCo sim asset" as well as the current "declare no sim asset".
NO_SIM_ASSET_CLAIM = ("sim-asset exception", re.compile(r"sim asset", re.IGNORECASE))
LEROBOT_CLAIM = ("LeRobot real-hardware", re.compile(r"real hardware.*lerobot", re.IGNORECASE))
NATIVE_CLAIM = ("native-driver real-hardware", re.compile(r"native (?:strands )?driver", re.IGNORECASE))
CLAIMS = (NO_SIM_ASSET_CLAIM, LEROBOT_CLAIM, NATIVE_CLAIM)


def _registry() -> dict[str, dict]:
    """Return the built-in robot registry, keyed by canonical name."""
    return json.loads(ROBOTS_JSON.read_text(encoding="utf-8"))["robots"]


def _arm_names() -> set[str]:
    """Return every registry name whose category is ``arm``."""
    return {name for name, entry in _registry().items() if entry.get("category") == "arm"}


def _compatibility_bullets() -> list[str]:
    """Return the *Compatibility notes* bullets, each whitespace-normalised.

    A bullet is joined across its continuation lines, so a rule keys on the
    sentence rather than on where the line happens to wrap.
    """
    text = ARMS_PAGE.read_text(encoding="utf-8")
    section = re.search(r"\n## Compatibility notes\b(.*?)(?:\n## |\Z)", text, re.DOTALL)
    assert section, "docs/robots/arms.md is missing a '## Compatibility notes' section"
    bullets: list[str] = []
    for line in section.group(1).splitlines():
        if line.startswith("- "):
            bullets.append(line[2:])
        elif line.strip() and bullets:
            bullets[-1] += " " + line.strip()
    return [" ".join(bullet.split()) for bullet in bullets]


def _names_in(bullet: str) -> set[str]:
    """Return the registry arm names a bullet lists.

    Only backticked tokens that are registry arm names count, so incidental code
    spans (``lerobot_type``, ``driver="strands"``) are not read as claims.
    """
    return {token for token in re.findall(r"`([^`]+)`", bullet)} & _arm_names()


def _claimed_arms(claim: tuple[str, re.Pattern[str]]) -> set[str] | None:
    """Return the arms the bullet making ``claim`` lists, or ``None`` if absent.

    ``None`` distinguishes "the page makes no such claim" from "it claims an
    empty set", so a rule can report the missing bullet rather than a set
    difference against nothing.
    """
    what, pattern = claim
    matching = [bullet for bullet in _compatibility_bullets() if pattern.search(bullet)]
    assert len(matching) <= 1, (
        f"{len(matching)} Compatibility notes bullets make the {what} claim: {matching}. "
        "Split or merge them so exactly one bullet owns it."
    )
    return _names_in(matching[0]) if matching else None


def _arms_without_a_sim_asset() -> set[str]:
    """Return the arms whose registry entry declares no ``asset`` block."""
    return {name for name in _arm_names() if not _registry()[name].get("asset")}


def _arms_with_a_lerobot_type() -> set[str]:
    """Return the arms whose registry entry names a ``hardware.lerobot_type``."""
    return {name for name in _arm_names() if (_registry()[name].get("hardware") or {}).get("lerobot_type")}


def _arms_with_a_native_driver() -> set[str]:
    """Return the arms a native Strands driver is registered for."""
    from strands_robots.drivers.registry import get_native_driver_class

    return {name for name in _arm_names() if get_native_driver_class(name) is not None}


def _report(claimed: set[str] | None, derived: set[str], what: str) -> str:
    """Return a failure message naming the exact edit the page needs."""
    if claimed is None:
        return (
            f"docs/robots/arms.md Compatibility notes makes no {what} claim, but the registry gives "
            f"that path to {sorted(derived)}. Add a bullet naming them, so a reader planning hardware "
            "work is not told the robot is simulation-only."
        )
    return (
        f"docs/robots/arms.md Compatibility notes: the {what} bullet lists "
        f"{sorted(claimed)} but the registry says {sorted(derived)}.\n"
        f"  missing from the page: {sorted(derived - claimed)}\n"
        f"  named but not true:    {sorted(claimed - derived)}"
    )


class TestTheBlockIsShapedTheWayTheRulesAssume:
    """Premises. Each rule below reads a set off the page; these pin that it can."""

    def test_the_page_has_a_compatibility_notes_section(self) -> None:
        assert _compatibility_bullets(), "no bullets found under '## Compatibility notes'"

    def test_the_registry_declares_arms(self) -> None:
        assert len(_arm_names()) > 10, f"only {len(_arm_names())} arms - the rules below would be near-vacuous"

    @pytest.mark.parametrize("claim", CLAIMS, ids=[claim[0] for claim in CLAIMS])
    def test_each_claim_the_page_makes_names_at_least_one_arm(self, claim: tuple[str, re.Pattern[str]]) -> None:
        """A bullet that makes a claim must name an arm, or its rule reads nothing.

        A page that omits the bullet entirely is a different failure, reported by
        the rule that grades that claim, so it is skipped here rather than
        conflated with an empty list.
        """
        what, _ = claim
        claimed = _claimed_arms(claim)
        if claimed is None:
            pytest.skip(f"the page makes no {what} claim - the rule for it reports that")
        assert claimed, f"the {what} bullet names no registry arm, so its rule reads nothing"

    def test_every_arm_declaring_hardware_declares_a_lerobot_type(self) -> None:
        """Pin the coincidence the LeRobot derivation currently rests on.

        The bullet is derived from ``hardware.lerobot_type`` rather than from the
        presence of a ``hardware`` block, and today every arm that has the block
        names a type - so the two readings pick the same set and the distinction
        is invisible. It is not invisible in general: ``reachy_mini`` declares
        ``{"driver": "strands"}`` with no type at all. When the first arm does
        that, this fails and says the two derivations have come apart, rather
        than the LeRobot bullet quietly gaining a robot LeRobot cannot build.
        """
        typeless = {
            name
            for name in _arm_names()
            if (_registry()[name].get("hardware") or {}) and not _arms_with_a_lerobot_type() & {name}
        }
        assert not typeless, (
            f"these arms declare a hardware block with no lerobot_type: {sorted(typeless)}. "
            "The LeRobot bullet is derived from the type, so they belong on the native-driver "
            "bullet (or on neither) - check which route each one actually has."
        )

    def test_every_declared_lerobot_type_is_one_lerobot_registers(self) -> None:
        """Declaring a ``lerobot_type`` must mean LeRobot can build it.

        The LeRobot bullet is derived from the registry alone so it grades on an
        install without LeRobot. This is the premise that makes the registry a
        sound stand-in: every type the arms declare is one LeRobot registers.
        """
        pytest.importorskip("lerobot", reason="lerobot is needed to read its robot-type registry")
        from lerobot.robots.config import RobotConfig

        from strands_robots import hardware_robot

        hardware_robot._ensure_lerobot_robots_registered()
        known = set(RobotConfig.get_known_choices())
        declared = {
            name: (_registry()[name].get("hardware") or {})["lerobot_type"] for name in _arms_with_a_lerobot_type()
        }
        unknown = {name: kind for name, kind in declared.items() if kind not in known}
        assert not unknown, f"registry declares lerobot types LeRobot does not register: {unknown}"


class TestEachCapabilityClaimMatchesTheRegistry:
    """The three set-membership claims, each against its source of truth."""

    def test_the_sim_asset_exception_names_every_arm_without_one(self) -> None:
        claimed = _claimed_arms(NO_SIM_ASSET_CLAIM)
        derived = _arms_without_a_sim_asset()
        assert claimed == derived, _report(claimed, derived, NO_SIM_ASSET_CLAIM[0])

    def test_the_lerobot_bullet_names_every_arm_declaring_a_lerobot_type(self) -> None:
        claimed = _claimed_arms(LEROBOT_CLAIM)
        derived = _arms_with_a_lerobot_type()
        assert claimed == derived, _report(claimed, derived, LEROBOT_CLAIM[0])

    def test_the_native_driver_bullet_names_every_arm_with_one(self) -> None:
        claimed = _claimed_arms(NATIVE_CLAIM)
        derived = _arms_with_a_native_driver()
        assert claimed == derived, _report(claimed, derived, NATIVE_CLAIM[0])


class TestTheClaimsAreConsistentWithEachOther:
    """Cross-checks that hold whichever names the bullets carry."""

    def test_no_arm_the_page_calls_sim_only_has_a_real_path(self) -> None:
        """The page says every unnamed arm is simulation-only, so no arm may be both."""
        real = _arms_with_a_lerobot_type() | _arms_with_a_native_driver()
        named = (_claimed_arms(LEROBOT_CLAIM) or set()) | (_claimed_arms(NATIVE_CLAIM) or set())
        assert real == named, (
            "docs/robots/arms.md says every arm it does not name is simulation-only, but the registry "
            f"gives a real-hardware path to {sorted(real - named)} and the page names {sorted(named - real)} "
            "without one."
        )

    def test_an_arm_with_no_sim_asset_has_a_real_hardware_path(self) -> None:
        """An arm with neither a sim asset nor a driver would be unusable either way."""
        real = _arms_with_a_lerobot_type() | _arms_with_a_native_driver()
        stranded = _arms_without_a_sim_asset() - real
        assert not stranded, f"arms with no sim asset and no real-hardware path: {sorted(stranded)}"


class TestTheRulesAreNotVacuous:
    """Constructed exemplars, so the rules are graded on a page that is wrong.

    After the fix the shipped page satisfies every rule, so it can no longer
    exercise a rejection - these drive the same comparison over text that is
    deliberately stale, including the exact wording this file was written for.
    """

    @staticmethod
    def _claimed_from(bullet: str) -> set[str]:
        """Apply the page's own extraction rule to one bullet of prose."""
        return _names_in(" ".join(bullet.split()))

    def test_the_wording_this_file_was_written_for_is_rejected(self) -> None:
        stale = (
            "`panda`, `so100`, and `ur5e` are also supported on real hardware via LeRobot. "
            "The rest are simulation-only at the moment."
        )
        claimed = self._claimed_from(stale)
        assert claimed == {"panda", "so100", "ur5e"}, claimed
        assert claimed != _arms_with_a_lerobot_type(), "the stale wording must not satisfy the LeRobot rule"

    def test_the_stale_sim_asset_wording_is_rejected(self) -> None:
        stale = "Exceptions: `hope_jr` and `omx` have no MuJoCo sim asset."
        assert self._claimed_from(stale) != _arms_without_a_sim_asset()

    def test_the_corrected_wording_is_accepted(self) -> None:
        fixed = "Exceptions: `hope_jr`, `omx` and `rebot_b601` declare no sim asset."
        assert self._claimed_from(fixed) == _arms_without_a_sim_asset()

    def test_the_claim_patterns_find_the_shipped_wording_too(self) -> None:
        """The patterns key on the subject, so a reword still grades the claim.

        Pinned against the exact bullets this file replaced: if a pattern only
        matched the new phrasing, a future reword would make the rule report
        "no such claim" instead of grading the names, and a wrong list would
        pass. The shipped page made no native-driver claim at all, which is why
        that pattern has no counterpart here.
        """
        shipped_sim_asset = "Exceptions: `hope_jr` and `omx` have no MuJoCo sim asset and require physical hardware."
        shipped_lerobot = "`panda`, `so100`, and `ur5e` are also supported on real hardware via LeRobot."
        assert NO_SIM_ASSET_CLAIM[1].search(shipped_sim_asset), "pattern misses the wording it replaced"
        assert LEROBOT_CLAIM[1].search(shipped_lerobot), "pattern misses the wording it replaced"
        assert not NATIVE_CLAIM[1].search(shipped_lerobot), "the shipped page made no native-driver claim"

    def test_an_incidental_code_span_is_not_read_as_a_claim(self) -> None:
        """``lerobot_type`` and ``driver="strands"`` are spans, not robot names."""
        bullet = 'names a `lerobot_type`, selected with `driver="strands"`: `koch`'
        assert self._claimed_from(bullet) == {"koch"}


def test_the_derived_sets_have_the_shape_the_page_describes() -> None:
    """Guard against a registry change that would make the page's structure wrong.

    The page presents LeRobot and native as two routes with an overlap. If one
    became empty, or every arm gained a path, the prose would need rewriting
    rather than relisting.
    """
    lerobot = _arms_with_a_lerobot_type()
    native = _arms_with_a_native_driver()
    assert lerobot, "no arm declares a lerobot_type - the LeRobot bullet has nothing to say"
    assert native, "no arm has a native driver - the native bullet has nothing to say"
    assert (lerobot | native) < _arm_names(), "every arm now has a real path - 'every other arm' is empty"
