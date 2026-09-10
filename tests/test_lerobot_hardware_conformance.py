"""LeRobot hardware conformance - every ``hardware.lerobot_type`` in the strands
registry must name a robot type that LeRobot actually registers.

This closes the gap left by ``test_registry_integrity.test_hardware_only_robots
_declare_lerobot_type``, which only checks that ``lerobot_type`` is a non-empty
string. That weaker check passed four entries (hope_jr, omx, bi_openarm,
earthrover) whose ``lerobot_type`` did NOT match any LeRobot
``@RobotConfig.register_subclass`` name, so ``Robot(name, mode="real")`` raised
``ValueError: Unsupported robot type`` at runtime instead of at test time.

Source-of-truth: the set of names passed to ``@RobotConfig.register_subclass``
in the vendored LeRobot tree. We parse the decorators from source rather than
importing each driver because several drivers only register when an optional
hardware SDK is installed (e.g. the reBot B601 needs ``motorbridge``); the type
is still a valid LeRobot choice, it just won't appear in a live
``RobotConfig.get_known_choices()`` on a CI host without that SDK. Parsing source
keeps the test dependency-free and deterministic.

If the vendored LeRobot tree is not present (pip-installed lerobot, or a slim
checkout), the test self-skips rather than failing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REGISTRY_PATH = Path(__file__).parent.parent / "strands_robots" / "registry" / "robots.json"

# Candidate locations for the LeRobot robots package (vendored sibling checkout
# first, then an installed distribution).
_LEROBOT_ROOTS = [
    Path(__file__).parent.parent.parent / "lerobot" / "src" / "lerobot" / "robots",
    Path(__file__).parent.parent / "lerobot" / "src" / "lerobot" / "robots",
]

_REGISTER_RE = re.compile(r"""@RobotConfig\.register_subclass\(\s*["']([a-zA-Z0-9_]+)["']""")


def _find_lerobot_robots_dir() -> Path | None:
    for root in _LEROBOT_ROOTS:
        if root.is_dir():
            return root
    # Fall back to an installed lerobot.
    try:
        import lerobot.robots as _lr  # type: ignore

        return Path(next(iter(_lr.__path__)))
    except Exception:
        return None


def _lerobot_registered_types(robots_dir: Path) -> set[str]:
    types: set[str] = set()
    for cfg in robots_dir.rglob("config*.py"):
        types |= set(_REGISTER_RE.findall(cfg.read_text(encoding="utf-8", errors="ignore")))
    for cfg in robots_dir.rglob("configuration*.py"):
        types |= set(_REGISTER_RE.findall(cfg.read_text(encoding="utf-8", errors="ignore")))
    return types


@pytest.fixture(scope="module")
def strands_hw_types() -> dict[str, str]:
    """Map strands robot name -> declared lerobot_type (hardware entries only).

    Excludes entries with ``requires_lerobot_from_source: true`` since those
    reference LeRobot types not yet available in any PyPI release within the
    pinned version range. They are valid on a lerobot-from-source install.
    """
    data = json.loads(REGISTRY_PATH.read_text())
    robots = data.get("robots", data)
    return {
        name: info["hardware"]["lerobot_type"]
        for name, info in robots.items()
        if isinstance(info.get("hardware"), dict)
        and info["hardware"].get("lerobot_type")
        and not info["hardware"].get("requires_lerobot_from_source")
    }


@pytest.fixture(scope="module")
def strands_hw_types_all() -> dict[str, str]:
    """Map strands robot name -> declared lerobot_type for ALL hardware entries.

    Unlike ``strands_hw_types`` this keeps entries flagged
    ``requires_lerobot_from_source``. Those entries deliberately reference
    LeRobot types that only exist in a lerobot-from-source install (not in a
    PyPI release within the pinned range). When the test runs against such a
    source tree, ``lerobot_types`` includes those types, so the strands entries
    that cover them must count as reachable - otherwise the coverage direction
    reports a false ``missing`` for a robot that is, in fact, drivable.
    """
    data = json.loads(REGISTRY_PATH.read_text())
    robots = data.get("robots", data)
    return {
        name: info["hardware"]["lerobot_type"]
        for name, info in robots.items()
        if isinstance(info.get("hardware"), dict) and info["hardware"].get("lerobot_type")
    }


@pytest.fixture(scope="module")
def lerobot_types() -> set[str]:
    robots_dir = _find_lerobot_robots_dir()
    if robots_dir is None:
        pytest.skip("LeRobot robots package not found (sim-only / no [lerobot] extra)")
    types = _lerobot_registered_types(robots_dir)
    if not types:
        pytest.skip(f"No @RobotConfig.register_subclass found under {robots_dir}")
    return types


def test_every_strands_lerobot_type_is_real(strands_hw_types: dict[str, str], lerobot_types: set[str]) -> None:
    """Each strands ``hardware.lerobot_type`` must be a registered LeRobot choice.

    Regression guard for hope_jr/omx/bi_openarm/earthrover, whose lerobot_type
    was a made-up shorthand that broke ``Robot(name, mode="real")``.
    """
    offenders = {name: lt for name, lt in strands_hw_types.items() if lt not in lerobot_types}
    assert not offenders, (
        "Robots whose hardware.lerobot_type is not a real LeRobot register_subclass "
        f"name (Robot(name, mode='real') would raise): {offenders}. "
        f"Valid LeRobot types: {sorted(lerobot_types)}"
    )


def test_full_lerobot_hardware_coverage(strands_hw_types_all: dict[str, str], lerobot_types: set[str]) -> None:
    """Every LeRobot robot type is reachable through at least one strands entry.

    This is the 'conform to all the toys' invariant: if LeRobot ships a robot,
    a user can drive it with ``Robot(name, mode='real')`` via some strands name.

    Uses ``strands_hw_types_all`` (not the reality-check subset) so that
    from-source-only LeRobot types - which appear in ``lerobot_types`` when the
    test runs against a source checkout - are matched by the strands entries
    that carry ``requires_lerobot_from_source``.
    """
    reachable = set(strands_hw_types_all.values())
    missing = lerobot_types - reachable
    assert not missing, (
        "LeRobot robot types not reachable from any strands registry entry "
        f"(add a hardware entry with this lerobot_type): {sorted(missing)}"
    )


def test_from_source_entries_are_documented() -> None:
    """Entries with ``requires_lerobot_from_source`` must declare the expected type.

    This guards against adding a from-source entry without documenting which
    unreleased lerobot_type it maps to. When the relevant LeRobot release lands
    on PyPI, remove the flag and the entries rejoin the standard conformance check.
    """
    data = json.loads(REGISTRY_PATH.read_text())
    robots = data.get("robots", data)
    from_source = {
        name: info["hardware"].get("lerobot_type")
        for name, info in robots.items()
        if isinstance(info.get("hardware"), dict) and info["hardware"].get("requires_lerobot_from_source")
    }
    # Each from-source entry must still declare a lerobot_type
    missing_type = {name for name, lt in from_source.items() if not lt}
    assert not missing_type, (
        "Entries with requires_lerobot_from_source=true must still declare "
        f"hardware.lerobot_type: {sorted(missing_type)}"
    )
    # Sanity: at least verify the expected entries are present
    assert "rebot_b601" in from_source
    assert "bi_rebot_b601" in from_source


# ---------------------------------------------------------------------------
# Motor family. ``test_every_strands_lerobot_type_is_real`` asks whether a
# declared ``lerobot_type`` is a name lerobot registers, and that is all it can
# ask: ``aloha`` ("2x ViperX 300s") declared ``bi_so_follower``, a real name, and
# passed for the registry's whole history while building a Feetech bus for
# Dynamixel servos. A servo family is a wire protocol - Feetech is a half-duplex
# TTL bus, Dynamixel is Protocol 2.0 - so the two are not configurable into each
# other. A description naming one while its lerobot_type drives the other
# describes a robot the mapping cannot move, and both facts are already in the
# tree, so the contradiction is readable without hardware.
# ---------------------------------------------------------------------------

_BUS_RE = re.compile(r"\b(FeetechMotorsBus|DynamixelMotorsBus)\b")

#: A bimanual package composes single arms rather than holding a bus, so the
#: scan follows ``from ..so_follower import`` to where the bus is built.
_SIBLING_IMPORT_RE = re.compile(r"^from \.\.([a-z0-9_]+) import", re.MULTILINE)

#: Words that name a motor family unambiguously: a bus, a protocol, or a servo
#: part number. Vendor names are deliberately absent - Trossen sells ViperX and
#: WidowX arms on Dynamixel *and* supplies the MuJoCo model this registry loads
#: for the Feetech SO-ARM100 ("TrossenRobotics SO-ARM100 (6-DOF, Feetech
#: servos)"), so "Trossen" names no family and grading on it would report that
#: correct entry as a contradiction.
_FAMILY_MARKERS: dict[str, tuple[str, ...]] = {
    "DynamixelMotorsBus": ("dynamixel", "viperx", "widowx", "xm430", "xm540", "xl330", "xl430"),
    "FeetechMotorsBus": ("feetech", "sts3215", "sts3032", "scs0009"),
}


def _buses_in_package(pkg: Path, seen: frozenset[Path] = frozenset()) -> set[str]:
    """Report the motor-bus classes ``pkg`` builds, following sibling re-exports.

    Args:
        pkg: A lerobot robot package directory.
        seen: Packages already visited, so a cyclic re-export terminates.

    Returns:
        The bus class names found, empty for a robot on neither bus (CAN,
        DDS, ZMQ).
    """
    if pkg in seen:
        return set()
    seen = seen | {pkg}
    found: set[str] = set()
    for py in sorted(pkg.glob("*.py")):
        source = py.read_text(encoding="utf-8", errors="ignore")
        found |= set(_BUS_RE.findall(source))
        for sibling in _SIBLING_IMPORT_RE.findall(source):
            candidate = pkg.parent / sibling
            if candidate.is_dir():
                found |= _buses_in_package(candidate, seen)
    return found


@pytest.fixture(scope="module")
def lerobot_bus_by_type() -> dict[str, str]:
    """Map lerobot robot type -> the single motor bus its package builds.

    A package building two buses is left out rather than guessed at: there is no
    one family for such a type, so no description of it can contradict one.
    """
    robots_dir = _find_lerobot_robots_dir()
    if robots_dir is None:
        pytest.skip("LeRobot robots package not found (sim-only / no [lerobot] extra)")
    by_type: dict[str, str] = {}
    configs = sorted(robots_dir.rglob("config*.py")) + sorted(robots_dir.rglob("configuration*.py"))
    for cfg in configs:
        registered = _REGISTER_RE.findall(cfg.read_text(encoding="utf-8", errors="ignore"))
        if not registered:
            continue
        buses = _buses_in_package(cfg.parent)
        if len(buses) == 1:
            for name in registered:
                by_type[name] = next(iter(buses))
    return by_type


@pytest.fixture(scope="module")
def strands_descriptions() -> dict[str, str]:
    """Map strands robot name -> its registry description."""
    data = json.loads(REGISTRY_PATH.read_text())
    robots = data.get("robots", data)
    return {name: str(info.get("description", "")) for name, info in robots.items()}


def _family_contradictions(
    hw_types: dict[str, str],
    descriptions: dict[str, str],
    bus_by_type: dict[str, str],
) -> dict[str, str]:
    """Report entries describing one servo family while declaring the other's type.

    Args:
        hw_types: Strands robot name -> declared ``hardware.lerobot_type``.
        descriptions: Strands robot name -> registry description.
        bus_by_type: lerobot type -> the motor bus its package builds.

    Returns:
        Offending robot name -> what it describes against what it drives. A type
        whose bus is unknown is not graded: there is no family to contradict.
    """
    contradictions: dict[str, str] = {}
    for name, lerobot_type in sorted(hw_types.items()):
        bus = bus_by_type.get(lerobot_type)
        if bus is None:
            continue
        description = descriptions.get(name, "").lower()
        for family, markers in _FAMILY_MARKERS.items():
            if family == bus:
                continue
            named = [marker for marker in markers if marker in description]
            if named:
                contradictions[name] = f"describes {named} but {lerobot_type} builds a {bus}"
    return contradictions


def test_a_declared_lerobot_type_drives_the_motor_family_the_description_names(
    strands_hw_types_all: dict[str, str],
    strands_descriptions: dict[str, str],
    lerobot_bus_by_type: dict[str, str],
) -> None:
    """No entry may describe one servo family and declare a type driving the other.

    The regression for ``aloha``, whose description named ViperX 300s (Dynamixel
    XM540/XM430) while ``hardware.lerobot_type`` was ``bi_so_follower`` - two
    SO-ARM followers on a Feetech STS3215 bus. ``Robot("aloha", mode="real")``
    built that bus and would have written Feetech packets to Dynamixel servos.
    """
    contradictions = _family_contradictions(strands_hw_types_all, strands_descriptions, lerobot_bus_by_type)
    assert not contradictions, (
        "Registry entries whose description names one motor family while their "
        "hardware.lerobot_type drives the other. A servo family is a wire "
        "protocol, so mode='real' would speak the wrong one to the servos: "
        f"{contradictions}"
    )


def test_the_motor_family_derivation_grades_something(
    strands_hw_types_all: dict[str, str],
    strands_descriptions: dict[str, str],
    lerobot_bus_by_type: dict[str, str],
) -> None:
    """Non-vacuity: the rule above passes trivially if it derives nothing.

    Three ways it could go quiet, each pinned: neither bus recognised, the
    bimanual sibling hop lost (``bi_so_follower`` builds no bus of its own, so
    without it every bimanual entry grades as unknown - which is the entry the
    rule exists for), and no description naming a family at all.
    """
    assert set(lerobot_bus_by_type.values()) == set(_FAMILY_MARKERS), (
        f"expected both motor families among lerobot's robots, got {sorted(set(lerobot_bus_by_type.values()))}"
    )
    assert lerobot_bus_by_type.get("bi_so_follower") == "FeetechMotorsBus", (
        "bi_so_follower composes two so_follower arms, so its bus is only reachable through the sibling-import hop"
    )
    claiming = {
        name
        for name, lerobot_type in strands_hw_types_all.items()
        if (bus := lerobot_bus_by_type.get(lerobot_type))
        and any(marker in strands_descriptions.get(name, "").lower() for marker in _FAMILY_MARKERS[bus])
    }
    assert claiming, "no registry description names the motor family its lerobot_type drives"


#: ``aloha`` exactly as it stood before this rule existed: a ViperX description
#: declaring the lerobot type for two Feetech SO arms.
_ALOHA_BEFORE = ("ALOHA Bimanual (2x ViperX 300s, 14-DOF + 2 grippers)", "bi_so_follower")


def test_the_rule_reports_the_entry_that_motivated_it(lerobot_bus_by_type: dict[str, str]) -> None:
    """Grade the rule, not the registry: a correct registry passes any rule.

    The live-registry cell above goes quiet the moment the registry is right,
    so on its own it pins nothing about which words the rule recognises. Here
    the offending entry is supplied, so narrowing ``_FAMILY_MARKERS`` until it
    no longer reads "ViperX" as Dynamixel fails.
    """
    description, lerobot_type = _ALOHA_BEFORE
    reported = _family_contradictions({"aloha": lerobot_type}, {"aloha": description}, lerobot_bus_by_type)
    assert "aloha" in reported, (
        f"the rule must report {description!r} declaring {lerobot_type!r}; "
        f"got {reported} - _FAMILY_MARKERS no longer recognises this description"
    )


def test_a_correctly_described_entry_is_not_reported(lerobot_bus_by_type: dict[str, str]) -> None:
    """Over-reach: the vendor that supplies both families is not a family claim.

    ``so100`` reads "TrossenRobotics SO-ARM100 (6-DOF, Feetech servos)" - Trossen
    sells Dynamixel ViperX arms and supplies this Feetech arm's MuJoCo model, so
    a rule counting the vendor as a Dynamixel marker reports a correct entry.
    """
    reported = _family_contradictions(
        {"so100": "so100_follower"},
        {"so100": "TrossenRobotics SO-ARM100 (6-DOF, Feetech servos)"},
        lerobot_bus_by_type,
    )
    assert reported == {}, f"a correct entry was reported as contradictory: {reported}"
