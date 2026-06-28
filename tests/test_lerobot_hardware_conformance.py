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
