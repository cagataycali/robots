"""Repo hygiene: the arms family's capability claims agree with the registry.

``docs/robots/arm/index.md`` used to end in a hand-written *Compatibility notes*
block: which arms have no sim asset, which drive real hardware through LeRobot,
which through a native driver. All three bullets went stale at once (``panda``
and ``ur5e`` were called LeRobot arms when LeRobot registers no Franka or UR
type; ``rebot_b601`` had no asset and was unlisted), so this file was written
to grade them.

The page is generated now. ``docs/hooks/robot_pages.py`` expands
``{{robot_family_table:arm}}`` into one row per arm with ``Sim``, ``Real`` and
``Drivers`` cells, and writes each arm's own ``docs/robots/<arm>.md`` with the
same facts as chips (``sr-chip-sim``, ``sr-chip-real``, ``driver: ...``). The
three claims survive as columns instead of bullets, and a reader still plans
hardware work from them, so a wrong cell is a wrong answer rather than a
cosmetic one. Both are graded against the repo's own sources of truth rather
than a restated list, so an arm that gains a driver fails the cell that should
have said so:

* LeRobot: the entry declares ``hardware.lerobot_type``.
  :func:`test_every_declared_lerobot_type_is_one_lerobot_registers` is the
  premise that makes declaring one mean the path works.
* Native: :func:`strands_robots.drivers.registry.get_native_driver_class`
  answers for the name.
* Sim: the entry declares an ``asset`` block.

Deliberately out of scope: the ``Joints`` column. Of the registry robots whose
asset compiles here, most declare a ``joints`` value other than the asset's
actuator count, so that field is a loose informational number whose contract
needs deciding before it can be graded, a different question from which arms
drive hardware.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_REPO = REPO_ROOT
ROBOTS_DIR = REPO_ROOT / "docs" / "robots"
ARMS_PAGE = ROBOTS_DIR / "arm" / "index.md"
HOOK = REPO_ROOT / "docs" / "hooks" / "robot_pages.py"
TABLE_TOKEN = "{{robot_family_table:arm}}"

#: One rendered row: robot link, description, joints, Sim, Real, Drivers.
_ROW = re.compile(r"^\| \[`(?P<robot>[a-z0-9_]+)`\]\([^)]*\) \| (?P<rest>.*) \|$", re.M)
_CODE = re.compile(r"`([^`]+)`")


def _hook():
    """The robot pages hook, loaded from the docs tree the build loads it from."""
    name = "docs_robot_pages_hook"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _registry() -> dict[str, dict]:
    """Every shipped robot: ``robots.json`` plus the ``robot_descriptions`` URDF tail (``registry_view.py``)."""
    return dict(_registry_view().merged())


def _registry_view():  # noqa: ANN202 - the docs hook, loaded by path: the docs venv is not the test venv
    """``docs/hooks/registry_view.py``: robots.json merged with the robot_descriptions URDF tail."""
    import importlib.util
    import sys

    name = "docs_hooks_registry_view"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _REPO / "docs" / "hooks" / "registry_view.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _arm_names() -> set[str]:
    """Return every registry name whose category is ``arm``."""
    return {name for name, entry in _registry().items() if entry.get("category") == "arm"}


def _rows(rendered: str) -> dict[str, dict[str, str]]:
    """Parse a rendered family table into robot -> {sim, real, drivers}."""
    out: dict[str, dict[str, str]] = {}
    for match in _ROW.finditer(rendered):
        cells = [c.strip() for c in match.group("rest").split("|")]
        assert len(cells) == 5, f"{match.group('robot')} row has {len(cells) + 1} cells, expected 6"
        _description, _joints, sim, real, drivers = cells
        out[match.group("robot")] = {"sim": sim, "real": real, "drivers": drivers}
    return out


def _published_rows() -> dict[str, dict[str, str]]:
    """The arms table as the build renders it from the page's token."""
    page = ARMS_PAGE.read_text(encoding="utf-8")
    assert TABLE_TOKEN in page, f"docs/robots/arm/index.md no longer carries {TABLE_TOKEN}"
    rendered = _hook().substitute(page, "../../", "../")
    assert TABLE_TOKEN not in rendered, "robot_pages.py left the family table token unexpanded"
    return _rows(rendered)


def _claimed(column: str, value: str | None = None) -> set[str]:
    """Arms whose ``column`` cell equals ``value``, or whose Drivers cell names ``column``."""
    rows = _published_rows()
    if column == "drivers":
        assert value is not None
        return {name for name, cells in rows.items() if value in _CODE.findall(cells["drivers"])}
    return {name for name, cells in rows.items() if cells[column] == value}


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


def _report(claimed: set[str], derived: set[str], what: str) -> str:
    """Return a failure message naming the exact edit the table needs."""
    return (
        f"docs/robots/arm/index.md: the {what} column marks {sorted(claimed)} but the registry says {sorted(derived)}.\n"
        f"  missing from the page: {sorted(derived - claimed)}\n"
        f"  marked but not true:   {sorted(claimed - derived)}"
    )


class TestTheTableIsShapedTheWayTheRulesAssume:
    """Premises. Each rule below reads a set off the table; these pin that it can."""

    def test_the_table_has_one_row_per_arm(self) -> None:
        rows = _published_rows()
        assert set(rows) == _arm_names(), (
            f"rows the registry does not hold as arms: {sorted(set(rows) - _arm_names())}; "
            f"arms with no row: {sorted(_arm_names() - set(rows))}"
        )

    def test_the_registry_declares_arms(self) -> None:
        assert len(_arm_names()) > 10, f"only {len(_arm_names())} arms, the rules below would be near-vacuous"

    def test_every_cell_is_yes_or_dash(self) -> None:
        for name, cells in _published_rows().items():
            assert cells["sim"] in {"yes", "-"}, f"{name} Sim cell is {cells['sim']!r}"
            assert cells["real"] in {"yes", "-"}, f"{name} Real cell is {cells['real']!r}"
            drivers = set(_CODE.findall(cells["drivers"]))
            assert drivers <= {"lerobot", "strands"}, f"{name} Drivers cell names {drivers}"
            assert (cells["real"] == "yes") == bool(drivers), f"{name}: Real and Drivers disagree"

    def test_every_arm_declaring_hardware_declares_a_lerobot_type(self) -> None:
        """Pin the coincidence the LeRobot derivation currently rests on.

        The LeRobot claim is derived from ``hardware.lerobot_type`` rather than
        from the presence of a ``hardware`` block, and today every arm that has
        the block names a type, so the two readings pick the same set and the
        distinction is invisible. It is not invisible in general:
        ``reachy_mini`` declares ``{"driver": "strands"}`` with no type at all.
        When the first arm does that, this fails and says the two derivations
        have come apart, rather than the LeRobot column quietly gaining a robot
        LeRobot cannot build.
        """
        typeless = {
            name
            for name in _arm_names()
            if (_registry()[name].get("hardware") or {}) and not _arms_with_a_lerobot_type() & {name}
        }
        assert not typeless, (
            f"these arms declare a hardware block with no lerobot_type: {sorted(typeless)}. "
            "The lerobot driver cell is derived from the type, so they belong on the native "
            "driver (or on neither): check which route each one actually has."
        )

    def test_every_declared_lerobot_type_is_one_lerobot_registers(self) -> None:
        """Declaring a ``lerobot_type`` must mean LeRobot can build it.

        The LeRobot column is derived from the registry alone so it grades on an
        install without LeRobot. This is the premise that makes the registry a
        sound stand-in: every type the arms declare is one LeRobot registers.
        """
        pytest.importorskip("lerobot", reason="lerobot is needed to read its robot-type registry")
        from lerobot.robots.config import RobotConfig

        from strands_robots.utils import ensure_lerobot_family_registered

        ensure_lerobot_family_registered("robots")
        known = set(RobotConfig.get_known_choices())
        declared = {
            name: (_registry()[name].get("hardware") or {})["lerobot_type"] for name in _arms_with_a_lerobot_type()
        }
        unknown = {name: kind for name, kind in declared.items() if kind not in known}
        assert not unknown, f"registry declares lerobot types LeRobot does not register: {unknown}"


class TestEachCapabilityClaimMatchesTheRegistry:
    """The three set-membership claims, each against its source of truth."""

    def test_the_sim_column_marks_every_arm_without_an_asset_as_dash(self) -> None:
        claimed = _claimed("sim", "-")
        derived = _arms_without_a_sim_asset()
        assert claimed == derived, _report(claimed, derived, "Sim (dash)")

    def test_the_lerobot_driver_cell_names_every_arm_declaring_a_lerobot_type(self) -> None:
        claimed = _claimed("drivers", "lerobot")
        derived = _arms_with_a_lerobot_type()
        assert claimed == derived, _report(claimed, derived, "Drivers (`lerobot`)")

    def test_the_strands_driver_cell_names_every_arm_with_a_native_driver(self) -> None:
        claimed = _claimed("drivers", "strands")
        derived = _arms_with_a_native_driver()
        assert claimed == derived, _report(claimed, derived, "Drivers (`strands`)")


class TestTheClaimsAreConsistentWithEachOther:
    """Cross-checks that hold whichever names the cells carry."""

    def test_no_arm_the_page_calls_sim_only_has_a_real_path(self) -> None:
        """A dash in Real says simulation-only, so no such arm may have a driver."""
        real = _arms_with_a_lerobot_type() | _arms_with_a_native_driver()
        marked = _claimed("real", "yes")
        assert real == marked, (
            "docs/robots/arm/index.md marks an arm with a dash in Real as simulation-only, but the registry "
            f"gives a real-hardware path to {sorted(real - marked)} and the page marks {sorted(marked - real)} "
            "without one."
        )

    def test_an_arm_with_no_sim_asset_has_a_real_hardware_path(self) -> None:
        """An arm with neither a sim asset nor a driver would be unusable either way."""
        real = _arms_with_a_lerobot_type() | _arms_with_a_native_driver()
        stranded = _arms_without_a_sim_asset() - real
        assert not stranded, f"arms with no sim asset and no real-hardware path: {sorted(stranded)}"


class TestEachArmPageCarriesTheSameFacts:
    """The chips on ``docs/robots/<arm>.md`` say what the family table says."""

    @pytest.mark.parametrize("name", sorted(_arm_names()))
    def test_the_arm_page_chips_match_the_registry(self, name: str) -> None:
        page = ROBOTS_DIR / f"{name}.md"
        assert page.is_file(), f"docs/robots/{name}.md is not generated"
        text = page.read_text(encoding="utf-8")
        has_asset = bool(_registry()[name].get("asset"))
        lerobot = name in _arms_with_a_lerobot_type()
        native = name in _arms_with_a_native_driver()
        assert ('class="sr-chip sr-chip-sim"' in text) == has_asset, f"{name}: sim chip disagrees with the asset block"
        assert ('class="sr-chip sr-chip-real"' in text) == (lerobot or native), (
            f"{name}: real chip disagrees with the drivers"
        )
        expected = ", ".join(d for d, on in (("lerobot", lerobot), ("strands", native)) if on)
        if expected:
            assert f"driver: {expected}</span>" in text, f"{name}: driver chip does not read 'driver: {expected}'"
        else:
            assert "sr-chip-driver" not in text, f"{name}: driver chip on a simulation-only arm"


class TestTheRulesAreNotVacuous:
    """Constructed exemplars, so the rules are graded on a table that is wrong.

    The shipped table satisfies every rule, so it can no longer exercise a
    rejection; these drive the same parser over rows that are deliberately
    stale, including the exact claim this file was written for.
    """

    def test_the_claim_this_file_was_written_for_is_rejected(self) -> None:
        stale = (
            "| [`panda`](../panda.md) | Franka | 7 | yes | yes | `lerobot` |\n"
            "| [`ur5e`](../ur5e.md) | UR | 6 | yes | yes | `lerobot` |\n"
            "| [`so100`](../so100.md) | SO-100 | 6 | yes | yes | `lerobot` |\n"
        )
        rows = _rows(stale)
        claimed = {n for n, c in rows.items() if "lerobot" in _CODE.findall(c["drivers"])}
        assert claimed == {"panda", "so100", "ur5e"}, claimed
        assert claimed != _arms_with_a_lerobot_type(), "the stale claim must not satisfy the LeRobot rule"

    def test_a_missing_sim_dash_is_rejected(self) -> None:
        stale = "| [`rebot_b601`](../rebot_b601.md) | ReBot | 6 | yes | yes | `lerobot` |\n"
        rows = _rows(stale)
        assert "rebot_b601" in _arms_without_a_sim_asset()
        assert rows["rebot_b601"]["sim"] != "-", "the exemplar must mark a sim asset the registry lacks"

    def test_the_parser_reads_a_correct_row(self) -> None:
        good = "| [`koch`](../koch.md) | Koch v1.1 | 7 | yes | yes | `lerobot` |\n"
        assert _rows(good) == {"koch": {"sim": "yes", "real": "yes", "drivers": "`lerobot`"}}


def test_the_derived_sets_have_the_shape_the_page_describes() -> None:
    """Guard against a registry change that would make the page's structure wrong.

    The table presents LeRobot and native as two routes with an overlap. If one
    became empty, or every arm gained a path, the prose around it would need
    rewriting rather than regenerating.
    """
    lerobot = _arms_with_a_lerobot_type()
    native = _arms_with_a_native_driver()
    assert lerobot, "no arm declares a lerobot_type, the lerobot column has nothing to say"
    assert native, "no arm has a native driver, the strands column has nothing to say"
    assert (lerobot | native) < _arm_names(), "every arm now has a real path, 'simulation-only' is empty"
