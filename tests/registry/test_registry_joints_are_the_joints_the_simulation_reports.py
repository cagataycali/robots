"""A simulated robot's registry ``joints`` is the count ``get_robot_state`` reports.

Issue #4147: the ``joints`` field in ``robots.json``, printed as the ``N joints``
chip on every robot page and as the ``Joints`` column of ``list_robots()``,
disagreed with the loaded MuJoCo model for 47 of the 66 simulated robots
(``unitree_go2`` said 40, the model has 12; ``leap_hand`` 41 against 16;
``shadow_hand`` 45 against 24; ``unitree_g1`` 46 against 29; 24 more counted a
floating base as a joint). The figure followed no rule, so nothing graded it.

The rule now: ``joints`` is the number of joints ``Robot(name).get_robot_state()``
reports, which is every joint of the model the robot loads except a free
floating base (reported as ``base``, not as a joint). Ball and passive joints
are reported, so they count. ``scripts/audit_registry_joints.py`` implements the
count and rewrites the registry; this module runs the same comparison.

Two layers, because the oracle is not on every machine:

* the per-robot cells compile the model the robot loads and compare, skipping a
  robot whose asset is not on disk rather than downloading the corpus
  (``allow_download=False``, the same refusal
  ``tests/registry/test_asset_family_joint_counts.py`` grades);
* the pinned cells read ``robots.json`` alone, so the corrected entries the issue
  named cannot drift back on an install with no MuJoCo and no assets.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO / "strands_robots" / "registry" / "robots.json"
SCRIPT = REPO / "scripts" / "audit_registry_joints.py"

#: The entries issue #4147 measured, with the count their loaded model reports.
CORRECTED: dict[str, tuple[int, str]] = {
    "unitree_go2": (12, "twelve hinge joints; the floating base is reported as base, not as a joint"),
    "leap_hand": (16, "sixteen hinge joints, one per actuator"),
    "shadow_hand": (24, "twenty-four hinge joints, four of them passive couplings"),
    "unitree_g1": (29, "twenty-nine hinge joints below the floating base"),
    "openarm": (18, "the scene the robot loads holds both arms, nine joints each"),
    "go1": (12, "twelve hinge joints; the floating base is not a reported joint"),
    "crazyflie": (0, "a quadrotor: its only joint is the free base, so no joint is reported"),
}


def _load_script() -> ModuleType:
    """Import the audit script; ``scripts/`` is not a package, so it is reached by path.

    Registered in ``sys.modules`` before it runs because the script defines a
    dataclass, and ``dataclasses`` looks the defining module up by name.
    """
    name = "audit_registry_joints"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def registry() -> dict[str, Any]:
    """The shipped robot registry, once."""
    return dict(json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["robots"])


@pytest.fixture
def host_asset_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read the machine's real asset cache; the registry conftest empties it per test.

    The cache is ``$STRANDS_ASSETS_DIR`` or ``$STRANDS_BASE_DIR/assets``, and the
    conftest repoints both at a temp dir, so both go back to the host's values.
    """
    monkeypatch.delenv("STRANDS_ASSETS_DIR", raising=False)
    monkeypatch.delenv("STRANDS_BASE_DIR", raising=False)


def _simulated() -> list[str]:
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["robots"]
    return sorted(name for name, spec in data.items() if spec.get("asset"))


class TestTheCorrectedEntriesStayCorrected:
    """Registry-only cells: the values the issue measured, pinned."""

    @pytest.mark.parametrize(("name", "expected", "why"), [(n, *v) for n, v in CORRECTED.items()])
    def test_the_entry_declares_what_its_model_reports(
        self, name: str, expected: int, why: str, registry: dict[str, Any]
    ) -> None:
        """One number per robot, the one ``get_robot_state`` reports."""
        assert registry[name].get("joints") == expected, f"{name} declares {registry[name].get('joints')!r}; {why}"

    def test_every_simulated_robot_declares_an_integer(self, registry: dict[str, Any]) -> None:
        """A missing or non-integer count would make the chip and the column silent, not wrong."""
        bad = [n for n in _simulated() if not isinstance(registry[n].get("joints"), int)]
        assert not bad, f"simulated robots without an integer joints count: {bad}"

    def test_the_count_reaches_the_discovery_surface(self) -> None:
        """The figure is reported, so a wrong one is read rather than inert."""
        from strands_robots.registry import get_robot, list_robots

        entry = get_robot("unitree_go2")
        assert entry is not None
        assert entry["joints"] == 12
        listed = {r["name"]: r["joints"] for r in list_robots()}
        assert listed["unitree_go2"] == 12
        assert listed["unitree_g1"] == 29


class TestTheRegistryMatchesTheLoadedModel:
    """Per-robot comparison against the model the robot loads, where it is on disk."""

    @pytest.mark.usefixtures("host_asset_cache")
    @pytest.mark.parametrize("name", _simulated())
    def test_joints_is_the_count_the_simulation_reports(self, name: str, registry: dict[str, Any]) -> None:
        """Compile the robot's model and count every joint but a free base."""
        pytest.importorskip("mujoco")
        script = _load_script()
        rows = script.audit({name: registry[name]}, allow_download=False)
        assert len(rows) == 1
        row = rows[0]
        if row.reported is None:
            pytest.skip(f"{name}: {row.note}")
        assert row.declared == row.reported, (
            f"{name} declares joints={row.declared} but its loaded model reports {row.reported} joints; "
            "run scripts/audit_registry_joints.py --write"
        )

    def test_the_rule_counts_every_joint_but_a_free_base(self, tmp_path: Path) -> None:
        """The counting rule on a model with one of each joint type."""
        mujoco = pytest.importorskip("mujoco")
        xml = tmp_path / "model.xml"
        xml.write_text(
            "<mujoco><worldbody>"
            "<body><freejoint/><geom size='0.1'/>"
            "<body><joint type='hinge'/><geom size='0.1'/>"
            "<body><joint type='slide'/><geom size='0.1'/>"
            "<body><joint type='ball'/><geom size='0.1'/></body></body></body></body>"
            "</worldbody></mujoco>",
            encoding="utf-8",
        )
        assert mujoco.MjModel.from_xml_path(str(xml)).njnt == 4
        assert _load_script().reported_joint_count(str(xml)) == 3

    def test_the_audit_never_downloads_by_default(
        self, registry: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An absent asset is a skip, not a fetch of the corpus (the conftest's empty cache is that case)."""
        pytest.importorskip("mujoco")
        from strands_robots.assets import manager

        attempted: list[str] = []

        def refuse(name: str, info: dict[str, Any]) -> bool:
            attempted.append(name)
            return False

        monkeypatch.setattr(manager, "_auto_download_robot", refuse)
        rows = _load_script().audit({n: registry[n] for n in ("unitree_go2", "so101")})
        assert not attempted, f"the audit tried to download {attempted}"
        assert all(r.note or r.reported is not None for r in rows)
