"""Every backend declares its recorded action columns from ``robot_action_keys``.

``DatasetRecorder`` resolves the ``action`` feature's column names from the
``action_names`` it is handed, and falls back to ``joint_names`` when a backend
passes none (``dataset_recorder.py``, the ``elif joint_names:`` branch). That
fallback is silent, and a backend taking it declares its action columns under
the *joint* vocabulary while its recording hook emits the *actuator* vocabulary
that ``robot_action_keys`` defines.

Nothing reports the mismatch. ``add_frame`` reads the action dict by declared
name, so a declared column the hook never emits is not an error - it takes the
``0.0`` fill and the frame records a command nobody issued, under a success
result. That is the #1715 fabrication reached without any narrow policy and
without a multi-robot scene: one robot, one rollout, every action column zero.

Newton was the backend on that fallback. Its schema builder and
``robot_action_keys`` both excluded the floating base's free joint, each from
its own copy of the rule, so the two vocabularies happened to agree and the
fallback happened to be correct - agreement by coincidence, with nothing
holding it. These tests pin the authority rather than the coincidence.

Runs solver-free: the schema builder is pure over the world model, so no
``newton`` / ``warp`` / ``lerobot`` install is required.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from strands_robots.simulation.models import SimRobot, SimWorld
from strands_robots.simulation.newton.simulation import NewtonSimEngine

_SO100_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
_FREE_BASE = "floating_base_joint"

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _world(*robots: str, joints: list[str] | None = None) -> SimWorld:
    world = SimWorld()
    for name in robots:
        world.robots[name] = SimRobot(
            name=name,
            urdf_path="so100.xml",
            data_config="so100",
            joint_names=list(joints if joints is not None else _SO100_JOINTS),
        )
    return world


def _engine(world: SimWorld, free_base: dict[str, str] | None = None) -> Any:
    """A solver-free ``NewtonSimEngine`` over ``world``.

    ``__new__`` rather than ``__init__``: the schema builder is the only surface
    under test and it reads the world model, so building a solver would add a
    Newton/Warp dependency without adding coverage. Mirrors the harness the
    merged Newton recording tests use.
    """
    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    engine._world = world
    engine.default_width = 320
    engine.default_height = 240
    if free_base is not None:
        engine._robot_free_base_joint = free_base
    return engine


def _schema(engine: Any) -> tuple[list[str], list[str]]:
    """Return ``(joint_names, action_names)`` from the backend's schema builder."""
    joint_names, action_names, *_rest = engine._collect_recording_schema()
    return joint_names, action_names


class TestNewtonDeclaresItsActionColumnsFromTheActionKeys:
    """The declared action columns are ``robot_action_keys``, per robot."""

    def test_a_fixed_base_robot_declares_every_joint_as_an_action_column(self):
        engine = _engine(_world("arm"))
        joint_names, action_names = _schema(engine)
        assert action_names == engine.robot_action_keys("arm")
        # A fixed-base robot has no free root, so the two vocabularies coincide
        # here. Pinned so the floating-base case below is read as the contrast.
        assert action_names == joint_names == _SO100_JOINTS

    def test_a_floating_base_robot_omits_the_free_joint_from_the_action_columns(self):
        """The free joint is not a commandable scalar, so it is not a column."""
        engine = _engine(
            _world("humanoid", joints=[_FREE_BASE, *_SO100_JOINTS]),
            free_base={"humanoid": _FREE_BASE},
        )
        _joint_names, action_names = _schema(engine)
        assert action_names == engine.robot_action_keys("humanoid")
        assert _FREE_BASE not in action_names
        assert action_names == _SO100_JOINTS

    def test_the_free_base_exclusion_is_not_re_derived_for_the_action_schema(self):
        """The action columns follow ``robot_action_keys``, not the joint list.

        This is the regression the change closes, and it is the one assertion
        here that fails on the pre-fix tree. Before, the action columns were the
        recorder's ``joint_names`` fallback; they matched only because the schema
        builder re-applied the free-base exclusion itself. A backend whose action
        keys are a different *vocabulary* - not merely a shorter list - exposes
        that: the declared columns must follow the keys the hook actually emits.
        """

        class _PrefixedActuators(NewtonSimEngine):
            """A robot whose actuators are named, not merely filtered, joints."""

            def robot_action_keys(self, robot_name: str) -> list[str]:
                return [f"a_{jn}" for jn in self.robot_joint_names(robot_name)]

        engine = _PrefixedActuators.__new__(_PrefixedActuators)
        engine._world = _world("arm")
        engine.default_width = 320
        engine.default_height = 240

        joint_names, action_names = _schema(engine)
        assert action_names == [f"a_{jn}" for jn in _SO100_JOINTS]
        # The state schema is unchanged - this is an action-side authority, and
        # conflating the two is what produced the defect.
        assert joint_names == _SO100_JOINTS
        assert action_names != joint_names

    def test_multi_robot_action_columns_are_namespaced_like_the_state_columns(self):
        engine = _engine(_world("alice", "bob"))
        joint_names, action_names = _schema(engine)
        assert action_names == [f"{r}__{k}" for r in ("alice", "bob") for k in _SO100_JOINTS]
        assert len(action_names) == 2 * len(_SO100_JOINTS)
        # Same prefixing as the state columns, so a frame keyed by the hook's
        # ``robot__actuator`` names resolves against the declared schema.
        assert action_names == joint_names

    def test_a_single_robot_scene_is_not_namespaced(self):
        engine = _engine(_world("solo"))
        _joint_names, action_names = _schema(engine)
        assert action_names == _SO100_JOINTS
        assert not any(name.startswith("solo__") for name in action_names)

    def test_the_declared_columns_are_exactly_what_the_hook_owes(self):
        """Declared == the hook's ``required_action_keys``, so no column is unowed.

        The hook scopes its ``required_action_keys`` to the driven robot with the
        same prefixing. In a single-robot session that set is the whole declared
        schema, which is the property that makes the ``0.0`` fill unreachable
        there: every declared column is one some frame owes a value for.
        """
        engine = _engine(_world("arm"))
        _joint_names, action_names = _schema(engine)
        assert action_names == list(engine.robot_action_keys("arm"))


class TestNoBackendFallsBackToTheJointNameVocabulary:
    """Structural guard: every ``start_recording`` declares ``action_names``.

    A backend that stops passing it silently re-enters the recorder's
    ``joint_names`` fallback, which is exactly the drift this change removed and
    which no behavioural test on that backend would fail.
    """

    SCHEMA_MODULES = [
        "strands_robots/simulation/isaac/recording.py",
        "strands_robots/simulation/newton/recording.py",
        "strands_robots/simulation/mujoco/recording.py",
    ]

    @staticmethod
    def _recorder_create_calls(source: str) -> list[ast.Call]:
        """Every ``_DatasetRecorder.create(...)`` call in ``source``."""
        return [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "create"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "_DatasetRecorder"
        ]

    @pytest.mark.parametrize("module_path", SCHEMA_MODULES)
    def test_every_recorder_create_declares_its_action_columns(self, module_path):
        source = (_REPO_ROOT / module_path).read_text(encoding="utf-8")
        calls = self._recorder_create_calls(source)
        assert calls, f"{module_path}: no _DatasetRecorder.create call found"
        for call in calls:
            keywords = {kw.arg for kw in call.keywords}
            assert "action_names" in keywords, (
                f"{module_path}: _DatasetRecorder.create at line {call.lineno} omits "
                "action_names, so the recorder falls back to the joint-name "
                "vocabulary for the action columns (see module docstring)"
            )

    def test_the_scan_is_not_vacuous(self):
        """Each module really is parsed and really does contain a create call."""
        found = {
            path: len(self._recorder_create_calls((_REPO_ROOT / path).read_text(encoding="utf-8")))
            for path in self.SCHEMA_MODULES
        }
        assert all(count >= 1 for count in found.values()), found

    def test_the_guard_detects_a_backend_that_omits_action_names(self):
        """Planted positive: the assertion is not satisfiable by an empty scan."""
        planted = "_DatasetRecorder.create(repo_id=repo_id, fps=fps, joint_names=joint_names)\n"
        calls = self._recorder_create_calls(planted)
        assert len(calls) == 1
        assert "action_names" not in {kw.arg for kw in calls[0].keywords}
