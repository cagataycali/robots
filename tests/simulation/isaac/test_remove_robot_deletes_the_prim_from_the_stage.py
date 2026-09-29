# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression: ``remove_robot`` deletes the robot's USD prim from the stage.

``IsaacSimulation.remove_robot`` used to prune only the in-Python registries
(``_robots``, ``_action_controllers``, ``_prim_registry``) and delegate the
actual prim deletion to ``destroy`` / world teardown, so the articulation stayed
on the stage after the call returned. Its two sibling verbs did not behave that
way: ``remove_camera`` calls ``delete_prim`` and ``remove_object`` removes
through ``world.scene``.

The leak was reachable from ordinary use because ``add_robot`` refuses only a
name still present in ``_robots`` -- so ``remove_robot("arm")`` followed by
``add_robot("arm")`` passed the refusal and composed a SECOND USD reference onto
the leftover prim rather than a fresh one. On the GPU that shows up as a cube
dropped onto the vacated spot coming to rest at the old arm's height instead of
the ground, and as two references under one prim path.

These pins are unit-level: the Kit ``delete_prim`` leaf is stood in (the pattern
:mod:`tests.simulation.isaac.test_step_refuses_a_scene_the_tensor_view_no_longer_covers`
uses), so what is graded is that the removal reaches ``delete_prim`` for every
path the robot occupies, marks the view stale, and leaves a re-add composing a
single reference. The live-Kit rest-height half is exercised on GPU.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.simulation import (  # noqa: E402 - after importorskip
    IsaacSimulation,
    _RobotState,
)
from tests.simulation._isaac_engine import isaac_engine


class _StageModel:
    """The USD references currently composed under each prim path.

    A reference add appends; ``delete_prim`` drops the path outright. This is the
    one property the leak is about: whether a path is clear when the next
    ``add_robot`` composes onto it.
    """

    def __init__(self) -> None:
        self.references: dict[str, int] = {}

    def add_reference(self, prim_path: str) -> None:
        self.references[prim_path] = self.references.get(prim_path, 0) + 1

    def delete(self, prim_path: str) -> None:
        self.references.pop(prim_path, None)


@pytest.fixture
def stage() -> _StageModel:
    return _StageModel()


@pytest.fixture
def fake_delete_prim(monkeypatch, stage: _StageModel) -> list[str]:
    """Stand in the ``isaacsim.core.utils.prims.delete_prim`` leaf.

    Records every path deleted and mutates the stage model, so a test can assert
    both which paths the removal reached and the resulting reference state.
    """
    deleted: list[str] = []

    def _delete_prim(prim_path: str) -> None:
        deleted.append(prim_path)
        stage.delete(prim_path)

    for name in ("isaacsim", "isaacsim.core", "isaacsim.core.utils", "isaacsim.core.utils.prims"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["isaacsim.core.utils.prims"].delete_prim = _delete_prim  # type: ignore[attr-defined]
    return deleted


def _engine(stage: _StageModel) -> Any:
    """A skeleton engine whose world is present so the delete path runs."""
    engine = isaac_engine()
    engine._world = object()  # truthy: the delete branch is gated on ``is not None``
    return engine


def _add_robot(engine: Any, stage: _StageModel, name: str, *, actual: str | None = None) -> None:
    """Register a robot and compose its reference, the way ``add_robot`` would."""
    prim_path = f"/World/Robots/{name}"
    engine._robots[name] = _RobotState(name=name, prim_path=prim_path, joint_names=["j0"], actual_prim_path=actual)
    engine._prim_registry.append(prim_path)
    stage.add_reference(actual or prim_path)


class TestRemovalDeletesThePrim:
    def test_delete_prim_is_called_for_the_robots_path(self, stage, fake_delete_prim):
        engine = _engine(stage)
        _add_robot(engine, stage, "arm")

        assert IsaacSimulation.remove_robot(engine, "arm")["status"] == "success"

        assert fake_delete_prim == ["/World/Robots/arm"]
        assert stage.references == {}
        assert "arm" not in engine._robots

    def test_a_relocated_robot_deletes_both_paths(self, stage, fake_delete_prim):
        """When the importer relocated the robot, ``actual_prim_path`` differs
        from ``prim_path`` and both must go, de-duplicated and requested-first."""
        engine = _engine(stage)
        _add_robot(engine, stage, "arm", actual="/so101_new_calib")

        assert IsaacSimulation.remove_robot(engine, "arm")["status"] == "success"

        assert fake_delete_prim == ["/World/Robots/arm", "/so101_new_calib"]
        assert stage.references == {}

    def test_the_common_case_deletes_once(self, stage, fake_delete_prim):
        """``actual_prim_path`` defaults to ``prim_path``; the two are one path,
        so the de-dup deletes it a single time rather than twice."""
        engine = _engine(stage)
        _add_robot(engine, stage, "arm")  # actual defaults to prim_path

        IsaacSimulation.remove_robot(engine, "arm")

        assert fake_delete_prim == ["/World/Robots/arm"]


class TestReAddComposesOneReference:
    def test_remove_then_re_add_leaves_a_single_reference(self, stage, fake_delete_prim):
        """The leak, stated as the reference count under the re-added path.

        Pre-fix, ``remove_robot`` deleted nothing, so the reference stayed and the
        re-add composed a second onto it -> count 2. With the delete, the re-add
        composes onto an empty path -> count 1.
        """
        engine = _engine(stage)
        _add_robot(engine, stage, "arm")
        assert stage.references == {"/World/Robots/arm": 1}

        assert IsaacSimulation.remove_robot(engine, "arm")["status"] == "success"

        _add_robot(engine, stage, "arm")  # the re-add ``add_robot`` now permits
        assert stage.references == {"/World/Robots/arm": 1}


class TestRemovalMarksTheViewStale:
    def test_removing_a_robot_marks_the_scene(self, stage, fake_delete_prim):
        """A robot is an articulation held in PhysX's tensor view, so deleting it
        invalidates the view the same way a dynamic ``remove_object`` does."""
        engine = _engine(stage)
        _add_robot(engine, stage, "arm")
        assert engine._physics_view_stale is False

        IsaacSimulation.remove_robot(engine, "arm")

        assert engine._physics_view_stale is True


class TestADeleteFailureLeavesBookkeepingIntact:
    def test_a_raising_delete_returns_error_and_keeps_the_robot(self, stage, monkeypatch):
        """Mirrors ``remove_camera``: a transient stage error returns the
        structured envelope and leaves the robot registered for retry, and does
        NOT mark the view stale (nothing was deleted)."""
        for name in ("isaacsim", "isaacsim.core", "isaacsim.core.utils", "isaacsim.core.utils.prims"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

        def _boom(prim_path: str) -> None:
            raise RuntimeError("stage torn down")

        sys.modules["isaacsim.core.utils.prims"].delete_prim = _boom  # type: ignore[attr-defined]

        engine = _engine(stage)
        _add_robot(engine, stage, "arm")

        result = IsaacSimulation.remove_robot(engine, "arm")

        assert result["status"] == "error"
        assert "arm" in engine._robots
        assert engine._physics_view_stale is False
