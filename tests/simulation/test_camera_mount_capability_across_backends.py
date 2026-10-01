# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every backend mounts a camera on a body with ``add_camera(parent_body=...)``.

``parent_body`` mounts a camera ON a moving body so a wrist view rides with the
arm - the documented remedy for a VLA whose model card declares a wrist image
feature. MuJoCo and Newton implemented it; Isaac refused it, so every pi
checkpoint (DROID ``left_wrist_0_rgb``, LIBERO ``image2``) got a world-fixed
"wrist" camera on Isaac, a view distribution none of them was trained on. Isaac
now authors the camera prim as a child of the link prim. These tests pin the
parts that need no Isaac Sim runtime: the parameter's domain and its place
before the world check, and the MuJoCo / Newton behaviour; the live mount is
verified on Isaac Sim (the camera's world pose follows the link).
"""

from __future__ import annotations

import inspect
import pathlib
import threading
import types
from typing import Any

import pytest

from strands_robots.simulation.isaac.simulation import IsaacSimulation
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine
from strands_robots.simulation.newton.simulation import NewtonSimEngine

#: The backends whose ``add_camera`` mounts a camera on a body: all three.
_MOUNTING_BACKENDS = ("mujoco", "newton", "isaac")

#: Mount points a caller plausibly passes - the namespaced ``<robot>/<body>`` form
#: ``list_bodies`` returns, which is what the docs prescribe.
_MOUNTS = ("so101/gripper", "arm/gripper", "panda/hand")


def _isaac_skeleton() -> Any:
    """An ``IsaacSimulation`` that has not run ``__init__``.

    The ``parent_body`` refusal is answered before ``with self._lock``, so it
    reads no instance state at all. Deliberately the *worst* case: an instance
    with no attributes proves the answer cannot depend on the world, the stage or
    the camera registry - which is the placement property under test.
    """
    return IsaacSimulation.__new__(IsaacSimulation)


def _isaac_add_camera(**kwargs: Any) -> dict[str, Any]:
    """Drive Isaac's ``add_camera`` on the bare skeleton.

    A funnel so the deliberately off-nominal ``self`` is stated once rather than
    at every call site.
    """
    return IsaacSimulation.add_camera(_isaac_skeleton(), **kwargs)


def _isaac_past_the_guard(**kwargs: Any) -> dict[str, Any]:
    """Drive Isaac's ``add_camera`` on a skeleton that can reach the world check.

    The bare skeleton cannot: ``with self._lock`` needs an attribute it has no
    reason to carry, and that asymmetry is itself the placement evidence. The
    refusal above answers an instance with *no* attributes, while a call the
    guard lets through immediately needs the lock - so the guard provably sits
    before it. This skeleton adds only the lock and a world that was never
    created, so a call past the guard lands on "No world created".
    """
    skeleton = _isaac_skeleton()
    skeleton._lock = threading.RLock()
    skeleton._world_created = False
    return IsaacSimulation.add_camera(skeleton, **kwargs)


def _newton_stub() -> Any:
    """A stand-in for ``self`` carrying only what Newton's ``add_camera`` reads.

    The three attributes the sibling
    ``tests/simulation/newton/test_add_camera_numeric_validation.py`` establishes,
    so the accepted path runs without the optional ``newton`` / ``warp`` packages.
    """
    return types.SimpleNamespace(
        _world=types.SimpleNamespace(cameras={}),
        _model=types.SimpleNamespace(body_label=["ground", "so101/gripper"]),
        _lock=threading.RLock(),
    )


def _text(result: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in result.get("content", []))


class TestIsaacChecksTheMountBeforeTheWorld:
    """What is answered before the lock: the mount's type, and that it needs a pose."""

    @pytest.mark.parametrize("bad", [7, "", "   ", ["so101", "gripper"]])
    def test_a_mount_that_is_not_a_body_name_is_refused(self, bad: Any) -> None:
        text = _text(_isaac_add_camera(name="wrist", parent_body=bad))
        assert "parent_body must be a body name" in text

    @pytest.mark.parametrize("kwargs", [{}, {"position": [0.0, 0.0, 0.05]}, {"target": [0.0, 0.0, 0.1]}])
    def test_a_mount_without_both_position_and_target_is_refused(self, kwargs: dict[str, Any]) -> None:
        text = _text(_isaac_add_camera(name="wrist", parent_body="so101/gripper", **kwargs))
        assert "needs both position and target" in text and "1.7 m" in text

    @pytest.mark.parametrize("mount", _MOUNTS)
    def test_a_well_formed_mount_reaches_the_world_like_any_camera(self, mount: str) -> None:
        text = _text(
            _isaac_past_the_guard(name="wrist", parent_body=mount, position=[0.0, 0.0, 0.05], target=[0.0, 0.0, 0.2])
        )
        assert "No world created" in text and "parent_body" not in text


class TestOmittingTheMountIsUnaffected:
    """The world-fixed camera still reaches the same path."""

    def test_omitting_it_passes_the_guard(self) -> None:
        text = _text(_isaac_past_the_guard(name="front", position=[1.0, 0.0, 0.5]))
        assert "No world created" in text
        assert "parent_body" not in text

    def test_an_explicit_none_passes_the_guard(self) -> None:
        """``None`` is the documented default and means "world-fixed"."""
        text = _text(_isaac_past_the_guard(name="front", parent_body=None))
        assert "No world created" in text


class TestTheMountingBackendsStillMount:
    """No regression: the two backends that implement the mount still accept it."""

    def test_newton_still_mounts(self) -> None:
        result = NewtonSimEngine.add_camera(
            _newton_stub(), name="wrist", parent_body="so101/gripper", position=[0.0, 0.0, 0.05]
        )
        assert result["status"] == "success", result

    def test_mujoco_still_mounts(self) -> None:
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine as Engine

        sim = Engine(tool_name="mount_parity_sim", mesh=False)
        try:
            assert sim.create_world()["status"] == "success"
            assert sim.add_robot(name="so101")["status"] == "success"
            result = sim.add_camera(
                name="wrist", parent_body="so101/gripper", position=[0.0, 0.0, 0.05], target=[0.0, 0.0, 0.1]
            )
            assert result["status"] == "success", result
            assert "so101/gripper" in _text(result)
        finally:
            sim.cleanup()


class TestEveryBackendDeclaresTheMount:
    """The parameter is on all three signatures, so no caller gets a TypeError.

    Declaring it is what moves the answer from Python's argument binding into the
    backend, where it can name the capability and the alternative. A backend that
    silently dropped it would be worse than the ``TypeError``; a backend that does
    not declare it cannot answer at all.

    The ``engine`` parameter is typed ``Any`` because ``add_camera`` is not on the
    ``SimEngine`` ABC - the abstract set is ``add_object`` / ``add_robot`` /
    ``create_world`` / ``step`` / ... and ``add_camera`` is defined independently
    by each backend. That absence is exactly why the three signatures could drift,
    so there is no common base to annotate against.
    """

    @pytest.mark.parametrize(
        ("backend", "engine"),
        [("mujoco", MuJoCoSimEngine), ("newton", NewtonSimEngine), ("isaac", IsaacSimulation)],
    )
    def test_the_parameter_is_declared(self, backend: str, engine: Any) -> None:
        params = inspect.signature(engine.add_camera).parameters
        assert "parent_body" in params, f"{backend}: {sorted(params)}"

    @pytest.mark.parametrize(
        ("backend", "engine"),
        [("mujoco", MuJoCoSimEngine), ("newton", NewtonSimEngine), ("isaac", IsaacSimulation)],
    )
    def test_omitting_it_is_the_world_fixed_default(self, backend: str, engine: Any) -> None:
        """Every backend defaults it to ``None`` - a world-fixed camera."""
        assert inspect.signature(engine.add_camera).parameters["parent_body"].default is None


class TestTheDocumentedRemedyHoldsOnEveryBackend:
    """The backend-agnostic guidance that prescribes the mount says it works everywhere."""

    @staticmethod
    def _doc() -> str:
        root = pathlib.Path(inspect.getfile(IsaacSimulation)).parents[3]
        return (root / "docs" / "learn" / "policies" / "lerobot-local.md").read_text(encoding="utf-8")

    def test_the_doc_prescribes_the_mount_without_an_isaac_caveat(self) -> None:
        doc = " ".join(self._doc().split())
        sentence = doc[doc.index("`parent_body` mounts a camera on a link") :][:400]
        assert "every backend" in sentence and "refuses" not in sentence
