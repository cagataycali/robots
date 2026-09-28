"""Isaac posture flags select a posture, so they are checked, not read by truthiness.

Three surfaces on the Isaac backend took a boolean that chooses between two
*postures* and read it by truthiness:

* ``set_object_kinematic(name, kinematic)`` - KINEMATIC (pinned) vs dynamic,
* ``set_object_collision(name, enabled)`` - collider on vs off,
* ``create_world(ground_plane=)`` - add the ground plane or not.

Every non-empty string is truthy, so ``"false"``/``"no"``/``"off"``/``"0"`` - the
spellings an operator reaches for to opt out - selected the *ON* posture while
spelling its refusal: ``set_object_kinematic("drop", "false")`` pinned the body
kinematic (a 1 kg cube at ``z=3`` then fell 0.00 m instead of 2.95 m), and
``create_world(ground_plane="false")`` ADDED the plane. ``None``/``0``/``[]`` took
the other branch without ever being a declared spelling of it.

The fix routes each flag through :func:`~strands_robots.utils.boolean_flag_error`,
the shared domain owner every other posture-flag surface on this backend already
uses, before the write. These pins are parametrized over that function itself
rather than a copied spelling list, so a spelling added to the domain is covered
here without an edit.
"""

import ast
import inspect
import textwrap
import threading
import types
from typing import Any

import numpy as np
import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.simulation import (  # noqa: E402
    IsaacConfig,
    IsaacSimulation,
    _ObjectState,
)
from strands_robots.utils import boolean_flag_error  # noqa: E402

#: Non-booleans the domain refuses (the opt-out spellings that read as truthy,
#: plus the branch-takers that are never a declared spelling).
_NON_BOOL = ("false", "no", "off", "0", "", None, 0, 1, [], "true")
#: The booleans the domain honours.
_BOOL = (True, False, np.bool_(True), np.bool_(False))


class _KinematicHandle:
    """Records the value written through the wrapper's own kinematic setter."""

    def __init__(self) -> None:
        self.written: Any = None

    def set_rigid_body_kinematic(self, value: Any) -> None:
        self.written = value


class _CollisionHandle:
    """Records the value written through the wrapper's own collision setter."""

    def __init__(self) -> None:
        self.written: Any = None

    def set_collision_enabled(self, value: Any) -> None:
        self.written = value


def _engine(handle: Any = None) -> Any:
    """A minimally-stubbed engine holding one object named ``drop``."""
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._config = IsaacConfig(render_mode="headless")
    engine._world = types.SimpleNamespace()
    engine._world_created = True
    engine._objects = {
        "drop": _ObjectState(
            name="drop",
            prim_path="/World/Objects/drop",
            shape="box",
            is_static=False,
            handle=handle,
        )
    }
    return engine


class TestSetObjectKinematicIsChecked:
    @pytest.mark.parametrize("bad", _NON_BOOL)
    def test_a_non_boolean_is_refused(self, bad: Any) -> None:
        result = _engine().set_object_kinematic("drop", bad)
        assert result["status"] == "error", result
        assert "kinematic" in result["content"][0]["text"]

    @pytest.mark.parametrize("bad", _NON_BOOL)
    def test_the_refusal_precedes_the_write(self, bad: Any) -> None:
        """A truthy string must not reach the setter; the handle stays untouched."""
        handle = _KinematicHandle()
        result = _engine(handle).set_object_kinematic("drop", bad)
        assert result["status"] == "error", result
        assert handle.written is None

    @pytest.mark.parametrize("good", [True, False])
    def test_a_boolean_reaches_the_write(self, good: bool) -> None:
        handle = _KinematicHandle()
        result = _engine(handle).set_object_kinematic("drop", good)
        assert result["status"] == "success", result
        assert handle.written is good


class TestSetObjectCollisionIsChecked:
    @pytest.mark.parametrize("bad", _NON_BOOL)
    def test_a_non_boolean_is_refused(self, bad: Any) -> None:
        result = _engine().set_object_collision("drop", bad)
        assert result["status"] == "error", result
        assert "enabled" in result["content"][0]["text"]

    @pytest.mark.parametrize("bad", _NON_BOOL)
    def test_the_refusal_precedes_the_write(self, bad: Any) -> None:
        handle = _CollisionHandle()
        result = _engine(handle).set_object_collision("drop", bad)
        assert result["status"] == "error", result
        assert handle.written is None

    @pytest.mark.parametrize("good", [True, False])
    def test_a_boolean_reaches_the_write(self, good: bool) -> None:
        handle = _CollisionHandle()
        result = _engine(handle).set_object_collision("drop", good)
        assert result["status"] == "success", result
        assert handle.written is good


class TestCreateWorldGroundPlaneIsChecked:
    def _fresh(self) -> Any:
        engine = IsaacSimulation.__new__(IsaacSimulation)
        engine._lock = threading.RLock()
        engine._config = IsaacConfig(render_mode="headless")
        engine._world_created = False
        return engine

    @pytest.mark.parametrize("bad", _NON_BOOL)
    def test_a_non_boolean_is_refused_before_any_launch(self, bad: Any) -> None:
        """The check is first in the method, so a bad flag is refused without
        launching Kit - no ground plane is added and no world is built."""
        result = self._fresh().create_world(ground_plane=bad)
        assert result["status"] == "error", result
        assert "ground_plane" in result["content"][0]["text"]

    def test_the_flag_is_routed_through_the_shared_domain(self) -> None:
        """Read off the source: create_world calls boolean_flag_error on
        ground_plane, so the refusal is the shared one rather than a local copy."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(IsaacSimulation.create_world)))
        routed = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "boolean_flag_error"
            and any(isinstance(a, ast.Name) and a.id == "ground_plane" for a in node.args)
            for node in ast.walk(tree)
        )
        assert routed, "create_world does not route ground_plane through boolean_flag_error"


class TestTheDomainIsTheSharedOne:
    """Parametrized over boolean_flag_error itself, not a copied spelling list."""

    @pytest.mark.parametrize("bad", _NON_BOOL)
    def test_non_booleans_are_in_the_refused_domain(self, bad: Any) -> None:
        assert boolean_flag_error(bad, "flag", "ctx") is not None

    @pytest.mark.parametrize("good", _BOOL)
    def test_booleans_are_honoured(self, good: Any) -> None:
        assert boolean_flag_error(good, "flag", "ctx") is None
