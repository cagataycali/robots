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
import types
from typing import Any

import numpy as np
import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.simulation import (  # noqa: E402
    IsaacSimulation,
    _ObjectState,
)
from strands_robots.utils import boolean_flag_error  # noqa: E402
from tests.simulation._isaac_engine import isaac_engine

#: Non-booleans the domain refuses (the opt-out spellings that read as truthy,
#: plus the branch-takers that are never a declared spelling).
_NON_BOOL: tuple[Any, ...] = ("false", "no", "off", "0", "", None, 0, 1, [], "true")
#: The booleans the domain honours.
_BOOL: tuple[Any, ...] = (True, False, np.bool_(True), np.bool_(False))


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
    engine = isaac_engine()
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


@pytest.mark.parametrize("bad", _NON_BOOL, ids=repr)
def test_a_non_boolean_is_refused_by_every_surface_before_it_writes(bad: Any) -> None:
    """The shared domain refuses it, and so does each surface: by its parameter
    name, before the setter is reached (the handle stays untouched) and, for
    ``create_world``, before any launch - the check is first in the method."""
    assert boolean_flag_error(bad, "flag", "ctx") is not None

    kinematic = _KinematicHandle()
    result = _engine(kinematic).set_object_kinematic("drop", bad)
    assert result["status"] == "error", result
    assert "kinematic" in result["content"][0]["text"]
    assert kinematic.written is None

    collision = _CollisionHandle()
    result = _engine(collision).set_object_collision("drop", bad)
    assert result["status"] == "error", result
    assert "enabled" in result["content"][0]["text"]
    assert collision.written is None

    result = isaac_engine().create_world(ground_plane=bad)
    assert result["status"] == "error", result
    assert "ground_plane" in result["content"][0]["text"]


@pytest.mark.parametrize("good", [True, False])
def test_a_boolean_reaches_the_write(good: bool) -> None:
    kinematic = _KinematicHandle()
    assert _engine(kinematic).set_object_kinematic("drop", good)["status"] == "success"
    assert kinematic.written is good
    collision = _CollisionHandle()
    assert _engine(collision).set_object_collision("drop", good)["status"] == "success"
    assert collision.written is good


def test_the_ground_plane_flag_is_routed_through_the_shared_domain() -> None:
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


@pytest.mark.parametrize("good", _BOOL)
def test_booleans_are_honoured_by_the_shared_domain(good: Any) -> None:
    assert boolean_flag_error(good, "flag", "ctx") is None
