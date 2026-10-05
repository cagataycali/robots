"""``get_observation`` names the cause when it returns an empty observation.

An empty dict is the method's degraded mode on every backend. Isaac logged a
WARNING for each such branch; MuJoCo and Newton returned ``{}`` silently, so a
loop that read ``get_observation()`` after ``destroy()`` saw nothing while
``send_action`` and ``step`` on the same engine answered ``status="error"``.

The guards run before any physics or GL call, so a bare instance with only the
world fields set exercises them without ``newton`` / ``warp`` installed.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine
from strands_robots.simulation.newton.simulation import NewtonSimEngine

_ROBOTS = {"so101": object()}


def _mujoco(world):
    engine = object.__new__(MuJoCoSimEngine)
    engine._world = world
    return engine


def _newton(world):
    engine = object.__new__(NewtonSimEngine)
    engine._world = world
    engine._model = None if world is None else object()
    return engine


@pytest.mark.parametrize(
    ("make", "world", "robot_name", "cause"),
    [
        (_mujoco, None, None, "No world"),
        (_mujoco, SimpleNamespace(_model=object(), robots={}), None, "no robots"),
        (_mujoco, SimpleNamespace(_model=object(), robots=_ROBOTS), "so10l", "Known: ['so101']"),
        (_newton, None, None, "No world"),
        (_newton, SimpleNamespace(robots={}), None, "No robots registered"),
        (_newton, SimpleNamespace(robots={"a": 1, "b": 2}), None, "Multiple robots"),
        (_newton, SimpleNamespace(robots=_ROBOTS), "so10l", "Known: ['so101']"),
    ],
    ids=[
        "mujoco-no-world",
        "mujoco-no-robots",
        "mujoco-unknown-robot",
        "newton-no-world",
        "newton-no-robots",
        "newton-ambiguous",
        "newton-unknown-robot",
    ],
)
def test_an_empty_observation_is_logged_with_its_cause(make, world, robot_name, cause, caplog):
    engine = make(world)
    with caplog.at_level(logging.WARNING, logger=type(engine).__module__):
        assert engine.get_observation(robot_name) == {}
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(cause in m for m in messages), messages


def test_get_observation_after_destroy_is_logged(caplog):
    engine = MuJoCoSimEngine()
    engine.create_world()
    engine.add_robot("so101")
    assert engine.get_observation(skip_images=True)
    assert engine.destroy()["status"] == "success"
    with caplog.at_level(logging.WARNING, logger=MuJoCoSimEngine.__module__):
        assert engine.get_observation() == {}
    assert any("No world" in r.getMessage() for r in caplog.records), caplog.text
