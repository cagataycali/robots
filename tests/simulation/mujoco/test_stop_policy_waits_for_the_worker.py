"""``stop_policy`` answers "Stopped" only once the worker has actually exited.

The stop is a cooperative REQUEST - the worker exits at its next control
tick, up to one control period later. This surface used to return "Stopped
on 'x'" the instant the flag was lowered, so the caller's natural next call
(run_policy on the same robot, a scene mutation, a second stop) landed inside
that window and was refused "while its policy is running" by a robot the
previous answer had just called stopped; list_policies_running still listed
it and a second stop reported was_running=True again. The repo's own tests
joined ``sim._policy_threads[name].result()`` by hand after every stop.

Pinned: after a stop that reports ``worker_exited=True`` the robot is gone
from list_policies_running and run_policy on it succeeds immediately; a
worker that does not exit within the budget keeps "Stopped" (the claim IS
durably lowered) and gains a second sentence saying the worker has not exited,
with ``worker_exited=False``; the
nothing-running case reports ``worker_exited=None``; and an empty
``robot_name`` names the rollouts in flight (or says none is).
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.mujoco.simulation import Simulation


def _text(result: dict) -> str:
    return "\n".join(c["text"] for c in result["content"] if isinstance(c, dict) and "text" in c)


def _json(result: dict) -> dict:
    return next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)


@pytest.fixture
def sim():
    s = Simulation(tool_name="stop_policy_waits", mesh=False)
    s.create_world()
    assert s.add_robot(name="so101", data_config="so101")["status"] == "success"
    assert s.add_robot(name="go2", data_config="unitree_go2", position=[1.0, 0.0, 0.0])["status"] == "success"
    yield s
    s.cleanup(policy_stop_timeout=0.5)


def test_stop_then_run_policy_on_the_same_robot_succeeds_immediately(sim):
    assert sim.start_policy("so101", policy_provider="mock")["status"] == "success"
    stopped = sim.stop_policy("so101")
    assert stopped["status"] == "success"
    assert _text(stopped).startswith("Stopped on 'so101'")
    assert _json(stopped) == {"robot": "so101", "was_running": True, "worker_exited": True}

    # No hand-rolled join here - that is the point.
    assert "so101" not in _text(sim.list_policies_running())
    run = sim.run_policy("so101", policy_provider="mock", duration=0.1, control_frequency=50.0)
    assert run["status"] == "success", _text(run)


def test_stop_reports_nothing_running_with_worker_exited_none(sim):
    r = sim.stop_policy("so101")
    assert r["status"] == "success"
    assert _json(r) == {"robot": "so101", "was_running": False, "worker_exited": None}


def test_empty_robot_name_names_the_rollouts_in_flight(sim):
    idle = sim.stop_policy("")
    assert idle["status"] == "error"
    assert "No policy is running" in _text(idle)

    assert sim.start_policy("go2", policy_provider="mock")["status"] == "success"
    busy = sim.stop_policy("")
    assert busy["status"] == "error"
    assert "Running now: ['go2']" in _text(busy)
    sim.stop_policy("go2")


class _StuckPolicy(MockPolicy):
    """A policy whose inference blocks until released - the stop flag is not read inside it."""

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()
        self.entered = threading.Event()

    async def get_actions(self, observation_dict, instruction, **kwargs):  # type: ignore[override]
        self.entered.set()
        while not self.release.is_set():
            await asyncio.sleep(0.01)
        return await super().get_actions(observation_dict, instruction, **kwargs)


def test_a_worker_that_does_not_exit_in_time_is_reported_not_called_stopped(sim, monkeypatch):
    monkeypatch.setattr(Simulation, "_STOP_POLICY_JOIN_TIMEOUT", 0.2)
    policy = _StuckPolicy()
    started = sim.start_policy("so101", policy_object=policy)
    assert started["status"] == "success", _text(started)
    assert policy.entered.wait(5.0)

    t0 = time.monotonic()
    r = sim.stop_policy("so101")
    waited = time.monotonic() - t0
    assert r["status"] == "success"
    text = _text(r)
    assert r["content"][0]["text"] == "Stopped on 'so101'"  # the stop itself HAS landed durably
    assert "has not exited after 0.2s" in text, text
    assert "run_policy on this robot is refused" in text
    assert _json(r)["worker_exited"] is False
    assert _json(r)["was_running"] is True
    assert 0.15 <= waited < 2.0

    policy.release.set()
    sim._policy_threads["so101"].result(timeout=5.0)
    again = sim.stop_policy("so101")
    assert _json(again)["was_running"] is False
