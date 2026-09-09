"""A sim peer's rollout is reported on every backend, through one seam.

The mesh has two surfaces that answer "is a rollout in flight": the ``status``
command and the ``robots`` section of the state topic. Both probed for
``_active_policy_robots``, a method only the MuJoCo engine defines, so a Newton
peer driving a rollout was invisible on both while the stop verb on that same
peer halted it:

===============  ==========  ============================  ============  ===============
peer             rollout     ``status``                    state topic   ``stop``
===============  ==========  ============================  ============  ===============
Newton           in flight   ``{"status": "unknown"}``      no flag       halts it
Newton           idle        ``{"status": "unknown"}``      no flag       nothing to halt
MuJoCo           in flight   ``running``, ``["arm"]``        ``true``      halts it
MuJoCo           idle        ``idle``, ``[]``                ``false``     nothing to halt
===============  ==========  ============================  ============  ===============

The two Newton rows were identical, so a caller could not tell a live rollout
from an idle world, and the state topic published ``active=false`` on every
robot - an affirmative "nothing is running" given on no evidence.

Both backends keep the per-robot ``policy_running`` claim already, and Isaac
already trusts it as a population for its own busy guard; what was missing was a
shared way to ask. These cases hold every backend to
:meth:`~strands_robots.simulation.base.SimEngine._rollouts_in_flight`, the
tri-state seam whose ``None`` means "this backend reports no population" rather
than "no robot is running", and hold the two mesh surfaces to reading it once.

Pinned at the bottom: the fleet stop's population is NOT this population. It
asks every robot in the world for a backend keeping no pruned rollout registry,
which is how a Newton rollout is halted today, and narrowing it to the in-flight
set is a decision about a safety path (#3359) rather than part of reporting.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from strands_robots.mesh import core as mesh_core
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.models import SimRobot, SimWorld

_JOINTS = ("j1", "j2")
_ARM = "arm"


def _newton(policy_running: bool) -> Any:
    """A Newton engine skeleton whose one robot carries the rollout claim.

    ``__new__`` plus the attributes the reporting path reads - the fixture shape
    this backend's own recording and stop tests use, so neither Warp nor a
    compiled model is needed to grade a flag read.
    """
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    world = SimWorld()
    robot = SimRobot(name=_ARM, urdf_path="arm.xml", data_config="so100", joint_names=list(_JOINTS))
    robot.policy_running = policy_running
    world.robots[_ARM] = robot
    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    engine._world = world
    engine._model = object()
    return engine


def _isaac(policy_running: bool) -> Any:
    """An Isaac engine skeleton whose one robot carries the rollout claim."""
    from strands_robots.simulation.isaac.simulation import IsaacSimulation, _RobotState

    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._world_created = True
    engine._world = object()
    state = _RobotState(name=_ARM, prim_path=f"/World/{_ARM}", joint_names=list(_JOINTS))
    state.policy_running = policy_running
    engine._robots = {_ARM: state}
    return engine


_BACKENDS = {"newton": _newton, "isaac": _isaac}


def _status(engine: Any) -> dict[str, Any]:
    """What the ``status`` command answers for a peer holding ``engine``."""
    mesh = mesh_core.Mesh(engine, peer_id="sim-1", peer_type="simulation")
    return mesh._dispatch({"action": "status"})


def _robots_section(engine: Any) -> dict[str, Any]:
    """The state topic's ``robots`` section for a peer holding ``engine``."""
    snapshot = mesh_core.Mesh(engine, peer_id="sim-2", peer_type="simulation")._read_state()
    assert snapshot is not None, "the peer published nothing on the state topic"
    section = snapshot.get("robots")
    assert isinstance(section, dict), f"no robots section in the snapshot: {sorted(snapshot)}"
    return section


class TestEveryBackendReportsTheRolloutItIsDriving:
    """The regression: a rollout in flight is distinguishable from an idle world."""

    @pytest.mark.parametrize("backend", sorted(_BACKENDS))
    def test_a_rollout_in_flight_is_reported_running_and_named(self, backend: str) -> None:
        answer = _status(_BACKENDS[backend](True))
        assert answer["status"] == "running"
        assert answer["robots_running"] == [_ARM]

    @pytest.mark.parametrize("backend", sorted(_BACKENDS))
    def test_an_idle_world_is_reported_idle_rather_than_unknown(self, backend: str) -> None:
        """The other half: "running" for everything would pass the case above."""
        answer = _status(_BACKENDS[backend](False))
        assert answer["status"] == "idle"
        assert answer["robots_running"] == []

    @pytest.mark.parametrize("backend", sorted(_BACKENDS))
    def test_the_backend_answers_the_seam_every_reader_asks(self, backend: str) -> None:
        assert _BACKENDS[backend](True)._rollouts_in_flight() == (_ARM,)
        assert _BACKENDS[backend](False)._rollouts_in_flight() == ()


class TestTheStateTopicMeasuresTheFlagOnEveryBackend:
    """``active`` beside each robot is read from the same population."""

    def test_the_robot_running_a_policy_is_flagged_active(self) -> None:
        assert _robots_section(_newton(True)) == {_ARM: {"active": True}}

    def test_an_idle_robot_is_flagged_inactive_because_that_was_measured(self) -> None:
        assert _robots_section(_newton(False)) == {_ARM: {"active": False}}

    def test_the_two_surfaces_agree_because_they_read_one_call(self) -> None:
        for running in (True, False):
            flagged = {name for name, entry in _robots_section(_newton(running)).items() if entry["active"]}
            assert flagged == set(_status(_newton(running))["robots_running"])


class TestNoVerdictIsNotAnIdleWorld:
    """``None`` from the seam is an absence of evidence, and stays one."""

    def test_the_base_default_reports_no_population(self) -> None:
        """Called unbound: the ABC cannot be instantiated, and reads no state."""
        assert SimEngine._rollouts_in_flight(cast(SimEngine, SimpleNamespace())) is None

    def test_a_torn_down_newton_world_has_no_population_rather_than_an_empty_one(self) -> None:
        engine = _newton(True)
        engine._world = None
        assert engine._rollouts_in_flight() is None

    def test_isaac_before_a_world_exists_has_no_population(self) -> None:
        engine = _isaac(True)
        engine._world_created = False
        assert engine._rollouts_in_flight() is None

    def test_a_peer_reporting_no_population_answers_unknown_not_idle(self) -> None:
        """ "idle" here would be the same affirmative claim, from the other side."""
        engine = _newton(True)
        engine._world = None
        assert _status(engine) == {"status": "unknown"}


class TestTheFleetStopPopulationIsNotTheReportingPopulation:
    """Reporting a rollout must not narrow which robots a stop asks (#3359)."""

    @staticmethod
    def _fleet_stop(engine: Any) -> dict[str, Any]:
        mesh = mesh_core.Mesh(engine, peer_id="sim-3", peer_type="simulation")
        return mesh._dispatch({"action": "stop"})

    def test_a_rollout_in_flight_is_still_halted_and_named(self) -> None:
        engine = _newton(True)
        result = self._fleet_stop(engine)
        assert result["ok"] is True
        assert result["stopped"] == [_ARM]
        assert engine._world.robots[_ARM].policy_running is False

    def test_an_idle_robot_is_still_asked_even_though_it_reports_as_idle(self) -> None:
        """A registry-only population would leave a blocking rollout untouched."""
        result = self._fleet_stop(_newton(False))
        assert result["ok"] is True
        assert result["stopped"] == []
        assert _ARM in result["results"]
