"""A blocking rollout is in flight, and every reader of that population says so.

Two entry points launch a rollout on this engine. ``start_policy`` submits it to
the executor and registers a Future; the blocking ``run_policy`` drives it on the
caller's own thread and registers nothing. ``_active_policy_robots`` -- the
population every "what is running" surface reads -- answered from the Future
table alone, so for as long as a blocking rollout drove the arm it was invisible:

* ``list_policies_running`` answered "No policies running.",
* the mesh ``status`` command answered ``idle`` with an empty ``robots_running``,
  and its state topic published ``active=False``, and
* the ``{"action": "stop"}`` fanout ``Mesh.emergency_stop`` broadcasts answered
  ``{"ok": True, "stopped": [], "note": "no policies running"}`` and halted
  NOTHING. :func:`~strands_robots.mesh.core._peers_that_did_not_stop` reads a
  top-level ``ok`` as a peer that stopped, so the operator was told the fleet had
  halted while the rollout kept driving the arm for the rest of its duration --
  measured at 8 of 10 seconds. That is the affirmative lie the surrounding stop
  branches are commented against, reached through the population rather than
  through the verdict.

``stop_policy`` was the only surface that also read the per-robot
``policy_running`` claim, so it and ``list_policies_running`` reported opposite
facts about the same instant ("Stopped on 'arm'" with ``was_running=True``
against "No policies running.") -- the two-sources drift #2833 is about, and the
thing ``docs/simulation/overview.md`` promised could not happen.

Pinned here: both launch shapes put the robot in the population, every reader
inherits that from the one verb that owns it, the fleet stop actually halts a
blocking rollout, and -- the half that keeps the widening from being over-broad
-- an idle world is still reported idle by all of them.

The blocking shape is set up through ``_announce_rollout``, the launcher seam the
blocking ``run_policy`` itself calls, so these read the state a real rollout
produces without racing one. The fleet-stop cells drive a genuine rollout,
because "it was really halted" is not a claim a flag can make.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from strands_robots.mesh import Mesh
from strands_robots.mesh.core import _peers_that_did_not_stop
from strands_robots.simulation import create_simulation
from strands_robots.simulation.model_registry import resolve_model_path

#: A rollout long enough that "ran to completion" and "was halted" are far apart.
DURATION = 8.0
CONTROL_HZ = 20.0
FULL_ROLLOUT_STEPS = int(DURATION * CONTROL_HZ)  # 160


@pytest.fixture
def sim() -> Any:
    """A MuJoCo sim holding one arm, torn down after the test."""
    engine = create_simulation("mujoco")
    engine.create_world()
    engine.add_robot(name="arm", urdf_path=str(resolve_model_path("so101")))
    try:
        yield engine
    finally:
        engine.cleanup()


def _readings(sim: Any) -> dict[str, Any]:
    """What every reader of the in-flight population answers, right now.

    One helper so a cell grades the surfaces together: they are meant to agree,
    and a table is how a disagreement between two of them shows up as a
    disagreement rather than as two unrelated failures.
    """
    mesh = Mesh(sim, peer_id="sim-1", peer_type="simulation")
    reported = mesh._running_policy_robots()
    # Never ``None`` here: the population reader is tri-state so a backend
    # keeping no rollout claim cannot be quoted as reporting an idle world, and
    # this engine always holds the registry.
    assert reported is not None, "a MuJoCo peer always reports its in-flight population"
    return {
        "population": sim._active_policy_robots(),
        "list_policies_running": sim.list_policies_running()["content"][0]["text"],
        "mesh_status": mesh._dispatch({"action": "status"}),
        "state_topic": sorted(reported),
    }


def _blocking_rollout(sim: Any) -> tuple[threading.Thread, dict[str, Any]]:
    """Drive a real blocking ``run_policy`` on another thread; return it and its result.

    The thread stands in for the arrangement a mesh peer serves under: the
    rollout occupies one thread while the wire handler answers on another.
    """
    result: dict[str, Any] = {}

    def drive() -> None:
        result["envelope"] = sim.run_policy(
            robot_name="arm",
            policy_provider="mock",
            duration=DURATION,
            control_frequency=CONTROL_HZ,
        )

    thread = threading.Thread(target=drive, daemon=True)
    thread.start()
    return thread, result


def _await_claim(sim: Any) -> None:
    """Block until the launcher has claimed the arm, so the window is really open."""
    deadline = threading.Event()
    for _ in range(600):
        if sim._world.robots["arm"].policy_running:
            return
        deadline.wait(0.05)
    raise AssertionError("the rollout never claimed the arm")


def _rollout_json(envelope: dict[str, Any]) -> dict[str, Any]:
    """The ``json`` block of a rollout envelope."""
    return next(block["json"] for block in envelope["content"] if "json" in block)


class TestThePopulationCountsBothLaunchShapes:
    """The population is about what drives the arm, not about how it was launched."""

    def test_a_blocking_rollout_puts_the_robot_in_the_population(self, sim: Any) -> None:
        sim._announce_rollout("arm")
        assert sim._active_policy_robots() == ["arm"]

    def test_a_background_rollout_still_does(self, sim: Any) -> None:
        """The Future-backed shape, which was the only one that ever counted."""
        assert sim.start_policy("arm", policy_provider="mock", duration=DURATION)["status"] == "success"
        assert sim._active_policy_robots() == ["arm"]
        sim.stop_policy("arm")

    def test_a_robot_holding_both_a_claim_and_a_future_is_named_once(self, sim: Any) -> None:
        """A union, not a concatenation: ``start_policy`` produces both at once."""
        assert sim.start_policy("arm", policy_provider="mock", duration=DURATION)["status"] == "success"
        assert sim._world.robots["arm"].policy_running is True
        assert sim._active_policy_robots() == ["arm"]
        sim.stop_policy("arm")

    def test_a_future_still_in_flight_counts_after_its_claim_is_lowered(self, sim: Any) -> None:
        """The tail of a background rollout, which is the half the claim cannot see.

        ``_drive_rollout`` lowers ``policy_running`` in a ``finally``, and the
        worker only returns - making its Future done and prunable - some time
        after that. Constructed directly rather than raced: a pending Future
        against a lowered claim IS that window, and "there was one" is the honest
        reading of it.
        """
        from concurrent.futures import Future

        pending: Future[None] = Future()
        sim._policy_threads["arm"] = pending
        assert sim._world.robots["arm"].policy_running is False
        try:
            assert sim._active_policy_robots() == ["arm"]
        finally:
            pending.set_result(None)

    def test_a_torn_down_world_answers_rather_than_raising(self, sim: Any) -> None:
        """The population is read by a status command and by a stop; both must answer."""
        sim._world = None
        assert sim._active_policy_robots() == []


class TestEveryReaderInheritsTheOneDefinition:
    """Four surfaces read this population. None of them re-derives it."""

    def test_every_reader_reports_a_blocking_rollout_as_in_flight(self, sim: Any) -> None:
        sim._announce_rollout("arm")
        readings = _readings(sim)
        assert readings["population"] == ["arm"]
        assert "arm" in readings["list_policies_running"]
        assert readings["mesh_status"] == {"status": "running", "robots_running": ["arm"]}
        assert readings["state_topic"] == ["arm"]

    def test_the_stop_verb_and_the_public_reader_agree_about_one_instant(self, sim: Any) -> None:
        """The invariant ``docs/simulation/overview.md`` states, at the instant it broke.

        ``stop_policy`` already unioned the claim in, so before the population
        did, these two answered "Stopped on 'arm'" with ``was_running=True`` and
        "No policies running." about the same robot at the same time.
        """
        sim._announce_rollout("arm")
        in_flight_reader = sim.list_policies_running()["content"][0]["text"]
        stopped = sim.stop_policy("arm")
        assert "arm" in in_flight_reader, in_flight_reader
        assert _rollout_json(stopped) == {"robot": "arm", "was_running": True}


class TestTheFleetStopHaltsABlockingRollout:
    """The safety half: the e-stop fanout must halt what it reports halting."""

    @staticmethod
    def _fanout(sim: Any) -> dict[str, Any]:
        """The wire shape ``Mesh.emergency_stop`` broadcasts: stop, no ``robot_name``."""
        return Mesh(sim, peer_id="sim-1", peer_type="simulation")._dispatch({"action": "stop"})

    def test_the_fanout_names_the_rollout_it_halted(self, sim: Any) -> None:
        thread, result = _blocking_rollout(sim)
        _await_claim(sim)

        answer = self._fanout(sim)

        assert answer["ok"] is True
        assert answer["stopped"] == ["arm"]
        assert _peers_that_did_not_stop([{"responder_id": "sim-1", "result": answer}]) == set()
        thread.join(timeout=30)
        assert not thread.is_alive(), "the rollout outlived the stop that reported halting it"

    def test_the_rollout_really_ends_early_rather_than_running_its_duration(self, sim: Any) -> None:
        """Read the rollout's own verdict, not the wall clock: a loaded host pacing
        the loop slowly must not read as a halt."""
        thread, result = _blocking_rollout(sim)
        _await_claim(sim)

        self._fanout(sim)
        thread.join(timeout=30)

        assert "envelope" in result, "the rollout never returned"
        rollout = _rollout_json(result["envelope"])
        assert rollout["stopped_early"] is True
        assert rollout["stopped_reason"] == "cancelled"
        assert rollout["steps_used"] < FULL_ROLLOUT_STEPS, rollout


class TestTheWideningIsNotOverBroad:
    """Widening a population that feeds an e-stop must not report a phantom rollout."""

    def test_every_reader_reports_an_idle_world_as_idle(self, sim: Any) -> None:
        readings = _readings(sim)
        assert readings["population"] == []
        assert readings["list_policies_running"] == "No policies running."
        assert readings["mesh_status"] == {"status": "idle", "robots_running": []}
        assert readings["state_topic"] == []

    def test_the_fleet_stop_still_says_so_when_nothing_runs(self, sim: Any) -> None:
        answer = Mesh(sim, peer_id="sim-1", peer_type="simulation")._dispatch({"action": "stop"})
        assert answer == {"ok": True, "stopped": [], "note": "no policies running"}

    def test_a_finished_blocking_rollout_leaves_the_population_empty(self, sim: Any) -> None:
        """The claim is lowered in a ``finally``, so a completed rollout is not
        reported forever."""
        sim.run_policy(robot_name="arm", policy_provider="mock", duration=0.5, control_frequency=CONTROL_HZ)
        assert sim._active_policy_robots() == []
        assert _readings(sim)["mesh_status"] == {"status": "idle", "robots_running": []}
