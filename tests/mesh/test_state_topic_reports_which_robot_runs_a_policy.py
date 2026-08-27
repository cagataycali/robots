"""The state topic's per-robot ``active`` flag is measured, not asserted.

``Mesh._read_state`` publishes a ``robots`` section naming every robot in the
sim world. The flag beside each name used to be the literal ``True``, so it was
constant for the life of the peer: a scene's idle arms and the one arm executing
a rollout were indistinguishable at 10 Hz, and the flag could not change when a
policy started or stopped.

The ``status`` command already answers that question from the running-policy
registry (``robots_running``), so the two surfaces disagreed about one fact.
These cases hold them to the same source: whatever set the flag marks active is
the set ``status`` reports running, in every phase of a real rollout, from a
parent ``Simulation`` peer and from a ``SimRobot`` child peer alike.

Also pinned, so the measurement cannot be traded for the old shortcut: a peer
that keeps no policy registry still gets its ``robots`` section, and a registry
that raises is named in ``degraded`` rather than answered with a fabricated
flag.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from strands_robots.mesh import core as mesh_core

_START_TIMEOUT_S = 10.0
_RUNNER = "so100"
_BYSTANDER = "arm_b"


def _active_names(snapshot: dict[str, Any] | None) -> set[str]:
    """The robots a state snapshot marks active, as a set of names."""
    assert snapshot is not None, "the peer published nothing on the state topic"
    robots = snapshot.get("robots")
    assert isinstance(robots, dict), f"no robots section in the snapshot: {sorted(snapshot)}"
    return {name for name, entry in robots.items() if entry["active"]}


def _await(predicate: Any, what: str) -> None:
    """Block until ``predicate()`` holds, so a phase is entered deterministically."""
    deadline = time.time() + _START_TIMEOUT_S
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


def _start_rollout(sim: Any) -> None:
    """Start a mock rollout on ``_RUNNER`` and wait for its first control step."""
    sim.start_policy(robot_name=_RUNNER, policy_provider="mock", duration=30.0, control_frequency=20.0)
    _await(
        lambda: getattr(sim._world.robots[_RUNNER], "policy_steps", 0) > 0,
        "the rollout to take its first control step",
    )


def _stop_rollout(sim: Any) -> None:
    """Stop the rollout and wait for the registry to drop it."""
    sim.stop_policy(_RUNNER)
    _await(lambda: not sim._active_policy_robots(), "the rollout to leave the registry")


@pytest.fixture
def two_robot_sim() -> Iterator[Any]:
    """A live MuJoCo scene holding two robots, one of which will run a policy."""
    pytest.importorskip("mujoco", reason="the flag is read off a live MuJoCo world")
    import strands_robots

    sim = strands_robots.Robot(_RUNNER, mode="sim")
    sim.add_robot(_BYSTANDER, data_config="so101", position=[0.6, 0.0, 0.0])
    try:
        yield sim
    finally:
        sim.cleanup()


class TestTheFlagTracksTheRunningPolicy:
    """The regression: the flag distinguishes a rollout from an idle scene."""

    def test_the_scene_holds_two_robots_so_the_flag_has_something_to_tell_apart(self, two_robot_sim: Any) -> None:
        """Premise. With one robot, "exactly one is active" is true by arity."""
        assert set(two_robot_sim._world.robots) == {_RUNNER, _BYSTANDER}

    def test_no_robot_is_active_while_the_scene_is_idle(self, two_robot_sim: Any) -> None:
        mesh = mesh_core.Mesh(two_robot_sim, peer_id="sim-idle")
        assert _active_names(mesh._read_state()) == set()

    def test_only_the_robot_running_a_policy_is_active(self, two_robot_sim: Any) -> None:
        mesh = mesh_core.Mesh(two_robot_sim, peer_id="sim-running")
        _start_rollout(two_robot_sim)
        try:
            # Independent of the registry the flag is read from: the runner's
            # rollout really advanced, so "active" describes a live policy.
            assert two_robot_sim._world.robots[_RUNNER].policy_steps > 0
            assert _active_names(mesh._read_state()) == {_RUNNER}
        finally:
            _stop_rollout(two_robot_sim)

    def test_no_robot_is_active_once_the_policy_stops(self, two_robot_sim: Any) -> None:
        mesh = mesh_core.Mesh(two_robot_sim, peer_id="sim-stopped")
        _start_rollout(two_robot_sim)
        assert _active_names(mesh._read_state()) == {_RUNNER}
        _stop_rollout(two_robot_sim)
        assert _active_names(mesh._read_state()) == set()


class TestTheStateTopicAndTheStatusCommandAgree:
    """One fact, one source: the flag and ``robots_running`` never disagree."""

    def test_they_agree_in_every_phase_of_a_rollout(self, two_robot_sim: Any) -> None:
        mesh = mesh_core.Mesh(two_robot_sim, peer_id="sim-agree")

        def both() -> tuple[set[str], set[str]]:
            return (
                _active_names(mesh._read_state()),
                set(mesh._dispatch({"action": "status"})["robots_running"]),
            )

        seen = []
        flagged, running = both()
        assert flagged == running
        seen.append(running)

        _start_rollout(two_robot_sim)
        try:
            flagged, running = both()
            assert flagged == running
            seen.append(running)
        finally:
            _stop_rollout(two_robot_sim)

        flagged, running = both()
        assert flagged == running
        seen.append(running)

        # Non-vacuity: agreeing on the empty set in all three phases would pass
        # the equality above while measuring nothing.
        assert seen == [set(), {_RUNNER}, set()]

    def test_a_child_peer_reports_the_same_map_as_its_parent(self, two_robot_sim: Any) -> None:
        """A ``SimRobot`` peer holds no registry and consults ``_sim_parent``.

        This is the wiring ``Simulation._attach_robot_to_mesh`` installs when a
        per-robot peer is published, so the child peer answers for the whole
        world exactly as the parent does.
        """
        child = two_robot_sim._world.robots[_BYSTANDER]
        child._world = two_robot_sim._world
        child._sim_parent = two_robot_sim
        parent_mesh = mesh_core.Mesh(two_robot_sim, peer_id="sim-parent")
        child_mesh = mesh_core.Mesh(child, peer_id="sim-child")

        _start_rollout(two_robot_sim)
        try:
            assert _active_names(parent_mesh._read_state()) == {_RUNNER}
            assert _active_names(child_mesh._read_state()) == {_RUNNER}
        finally:
            _stop_rollout(two_robot_sim)


class _WorldStub:
    """The two attributes ``_read_state``'s sim-world probe reads."""

    def __init__(self) -> None:
        self._data = SimpleNamespace(time=12.5)
        self._model = None
        self.robots = {"arm0": object(), "arm1": object()}


class _NoRegistryRobot:
    """A peer holding a sim world but keeping no running-policy registry."""

    tool_name_str = "worldbot"

    def __init__(self) -> None:
        self._world = _WorldStub()


class _DataAttributeRobot(_NoRegistryRobot):
    """A peer carrying the registry's name as data rather than as a method."""

    _active_policy_robots: list[str] = ["arm0"]


class _RaisingRegistryRobot(_NoRegistryRobot):
    """A peer whose running-policy registry cannot be read."""

    def _active_policy_robots(self) -> list[str]:
        raise RuntimeError("policy registry unreadable")


class TestAPeerThatKeepsNoPolicyRegistry:
    """Deriving the flag must not cost the section, nor invent a rollout."""

    def test_its_robots_section_is_still_published(self) -> None:
        snapshot = mesh_core.Mesh(_NoRegistryRobot(), peer_id="no-registry")._read_state()
        assert snapshot is not None
        assert set(snapshot["robots"]) == {"arm0", "arm1"}
        assert snapshot["sim_time"] == 12.5

    def test_none_of_its_robots_is_reported_active(self) -> None:
        """No policy runs through the simulation API on such a peer."""
        assert _active_names(mesh_core.Mesh(_NoRegistryRobot(), peer_id="no-reg2")._read_state()) == set()

    def test_the_registry_name_carried_as_data_is_not_a_registry(self) -> None:
        """Only something callable is consulted, so a peer that happens to carry
        the name as an attribute reads as keeping no registry rather than having
        its names taken for a rollout.
        """
        snapshot = mesh_core.Mesh(_DataAttributeRobot(), peer_id="data-attr")._read_state()
        assert _active_names(snapshot) == set()
        assert snapshot is not None
        assert set(snapshot["robots"]) == {"arm0", "arm1"}


class TestARaisingRegistryIsReportedRatherThanSubstituted:
    """The state topic names a failing probe; it does not publish a guess."""

    def test_the_failure_is_named_in_degraded(self) -> None:
        snapshot = mesh_core.Mesh(_RaisingRegistryRobot(), peer_id="raising")._read_state()
        assert snapshot is not None
        assert snapshot["degraded"]["sim_world"]["reason"] == "RuntimeError"
        assert "policy registry unreadable" in snapshot["degraded"]["sim_world"]["detail"]

    def test_no_flag_is_published_for_a_probe_that_could_not_answer(self) -> None:
        snapshot = mesh_core.Mesh(_RaisingRegistryRobot(), peer_id="raising2")._read_state()
        assert snapshot is not None
        assert "robots" not in snapshot
        # The section handler still keeps what the probe had already read.
        assert snapshot["sim_time"] == 12.5
