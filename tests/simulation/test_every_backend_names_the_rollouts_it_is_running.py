# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: ``list_policies_running`` answers on every backend.

``docs/simulation/overview.md`` lists ``list_policies_running`` in the Policy
action table beside ``run_policy`` / ``start_policy`` / ``stop_policy``, with no
backend qualifier, and documents ``stop_policy`` -- a base contract since a
robot's stop was promoted to the ABC -- as deriving its verdict from "the same
in-flight population ``list_policies_running`` reads", so "the two never report
opposite facts about the same robot at the same instant".
``docs/device-connect.md`` makes the same promise for the Device Connect stop,
whose sim driver runs on any backend.

Only one of that documented pair existed off MuJoCo. Measured on a real Newton
world driving ``so101``, in all three phases of a rollout::

    AttributeError: 'NewtonSimEngine' object has no attribute 'list_policies_running'

The population it needs is on the ABC -- ``SimEngine._rollouts_in_flight``, the
seam the mesh's own reporting surfaces read -- so the verb is answered there and
every backend inherits it. MuJoCo's copy is gone: its override of that seam
delegates to ``_active_policy_robots``, so the population, and the Future prune
that reader performs, are unchanged.

The rollouts here are real: each behavioural case drives ``run_policy`` with a
``mock`` policy and reads the verb from the ``observer`` lane, so the claim under
the reading is the one the backend's own rollout hook raised.

Also pinned: a backend reporting no population at all must not answer "No
policies running.". That is an affirmative claim about robots it cannot see, and
it is the same claim the state topic was refused for publishing as
``active=false``. The base refuses instead and names the seam to override, which
is what ``stop_policy`` does for a halt it cannot stand behind.
"""

from __future__ import annotations

import importlib.util
import textwrap
import types
from typing import Any

import pytest

from strands_robots.simulation.base import SimEngine

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None

_ROBOT = "so101"

_BACKENDS = [
    pytest.param("mujoco", id="mujoco"),
    pytest.param(
        "newton",
        id="newton",
        marks=pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed"),
    ),
]


class _NoRegistry:
    """A backend reporting no in-flight population - the ABC's own default."""

    def _rollouts_in_flight(self) -> tuple[str, ...] | None:
        return None


class _IdleRegistry:
    """A backend reporting a population that happens to be empty."""

    def _rollouts_in_flight(self) -> tuple[str, ...] | None:
        return ()


def _text(envelope: dict[str, Any]) -> str:
    return str(envelope["content"][0]["text"])


def _readings(backend: str) -> dict[str, Any]:
    """Ask the verb before, during and after a real rollout."""
    from strands_robots.simulation import create_simulation

    sim = create_simulation(backend=backend)
    try:
        sim.create_world(ground_plane=True)
        sim.add_robot(_ROBOT)
        idle = dict(sim.list_policies_running())
        during: list[dict[str, Any]] = []

        def observe(_event: Any) -> None:
            if not during:
                during.append(dict(sim.list_policies_running()))

        rollout = sim.run_policy(
            robot_name=_ROBOT,
            policy_provider="mock",
            n_steps=6,
            control_frequency=20.0,
            fast_mode=True,
            observer=observe,
        )
        assert rollout.get("status") == "success", f"the rollout itself failed: {rollout}"
        assert during, "the observer never ran, so nothing was read mid-rollout"
        return {"idle": idle, "in_flight": during[0], "after": dict(sim.list_policies_running())}
    finally:
        sim.destroy()


@pytest.fixture(scope="module")
def readings() -> dict[str, dict[str, Any]]:
    return {}


def _for(readings: dict[str, dict[str, Any]], backend: str) -> dict[str, Any]:
    if backend not in readings:
        readings[backend] = _readings(backend)
    return readings[backend]


@pytest.mark.parametrize("backend", _BACKENDS)
class TestTheVerbAnswersOnEveryBackend:
    """It used to raise ``AttributeError`` off MuJoCo."""

    def test_a_rollout_in_flight_is_named(self, backend: str, readings: Any) -> None:
        answer = _for(readings, backend)["in_flight"]
        assert answer["status"] == "success"
        assert _ROBOT in _text(answer)

    def test_an_idle_world_reports_none_running(self, backend: str, readings: Any) -> None:
        assert _text(_for(readings, backend)["idle"]) == "No policies running."

    def test_a_finished_rollout_leaves_the_population(self, backend: str, readings: Any) -> None:
        """MuJoCo reached this through a Future prune; the delegation keeps it."""
        assert _text(_for(readings, backend)["after"]) == "No policies running."

    def test_the_two_phases_are_distinguishable(self, backend: str, readings: Any) -> None:
        r = _for(readings, backend)
        assert r["in_flight"] != r["idle"]


@pytest.mark.parametrize("backend", _BACKENDS)
class TestTheVerbAndTheMeshAgree:
    """The docs promise one population behind both surfaces."""

    def test_a_rollout_named_by_the_verb_is_reported_by_the_status_command(self, backend: str) -> None:
        from strands_robots.mesh import Mesh
        from strands_robots.simulation import create_simulation

        sim = create_simulation(backend=backend)
        try:
            sim.create_world(ground_plane=True)
            sim.add_robot(_ROBOT)
            mesh = Mesh(robot=sim, peer_id=f"{backend}-agreement")
            seen: list[tuple[str, list[str]]] = []

            def observe(_event: Any) -> None:
                if not seen:
                    verb = _text(sim.list_policies_running())
                    status = mesh._dispatch({"action": "status"})
                    seen.append((verb, list(status.get("robots_running", []))))

            sim.run_policy(
                robot_name=_ROBOT,
                policy_provider="mock",
                n_steps=6,
                control_frequency=20.0,
                fast_mode=True,
                observer=observe,
            )
            assert seen, "the observer never ran"
            verb_text, robots_running = seen[0]
            assert robots_running == [_ROBOT]
            for name in robots_running:
                assert name in verb_text, f"{name!r} is on the wire but not in {verb_text!r}"
        finally:
            sim.destroy()


class TestABackendWithNoPopulationDoesNotClaimThereIsNone:
    """ "No policies running" is an affirmative claim, so it needs evidence."""

    def test_it_is_refused_rather_than_reported_as_idle(self) -> None:
        answer = SimEngine.list_policies_running(_NoRegistry())  # type: ignore[arg-type]
        assert answer["status"] == "error"
        assert "No policies running." not in _text(answer)

    def test_the_refusal_names_the_backend_and_the_seam_to_override(self) -> None:
        answer = SimEngine.list_policies_running(_NoRegistry())  # type: ignore[arg-type]
        assert "_NoRegistry" in _text(answer)
        assert "_rollouts_in_flight" in _text(answer)

    def test_an_empty_population_is_still_reported_as_idle(self) -> None:
        """The refusal is for an absent verdict, not for an idle world."""
        answer = SimEngine.list_policies_running(_IdleRegistry())  # type: ignore[arg-type]
        assert answer["status"] == "success"
        assert _text(answer) == "No policies running."


class TestIsaacAnswersItToo:
    """Isaac needs no Isaac Sim here: the population is its own registry."""

    @staticmethod
    def _stub(running: dict[str, bool]) -> Any:
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        stub = types.SimpleNamespace(
            _world_created=True,
            _robots={n: types.SimpleNamespace(policy_running=f) for n, f in running.items()},
        )
        # The verb lives on the ABC and reads the seam, so the stand-in answers
        # it with Isaac's own reader - both halves are the production code.
        stub._rollouts_in_flight = lambda: IsaacSimulation._rollouts_in_flight(stub)  # type: ignore[arg-type]
        return stub

    def test_it_names_the_robot_holding_the_claim(self) -> None:
        stub = self._stub({"idle_arm": False, "busy_arm": True})
        answer = SimEngine.list_policies_running(stub)
        assert answer["status"] == "success"
        assert "busy_arm" in _text(answer)
        assert "idle_arm" not in _text(answer)

    def test_an_idle_isaac_world_reports_none_running(self) -> None:
        assert _text(SimEngine.list_policies_running(self._stub({"arm": False}))) == "No policies running."

    def test_before_a_world_exists_it_is_refused_rather_than_reported_as_idle(self) -> None:
        """Isaac's reader answers ``None`` before ``create_world``, so this is
        the refusal branch reached by a shipped backend rather than only by a
        stand-in: there is no world whose rollouts could be enumerated, and
        "no policies running" would be a claim about robots that do not exist.
        """
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        stub = types.SimpleNamespace(_world_created=False, _robots={})
        stub._rollouts_in_flight = lambda: IsaacSimulation._rollouts_in_flight(stub)  # type: ignore[arg-type]
        answer = SimEngine.list_policies_running(stub)  # type: ignore[arg-type]
        assert answer["status"] == "error"
        assert "No policies running." not in _text(answer)


class TestThePopulationIsSpelledOnce:
    """One reader, so the verb and the mesh cannot drift (the #2833 rule)."""

    def test_no_backend_re_implements_the_verb(self) -> None:
        from strands_robots.simulation.isaac.simulation import IsaacSimulation
        from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine
        from strands_robots.simulation.newton.simulation import NewtonSimEngine

        base = SimEngine.list_policies_running
        overriding = [
            engine.__name__
            for engine in (MuJoCoSimEngine, NewtonSimEngine, IsaacSimulation)
            if engine.list_policies_running is not base
        ]
        assert not overriding, f"these engines answer it with their own copy: {overriding}"

    def test_it_reads_the_seam_rather_than_one_engine_s_registry(self) -> None:
        import ast
        import inspect

        tree = ast.parse(textwrap.dedent(inspect.getsource(SimEngine.list_policies_running)))
        called = {
            ast.unparse(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "self._rollouts_in_flight" in called, f"it reads something else: {sorted(called)}"
        assert "self._active_policy_robots" not in called, (
            "reading the MuJoCo-only registry is what made this verb backend-specific"
        )
