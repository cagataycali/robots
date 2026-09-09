"""A rollout can be stopped on every simulation backend, through one verb.

``stop_policy`` is the counterpart every ``start_policy`` docstring names, and
it lived on the MuJoCo engine alone. On the other backends the attribute did not
exist, so the two remote stop paths each worked around its absence rather than
reading its answer:

* :meth:`~strands_robots.mesh.Mesh._dispatch` probes ``hasattr(r,
  "stop_policy")`` and, failing it, answered ``"peer exposes no stop_task"`` for
  a simulation it could in fact have stopped.
* The Device Connect ``stop`` RPC re-derived the verdict inline from the
  per-robot flag - a second construction of the ``was_running`` answer that
  :meth:`~strands_robots.simulation.models.SimRobot.request_policy_stop` exists
  to keep in one place ("EVERY stop path goes through here ... so they cannot
  drift to different answers about whether a rollout was halted") - and read the
  robot registry as ``sim._world.robots``, which is the MuJoCo and Newton
  spelling. The Isaac engine keeps its robots in ``_robots`` and its ``_world``
  is the Isaac ``World``, which has no ``.robots``, so that read raised
  ``AttributeError`` and the handler's recovery path reported it as "the
  simulation changed under the stop loop" - a race that had not happened.

This is the promotion :meth:`~strands_robots.simulation.base.SimEngine.run_multi_policy`
already had (#2157), and ``tests/simulation/test_run_multi_policy_base_contract.py``
states its rule: a capability every backend is asked for answers in the tool
envelope on all of them, never with ``AttributeError`` because there was no
contract. The flag write stays backend-owned behind
:meth:`~strands_robots.simulation.base.SimEngine._request_policy_stop`, the
third member of the ``_make_run_policy_hook`` / ``_release_run_policy_hook``
seam that raises and lowers the same flag around a rollout the shared facade
drives.

The same divergence had a documentation half, pinned at the bottom: the base
``start_policy`` summary line promised "a background thread (non-blocking)" and
its next line said "synchronous passthrough to ``run_policy``", while
``docs/api-reference.md`` called it an async rollout unconditionally and
``docs/troubleshooting.md`` prescribed it as the fix for a hanging agent. On the
two backends shipped on that default it blocks for the whole ``duration``, so
the prescribed remedy was the hang.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.models import SimRobot, SimWorld

_DOCS = Path(__file__).resolve().parents[2] / "docs"
_JOINTS = ("j1", "j2")


class _MinimalEngine(SimEngine):
    """The smallest backend the ABC admits: no registry, no override.

    A third-party backend written against the documented ABC looks like this,
    and it is the population the base defaults have to answer for. Only the
    abstract methods are implemented; ``_request_policy_stop`` is deliberately
    NOT overridden, so this is also the "no durable claim" column.
    """

    def __init__(self) -> None:
        self._q: Any = np.zeros(len(_JOINTS))
        super().__init__()

    def create_world(
        self,
        timestep: float | None = None,
        gravity: list[float] | None = None,
        ground_plane: bool = True,
        terrain: str | None = None,
        difficulty: float = 1.0,
    ) -> dict[str, Any]:
        return {"status": "success", "content": []}

    def destroy(self) -> dict[str, Any]:
        return {"status": "success", "content": []}

    def reset(self) -> dict[str, Any]:
        return {"status": "success", "content": []}

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        return {"status": "success", "content": []}

    def get_state(self) -> dict[str, Any]:
        return {"status": "success", "content": []}

    def add_robot(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "success", "content": []}

    def remove_robot(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "success", "content": []}

    def list_robots(self) -> list[str]:
        return ["arm"]

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return list(_JOINTS)

    def add_object(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "success", "content": []}

    def remove_object(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "success", "content": []}

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
        return {"observation.state": self._q.copy()}

    def send_action(
        self,
        action: dict[str, Any] | Sequence[float],
        robot_name: str | None = None,
        n_substeps: int = 1,
    ) -> dict[str, Any]:
        values = [float(action[j]) for j in _JOINTS] if isinstance(action, dict) else [float(v) for v in action]
        self._q = np.asarray(values, dtype=float)
        return {"status": "success", "content": []}

    def render(
        self, camera_name: str = "default", width: int | None = None, height: int | None = None
    ) -> dict[str, Any]:
        return {"image": np.zeros((height or 48, width or 64, 3), dtype=np.uint8)}

    def physics_timestep(self) -> float | None:
        return 0.002


def _text(result: dict[str, Any]) -> str:
    return result["content"][0]["text"]


def _verdicts(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [b["json"] for b in result.get("content", []) if isinstance(b, dict) and isinstance(b.get("json"), dict)]


def _newton_engine(policy_running: bool) -> Any:
    """A Newton engine skeleton whose one robot carries the durable claim.

    ``__new__`` plus the handful of attributes the stop path reads - the fixture
    shape this backend's own recording tests use, so neither Warp nor a compiled
    model is needed to grade a flag write.
    """
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    world = SimWorld()
    robot = SimRobot(name="arm", urdf_path="arm.xml", data_config="so100", joint_names=list(_JOINTS))
    robot.policy_running = policy_running
    world.robots["arm"] = robot
    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    engine._world = world
    engine._model = object()
    return engine


# --------------------------------------------------------------------------- #
# The verb exists on every backend                                            #
# --------------------------------------------------------------------------- #
class TestTheVerbIsOnTheBase:
    """``stop_policy`` answers in the envelope everywhere, never AttributeError."""

    def test_the_base_engine_declares_the_counterpart_start_policy_names(self) -> None:
        assert hasattr(SimEngine, "stop_policy")

    @pytest.mark.parametrize(
        "module_path,class_name",
        [
            ("strands_robots.simulation.newton.simulation", "NewtonSimEngine"),
            ("strands_robots.simulation.isaac.simulation", "IsaacSimulation"),
        ],
    )
    def test_a_backend_without_its_own_override_inherits_one(self, module_path: str, class_name: str) -> None:
        """These two shipped on the base default, so they had no verb at all."""
        import importlib

        engine_cls = getattr(importlib.import_module(module_path), class_name)
        assert hasattr(engine_cls, "stop_policy")


# --------------------------------------------------------------------------- #
# The base default: a stated refusal, not a guess                             #
# --------------------------------------------------------------------------- #
class TestTheBaseDefaultRefusesWithAReachableRemedy:
    """No durable claim to move means no verdict, and saying so beats guessing."""

    def test_a_backend_with_no_claim_refuses_and_names_its_class(self) -> None:
        result = _MinimalEngine().stop_policy("arm")
        assert result["status"] == "error"
        assert "_MinimalEngine keeps no durable per-robot rollout claim" in _text(result)

    def test_the_refusal_names_remedies_this_engine_actually_has(self) -> None:
        """A remedy naming a verb the backend lacks is what this replaces."""
        engine = _MinimalEngine()
        text = _text(engine.stop_policy("arm"))
        assert "n_steps" in text
        assert "stop_when" in text
        for named in ("run_policy", "_request_policy_stop"):
            assert hasattr(engine, named), f"the refusal names {named!r}, which must exist here"

    def test_nothing_running_is_not_reported_as_the_answer(self) -> None:
        """``was_running=False`` here would be an affirmative claim on no evidence."""
        assert _verdicts(_MinimalEngine().stop_policy("arm")) == []

    def test_an_empty_robot_name_is_refused_not_matched_to_the_sole_robot(self) -> None:
        result = _MinimalEngine().stop_policy("")
        assert result["status"] == "error"
        assert "requires 'robot_name'" in _text(result)

    def test_an_unknown_robot_is_named_before_any_backend_is_asked(self) -> None:
        result = _MinimalEngine().stop_policy("ghost")
        assert result["status"] == "error"
        assert "Robot 'ghost' not found." in _text(result)


# --------------------------------------------------------------------------- #
# A backend that holds the durable claim answers from it                      #
# --------------------------------------------------------------------------- #
class TestNewtonAnswersFromTheDurableClaim:
    """Newton's robots are ``SimRobot``s, so its stop is a real one."""

    def test_a_rollout_in_flight_is_halted_and_reported(self) -> None:
        engine = _newton_engine(policy_running=True)
        result = engine.stop_policy("arm")
        assert result["status"] == "success"
        assert _verdicts(result) == [{"robot": "arm", "was_running": True}]
        assert engine._world.robots["arm"].policy_running is False

    def test_the_stop_is_durable_not_only_a_lowered_flag(self) -> None:
        """A bare flag write is what a worker before its first frame overwrites (#2833)."""
        engine = _newton_engine(policy_running=True)
        before = engine._world.robots["arm"].policy_stops
        engine.stop_policy("arm")
        assert engine._world.robots["arm"].policy_stops == before + 1

    def test_stopping_twice_is_idempotent_and_honest_the_second_time(self) -> None:
        engine = _newton_engine(policy_running=True)
        assert _verdicts(engine.stop_policy("arm")) == [{"robot": "arm", "was_running": True}]
        again = engine.stop_policy("arm")
        assert again["status"] == "success"
        assert _verdicts(again) == [{"robot": "arm", "was_running": False}]

    def test_a_torn_down_world_has_no_verdict_rather_than_an_idle_one(self) -> None:
        engine = _newton_engine(policy_running=True)
        engine._world = None
        assert engine._request_policy_stop("arm") is None


# --------------------------------------------------------------------------- #
# The Device Connect stop reads the verb instead of working around it          #
# --------------------------------------------------------------------------- #
class TestTheStopRpcReadsTheVerb:
    """One owner for the verdict, and one accessor for the population."""

    @staticmethod
    def _driver(sim: Any) -> Any:
        from strands_robots.device_connect.sim_driver import SimulationDeviceDriver

        driver = SimulationDeviceDriver.__new__(SimulationDeviceDriver)
        driver._sim = sim
        return driver

    def test_one_rollout_is_stopped_through_the_verb_and_its_answer_returned(self) -> None:
        engine = _newton_engine(policy_running=True)
        answer = self._driver(engine)._stop_one_rollout("arm")
        assert answer["status"] == "success"
        assert _verdicts(answer) == [{"robot": "arm", "was_running": True}]

    def test_the_population_comes_from_list_robots_not_a_private_registry(self) -> None:
        """An engine whose ``_world`` has no ``.robots`` is the Isaac shape.

        The private read raised ``AttributeError`` here, which the ``stop``
        handler's recovery path reported as a scene teardown racing the loop.
        """

        class _IsaacShapedSim:
            def __init__(self) -> None:
                self._world = object()  # the Isaac ``World``: no ``.robots``
                self._robots = {"arm": object()}

            def list_robots(self) -> list[str]:
                return list(self._robots)

        assert self._driver(_IsaacShapedSim())._stop_targets() == ["arm"]

    def test_a_backend_that_refuses_is_carried_through_as_a_refusal(self) -> None:
        """The handler grades one envelope shape whatever the backend answers."""
        from strands_robots.mesh.core import _reports_failure_to_stop

        answer = self._driver(_MinimalEngine())._stop_one_rollout("arm")
        assert answer["status"] == "error"
        assert _reports_failure_to_stop(answer) is True


# --------------------------------------------------------------------------- #
# describe(): which of the two start_policy implementations you hold          #
# --------------------------------------------------------------------------- #
class TestDescribeSaysWhichStartPolicyYouHold:
    """The surface an agent is told to call first, instead of guessing."""

    def test_the_base_entry_states_the_synchronous_reading(self) -> None:
        entry = _MinimalEngine().describe()["methods"]["start_policy"]
        assert "synchronous" in entry
        assert "runs the rollout to completion" in entry

    def test_the_base_surface_advertises_the_counterpart_too(self) -> None:
        methods = _MinimalEngine().describe()["methods"]
        assert "robot_name" in methods["stop_policy"]
        assert "was_running" in methods["stop_policy"]

    def test_the_entry_names_the_signal_that_tells_the_two_apart(self) -> None:
        """MuJoCo's own describe cell pins the other column of this comparison."""
        entry = _MinimalEngine().describe()["methods"]["start_policy"]
        assert "list_policies_running" in entry


# --------------------------------------------------------------------------- #
# The documented surfaces say the same thing the code does                     #
# --------------------------------------------------------------------------- #
def _selected_actions() -> list[str]:
    """Action names from the SimEngine "Selected actions" table in api-reference."""
    lines = (_DOCS / "api-reference.md").read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "Selected actions:")
    names: list[str] = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        if not line.startswith("|"):
            continue
        cell = line.split("|")[1]
        names.extend(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)\(", cell))
    return names


class TestTheDocumentedActionsResolve:
    """A row in the SimEngine action table names a method callers really have."""

    def test_the_table_is_still_being_read(self) -> None:
        """A broken parse would make the rule below vacuously green."""
        actions = _selected_actions()
        assert len(actions) >= 8, actions
        assert "stop_policy" in actions

    @pytest.mark.parametrize("action", _selected_actions())
    def test_every_documented_action_is_on_the_base_engine(self, action: str) -> None:
        assert hasattr(SimEngine, action), (
            f"docs/api-reference.md lists {action!r} as a SimEngine action, but it does not resolve "
            "there - a caller following the table gets AttributeError on every backend that does "
            "not happen to override it"
        )


#: A surface makes the claim either by describing start_policy as not blocking,
#: or by PRESCRIBING it as the cure for something that blocks - which is the
#: form the troubleshooting table used ("Agent hangs ... Use start_policy
#: instead of run_policy"), and it carries the same promise without the words.
_BACKGROUND_CLAIM = re.compile(r"background|non-blocking|\basync\b|\bhang|instead of", re.IGNORECASE)
_CONDITION = re.compile(r"MuJoCo|backend", re.IGNORECASE)


def _start_policy_claims() -> list[tuple[str, str]]:
    """``(surface, text)`` for each documented place that describes start_policy.

    The base docstring's SUMMARY LINE is graded on its own, because that line is
    what ``help()``, an IDE tooltip and the rendered API reference show, and it
    is where the promise and the body disagreed: "Start policy execution in a
    background thread (non-blocking)." followed by "Default implementation:
    synchronous passthrough". Grading the whole docstring would have read the
    qualification three paragraphs down as if the summary carried it. The rest of
    the body is graded too, plus every markdown line naming ``start_policy``.
    """
    doc = SimEngine.start_policy.__doc__ or ""
    summary, _, body = doc.strip().partition("\n")
    claims: list[tuple[str, str]] = [
        ("SimEngine.start_policy docstring summary line", summary),
        ("SimEngine.start_policy docstring body", body),
    ]
    for name in ("api-reference.md", "troubleshooting.md"):
        for number, line in enumerate((_DOCS / name).read_text(encoding="utf-8").splitlines(), start=1):
            if "start_policy" in line:
                claims.append((f"docs/{name}:{number}", line))
    return claims


class TestNoSurfacePromisesABackgroundThreadUnconditionally:
    """A claim that start_policy does not block must say where that is true."""

    def test_the_surfaces_are_still_being_found(self) -> None:
        surfaces = [name for name, _ in _start_policy_claims()]
        assert len(surfaces) >= 4, surfaces
        assert any("api-reference" in name for name in surfaces)
        assert any("summary line" in name for name in surfaces)

    @pytest.mark.parametrize("surface,text", _start_policy_claims())
    def test_a_background_claim_names_the_condition(self, surface: str, text: str) -> None:
        if not _BACKGROUND_CLAIM.search(text):
            return
        assert _CONDITION.search(text), (
            f"{surface} presents start_policy as a call that does not block, without naming the "
            "backend condition; it blocks for the whole duration on every engine that ships on "
            "the base default, so an unconditional claim points a caller at the hang"
        )
