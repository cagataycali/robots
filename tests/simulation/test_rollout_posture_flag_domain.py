# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A rollout posture flag is checked, never read by truthiness.

``run_policy`` tables the domains of its quantities - the horizon, the rate, the
seed, the substeps - and read all three of its *postures* raw. A posture selects
a branch rather than scaling a quantity, so there is nothing to clamp and no
partial effect, and every non-empty string is truthy: ``"false"`` / ``"no"`` /
``"off"`` / ``"0"``, the spellings a caller reaches for when opting out, selected
the posture being opted out of, while ``0`` / ``""`` / ``None`` / ``[]`` selected
the other one without ever being a declared spelling of it. Measured on a MuJoCo
rollout, every row returning ``status="success"``:

* ``fast_mode="false"`` ran 30 steps at 30Hz in 0.006s where ``False`` takes
  1.001s - the whole rollout issued as one burst instead of paced on a deadline,
  which falsifies both claims the deadline pacer exists to hold (``duration`` in
  wall-clock seconds, ``fast_mode=False`` as real-time pacing);
* ``reset_between="no"`` reset between episodes, while ``0`` / ``""`` / ``None``
  / ``[]`` did not - so every episode after the first began in the state the
  previous one ended in and was still recorded as an independent episode;
* ``wbc_install_torque_control=None`` withheld the torque shim a position-servo
  humanoid needs for a stable gait, which is the one thing that flag installs.

The two flags that default to ``True`` are the sharper pair: for them a falsy
non-boolean REMOVES behaviour the rollout was going to get.

The same ``fast_mode`` field is already held to this domain by the mesh wire
schema (:func:`~strands_robots.mesh.security.validate_command`) before it is
forwarded into the same rollout, so a value an untrusted ``tell()`` could not get
past the transport was reachable from every local caller.

These tests pin that each flag is refused unless it is a boolean, on every
surface that reads it and in the shape that surface answers in - a structured
error from the public entry points, a ``ValueError`` from ``PolicyRunner.run``,
which is drivable directly - that the refusal precedes the effect the truthy
value had, that ``reset_between`` is not reported for a single-episode rollout
that never reads it, that the booleans still select the posture they name, and
that a fourth flag added to any of these signatures cannot skip the domain.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import re
import textwrap
from typing import Any

import pytest

import strands_robots.tools.run_policy as run_policy_tool
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.policy_runner import PolicyRunner
from strands_robots.utils import boolean_flag_error

pytest.importorskip("mujoco")

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots.mesh import pacing  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# One actuated hinge: enough to drive a rollout and to record a dataset column,
# with no asset download.
_ARM_XML = """
<mujoco model="posture_flag_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="base" pos="0 0 0.1">
      <joint name="pan" type="hinge" axis="0 0 1"/>
      <geom name="link" type="capsule" fromto="0 0 0 0.2 0 0" size="0.03"/>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="30"/>
  </actuator>
</mujoco>
"""

#: The postures of the rollout surface, and what each one gates.
ROLLOUT_POSTURES = ["fast_mode", "reset_between", "wbc_install_torque_control"]

#: Values no posture can be read from. The strings and the non-zero numbers are
#: the ones that selected the affirmative branch; the rest silently took the
#: other one without ever being a declared spelling of it.
UNUSABLE_FLAGS: list[Any] = ["false", "no", "off", "0", "true", 1, 0, "", None, [], float("nan")]

#: The spellings of *off* that are truthy, so each one selected *on*.
TRUTHY_SPELLINGS_OF_OFF = ["false", "no", "off", "0"]

#: Functions that read a posture flag, and the guard each routes it through.
#: ``SimEngine.start_policy`` is absent by delegation, pinned separately below.
CHECKING_SURFACES = {
    "SimEngine.run_policy": (SimEngine.run_policy, "_validate_posture_flag"),
    "MuJoCoSimulation.start_policy": (Simulation.start_policy, "_validate_posture_flag"),
    "PolicyRunner.run": (PolicyRunner.run, "boolean_flag_error"),
    "run_policy tool": (run_policy_tool.run_policy, "boolean_flag_error"),
}


@pytest.fixture
def sim(tmp_path):
    model = tmp_path / "posture_flag_arm.xml"
    model.write_text(_ARM_XML)
    s = Simulation(tool_name="posture_flag", mesh=False)
    s.create_world()
    s.add_robot("arm", urdf_path=str(model))
    yield s
    s.cleanup()


def _text(result: dict[str, Any]) -> str:
    return " ".join(c["text"] for c in result.get("content", []) if "text" in c)


def _steps_taken(sim) -> int:
    """Control steps the engine has advanced, off the public state report."""
    match = re.search(r"\(step (\d+)\)", _text(sim.get_state()))
    assert match, "the state report no longer carries a step count"
    return int(match.group(1))


def _rollout(sim, **kwargs: Any) -> dict[str, Any]:
    """A two-step rollout, fast so a paced default cannot dominate the suite."""
    params: dict[str, Any] = {
        "robot_name": "arm",
        "policy_provider": "mock",
        "n_steps": 2,
        "control_frequency": 200.0,
        "fast_mode": True,
    }
    params.update(kwargs)
    return sim.run_policy(**params)


def _bool_parameters(fn: Any) -> set[str]:
    """Every parameter of *fn* annotated exactly ``bool``.

    ``from __future__ import annotations`` makes every annotation a string, and
    :func:`inspect.signature` follows ``__wrapped__`` so a ``@tool`` reports the
    signature an agent calls. ``bool | None`` is deliberately excluded: that is a
    tri-state whose ``None`` is a documented spelling, a different domain.
    """
    return {
        name
        for name, param in inspect.signature(fn).parameters.items()
        if str(param.annotation) in ("bool", "<class 'bool'>")
    }


def _checked_parameters(fn: Any, guard: str) -> set[str]:
    """Every parameter *fn* routes through *guard* by name, read from its source."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(inspect.unwrap(fn))))
    checked: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != guard:
            continue
        first = node.args[0]
        if isinstance(first, ast.Name):
            checked.add(first.id)
    return checked


class TestTheDomainIsTheSharedOne:
    """The facade validator is the boolean-flag domain, and nothing more."""

    @pytest.mark.parametrize("value", UNUSABLE_FLAGS)
    @pytest.mark.parametrize("param", ROLLOUT_POSTURES)
    def test_a_non_boolean_reports_the_flag_the_method_and_the_value(self, param, value):
        error = SimEngine._validate_posture_flag(value, param, "run_policy")
        assert error is not None
        assert error["status"] == "error"
        text = error["content"][0]["text"]
        assert param in text
        assert text.startswith("run_policy: ")
        assert text == boolean_flag_error(value, param, "run_policy")

    @pytest.mark.parametrize("value", [True, False])
    @pytest.mark.parametrize("param", ROLLOUT_POSTURES)
    def test_a_boolean_is_accepted(self, param, value):
        assert SimEngine._validate_posture_flag(value, param, "run_policy") is None


class TestTheRolloutRefusesBeforeTheEffect:
    """Each posture is refused at the entry point, ahead of what it selected."""

    @pytest.mark.parametrize("value", TRUTHY_SPELLINGS_OF_OFF)
    @pytest.mark.parametrize("param", ROLLOUT_POSTURES)
    def test_a_truthy_spelling_of_off_is_refused_and_nothing_runs(self, sim, param, value):
        before = _steps_taken(sim)
        result = _rollout(sim, n_episodes=2, **{param: value})
        assert result["status"] == "error", result
        assert param in _text(result)
        assert _steps_taken(sim) == before, "the refused rollout advanced the sim"

    @pytest.mark.parametrize("value", [0, "", None, []])
    @pytest.mark.parametrize("param", ["reset_between", "wbc_install_torque_control"])
    def test_a_falsy_non_boolean_cannot_withhold_a_default_on_behaviour(self, sim, param, value):
        """Both flags default to True, so a falsy non-boolean REMOVED behaviour."""
        result = _rollout(sim, n_episodes=2, **{param: value})
        assert result["status"] == "error", result
        assert param in _text(result)

    def test_the_booleans_still_select_the_posture_they_name(self, sim, monkeypatch):
        """``reset_between`` and the WBC shim: True runs the gated call, False does not."""
        calls: dict[str, int] = {"reset": 0, "wbc": 0}
        real_reset, real_wbc = sim.reset, sim._maybe_install_wbc_torque_control
        monkeypatch.setattr(
            sim, "reset", lambda *a, **k: (calls.__setitem__("reset", calls["reset"] + 1), real_reset(*a, **k))[1]
        )
        monkeypatch.setattr(
            sim,
            "_maybe_install_wbc_torque_control",
            lambda *a, **k: (calls.__setitem__("wbc", calls["wbc"] + 1), real_wbc(*a, **k))[1],
        )

        assert _rollout(sim, n_episodes=2, reset_between=True, wbc_install_torque_control=True)["status"] == "success"
        assert (calls["reset"], calls["wbc"]) == (1, 1)

        calls.update(reset=0, wbc=0)
        assert _rollout(sim, n_episodes=2, reset_between=False, wbc_install_torque_control=False)["status"] == "success"
        assert (calls["reset"], calls["wbc"]) == (0, 0)

    def test_fast_mode_decides_whether_the_deadline_pacer_is_built(self, sim, monkeypatch):
        """The posture is a resource, not a sleep: False acquires a Ticker, True does not."""
        periods: list[float] = []
        real = pacing.Ticker
        monkeypatch.setattr(
            pacing, "Ticker", lambda period, *a, **k: (periods.append(period), real(period, *a, **k))[1]
        )

        assert _rollout(sim, fast_mode=True)["status"] == "success"
        assert periods == []
        assert _rollout(sim, fast_mode=False, control_frequency=500.0)["status"] == "success"
        assert periods == [pytest.approx(1.0 / 500.0)]

    def test_reset_between_is_not_reported_for_a_rollout_that_never_reads_it(self, sim):
        """A single-episode rollout consumes it nowhere, so it is not refused there."""
        assert _rollout(sim, n_episodes=1, reset_between="false")["status"] == "success"
        assert _rollout(sim, n_episodes=2, reset_between="false")["status"] == "error"


class TestTheRefusalTakesEachSurfacesOwnShape:
    """A structured error where a caller reads one; a raise where it cannot."""

    @pytest.mark.parametrize("value", TRUTHY_SPELLINGS_OF_OFF)
    def test_the_runner_raises_because_it_is_drivable_directly(self, sim, value):
        from strands_robots.policies.mock import MockPolicy

        with pytest.raises(ValueError, match="fast_mode"):
            PolicyRunner(sim).run("arm", MockPolicy(), n_steps=2, control_frequency=200.0, fast_mode=value)

    def test_start_policy_refuses_what_run_policy_refuses_and_claims_nothing(self, sim):
        refused = sim.start_policy(robot_name="arm", policy_provider="mock", n_steps=2, fast_mode="false")
        assert refused["status"] == "error"
        assert "fast_mode" in _text(refused)
        assert "No policies running" in _text(sim.list_policies_running())

    def test_the_tool_answers_with_an_envelope_rather_than_a_crash(self, sim):
        result = run_policy_tool.run_policy(
            simulation=sim, robot_name="arm", policy_provider="mock", n_episodes=1, n_steps=2, fast_mode="false"
        )
        assert result["status"] == "error"
        assert _text(result).startswith("run_policy: fast_mode must be a boolean")

    def test_the_tool_refuses_before_its_recording_step_replaces_a_dataset(self, sim, tmp_path):
        """The reason this tool checks rather than deferring to the rollout."""
        pytest.importorskip("lerobot")
        root = tmp_path / "dataset"
        assert sim.start_recording(repo_id="local/posture_flag", fps=30, root=str(root))["status"] == "success"
        assert _rollout(sim, n_steps=4, control_frequency=30.0)["status"] == "success"
        assert sim.stop_recording()["status"] == "success"

        def truth() -> tuple[int, int]:
            info = json.loads((root / "meta" / "info.json").read_text())
            return info["total_episodes"], info["total_frames"]

        assert truth() == (1, 4)
        result = run_policy_tool.run_policy(
            simulation=sim,
            robot_name="arm",
            policy_provider="mock",
            n_episodes=2,
            n_steps=4,
            control_frequency=30.0,
            dataset_fps=30,
            dataset_root=str(root),
            fast_mode="false",
        )
        assert result["status"] == "error"
        assert truth() == (1, 4), "the refused call replaced the dataset it was pointed at"


class TestEveryPostureFlagOfTheseSurfacesHasADomain:
    """Derived from each signature, so a fourth flag cannot skip the domain."""

    @pytest.mark.parametrize("surface", sorted(CHECKING_SURFACES))
    def test_every_boolean_parameter_is_routed_through_the_guard(self, surface):
        fn, guard = CHECKING_SURFACES[surface]
        declared = _bool_parameters(fn)
        assert declared, f"{surface} declares no bool parameter; the roster would grade nothing"
        assert declared == _checked_parameters(fn, guard), (
            f"{surface}: every bool parameter must be checked on the shared domain"
        )

    def test_start_policy_carries_no_posture_run_policy_does_not_check(self):
        """The base surface delegates, so its flags are covered by that check."""
        assert _bool_parameters(SimEngine.start_policy) <= _bool_parameters(SimEngine.run_policy)
