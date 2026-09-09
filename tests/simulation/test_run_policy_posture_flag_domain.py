# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rollout facades check their posture flags instead of reading them by truthiness.

:meth:`SimEngine.run_policy` takes four flags that each select one of two
postures: ``fast_mode`` (pace the loop at ``control_frequency`` or run it
unpaced), ``reset_between`` (reset the scene between episodes or carry the end
state over), ``wbc_install_torque_control`` (install the WBC torque shim for the
call or leave the actuators alone) and ``async_rtc`` (overlap inference with
actuation on a background thread, drain each chunk first, or ``None`` to resolve
from the policy). :meth:`SimEngine.eval_policy` takes ``async_rtc`` as a plain
``bool``, and MuJoCo's :meth:`start_policy` takes ``fast_mode`` and submits the
rollout to a worker. Every one of them was read by truthiness, so every
non-empty string selected the posture the word asks to skip, and every falsy
non-boolean took the other branch without being a declared spelling of it.
Measured on ``main`` at ``a03d98d``, driving the real facade over a mock policy,
three steps at 50 Hz:

| call | selected | asked for |
|---|---|---|
| ``run_policy(fast_mode="false")`` | pacer never acquired | a paced loop |
| ``run_policy(fast_mode=0)`` | pacer acquired | (not a declared spelling) |
| ``run_policy(n_episodes=2, reset_between=0)`` | no reset between episodes | (not a declared spelling) |
| ``run_policy(wbc_install_torque_control=0)`` | shim not installed | (not a declared spelling) |
| ``run_policy(async_rtc="false")`` | ``rtc_async_enabled=True`` | the synchronous loop |

Every row reported ``status="success"``. The ``async_rtc`` row is the sharpest:
the word for "no" started the background inference thread, and an evaluation's
success rate carries no field saying which pipeline produced it. The
``reset_between`` row is the one with a physical consequence - episode two
started from wherever episode one left the arm, and the dataset those two
episodes were flushed into does not record that.

The fix binds the shared :func:`~strands_robots.utils.boolean_flag_error`
domain to the facade's error envelope (``SimEngine._validate_posture_flags``),
the way the numeric knobs beside these flags are already bound, and checks the
flags ahead of robot resolution so a refused call builds no policy and touches
no scene. ``start_policy`` checks ``fast_mode`` before the submit, because a
refusal on the worker is discarded with the future and the caller reads
"started". The ``run_policy`` *tool* checks ``fast_mode`` before it starts the
recording it was asked to make, for the reason its own pre-flight block states:
the facade's refusal would otherwise arrive after the dataset at ``dataset_root``
had been replaced with an empty one.

``async_rtc=None`` on ``run_policy`` is the documented "resolve from the policy"
spelling and is left alone; only a supplied value is held to the domain.
"""

from __future__ import annotations

import inspect
import math
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import strands_robots.mesh.pacing as pacing_mod
import strands_robots.tools.run_policy as rp_mod
from strands_robots.simulation.base import SimEngine
from strands_robots.utils import boolean_flag_error

#: Truthy spellings of "off": each selected the posture the word asks to skip.
TRUTHY_NON_BOOLEANS: tuple[Any, ...] = ("false", "no", "off", "0", 1, 2, math.nan)
#: Falsy non-booleans: each took the other branch without being a spelling of it.
FALSY_NON_BOOLEANS: tuple[Any, ...] = (0, 0.0, "", [], {})
#: ``None`` is a declared sentinel for ``async_rtc`` on ``run_policy`` and for
#: nothing else here, so it is parametrized separately.
UNDECLARED: tuple[Any, ...] = (*TRUTHY_NON_BOOLEANS, *FALSY_NON_BOOLEANS)

#: The posture flags ``SimEngine.run_policy`` reads, keyed to the branch each
#: selects, so the roster below can be checked against the signature.
RUN_POLICY_FLAGS: tuple[str, ...] = ("fast_mode", "reset_between", "wbc_install_torque_control", "async_rtc")


class _Sim(SimEngine):
    """Minimal ``SimEngine`` that records the branch each flag selects.

    It counts ``reset`` calls (the ``reset_between`` branch), the torque-shim
    installs (the ``wbc_install_torque_control`` branch) and ``send_action``
    calls (whether a rollout ran at all). The pacer is observed by replacing the
    ``Ticker`` class the runner imports lazily, since ``fast_mode`` is read as
    "acquire one or do not".
    """

    def __init__(self) -> None:
        self._robots = {"arm": ["j0", "j1", "j2"]}
        self._lock = threading.Lock()
        self.resets = 0
        self.sends = 0
        self.shim_installs = 0

    def create_world(self, timestep=None, gravity=None, ground_plane=True):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def destroy(self):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def reset(self):  # type: ignore[no-untyped-def]
        self.resets += 1
        return {"status": "success"}

    def step(self, n_steps: int = 1):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def get_state(self):  # type: ignore[no-untyped-def]
        return {"sim_time": 0.0, "step_count": self.sends}

    def add_robot(self, name, **kw):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def remove_robot(self, name):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def list_robots(self) -> list[str]:
        return list(self._robots)

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return list(self._robots.get(robot_name, []))

    def add_object(self, name, **kw):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def remove_object(self, name):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images=False):  # type: ignore[no-untyped-def]
        return {n: 0.0 for n in self._robots["arm"]}

    def send_action(self, action, robot_name=None, n_substeps=1):  # type: ignore[no-untyped-def]
        self.sends += 1

    def render(self, camera_name="default", width=None, height=None):  # type: ignore[no-untyped-def]
        return {"image": np.zeros((height or 48, width or 64, 3), dtype=np.uint8)}

    def _maybe_install_wbc_torque_control(self, policy: Any, robot_name: str) -> None:
        self.shim_installs += 1
        return None


@pytest.fixture
def pacers(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the period of every pacer the runner acquires, and acquire none.

    ``PolicyRunner.run`` imports ``Ticker`` from the pacing module inside the
    call, so replacing the attribute on that module is what the runner sees.
    """
    acquired: list[float] = []

    class _Ticker:
        def __init__(self, period: float) -> None:
            acquired.append(period)

        def __enter__(self) -> _Ticker:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def wait(self) -> None:
            return None

        def tick(self) -> None:
            return None

    monkeypatch.setattr(pacing_mod, "Ticker", _Ticker)
    return acquired


def _run(sim: _Sim, **kwargs: Any) -> dict[str, Any]:
    """Drive the real facade over the mock policy, three steps at 50 Hz.

    The values under test are deliberately outside the declared ``bool``; one
    funnel states that once instead of suppressing it per call.
    """
    kwargs.setdefault("fast_mode", True)
    return sim.run_policy("arm", policy_provider="mock", n_steps=3, control_frequency=50.0, **kwargs)


def _text(result: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in result.get("content") or [] if isinstance(block, dict))


def _json(result: dict[str, Any]) -> dict[str, Any]:
    return next(block["json"] for block in result["content"] if "json" in block)


class TestAnUndeclaredSpellingIsRefusedNamingTheFlag:
    """Neither half is read as a posture, and the refusal is the shared domain's."""

    @pytest.mark.parametrize("flag", RUN_POLICY_FLAGS)
    @pytest.mark.parametrize("value", UNDECLARED, ids=repr)
    def test_run_policy_refuses_it(self, pacers: list[float], flag: str, value: Any) -> None:
        sim = _Sim()
        result = _run(sim, **{flag: value})
        assert result["status"] == "error"
        assert _text(result) == boolean_flag_error(value, flag, "run_policy")

    @pytest.mark.parametrize("flag", RUN_POLICY_FLAGS)
    @pytest.mark.parametrize("value", UNDECLARED, ids=repr)
    def test_the_refusal_precedes_every_side_effect(self, pacers: list[float], flag: str, value: Any) -> None:
        """No pacer, no shim, no reset, no action: nothing the flag selects ran."""
        sim = _Sim()
        _run(sim, n_episodes=2, **{flag: value})
        assert (pacers, sim.shim_installs, sim.resets, sim.sends) == ([], 0, 0, 0)

    @pytest.mark.parametrize("value", (*UNDECLARED, None), ids=repr)
    def test_eval_policy_refuses_a_non_boolean_async_rtc(self, value: Any) -> None:
        """``eval_policy`` declares ``async_rtc: bool``, so ``None`` is refused there too."""
        sim = _Sim()
        result = sim.eval_policy("arm", policy_provider="mock", n_episodes=1, max_steps=3, async_rtc=value)
        assert result["status"] == "error"
        assert _text(result) == boolean_flag_error(value, "async_rtc", "eval_policy")
        assert sim.sends == 0

    def test_two_mistyped_flags_report_the_first_in_signature_order(self, pacers: list[float]) -> None:
        result = _run(_Sim(), fast_mode="false", reset_between="false")
        assert _text(result) == boolean_flag_error("false", "fast_mode", "run_policy")


class TestWhyEachHalfIsRefused:
    """The branch each undeclared spelling selected, measured on the fixed tree.

    These cells drive the branch through a declared spelling so the behaviour
    the flag selects is pinned beside the refusal - a future reader can see what
    ``"false"`` would have done, not only that it is refused.
    """

    def test_fast_mode_false_acquires_one_pacer_at_the_control_period(self, pacers: list[float]) -> None:
        result = _run(_Sim(), fast_mode=False)
        assert result["status"] == "success", _text(result)
        assert pacers == [pytest.approx(1.0 / 50.0)]

    def test_fast_mode_true_acquires_no_pacer(self, pacers: list[float]) -> None:
        result = _run(_Sim(), fast_mode=True)
        assert result["status"] == "success", _text(result)
        assert pacers == []

    def test_reset_between_false_carries_the_scene_across_episodes(self, pacers: list[float]) -> None:
        sim = _Sim()
        result = _run(sim, n_episodes=2, reset_between=False)
        assert result["status"] == "success", _text(result)
        assert sim.resets == 0

    def test_reset_between_true_resets_once_between_two_episodes(self, pacers: list[float]) -> None:
        sim = _Sim()
        result = _run(sim, n_episodes=2, reset_between=True)
        assert result["status"] == "success", _text(result)
        assert sim.resets == 1

    def test_wbc_install_torque_control_false_installs_no_shim(self, pacers: list[float]) -> None:
        sim = _Sim()
        result = _run(sim, wbc_install_torque_control=False)
        assert result["status"] == "success", _text(result)
        assert sim.shim_installs == 0

    def test_wbc_install_torque_control_true_installs_the_shim(self, pacers: list[float]) -> None:
        sim = _Sim()
        result = _run(sim, wbc_install_torque_control=True)
        assert result["status"] == "success", _text(result)
        assert sim.shim_installs == 1

    @pytest.mark.parametrize(("value", "enabled"), [(True, True), (False, False), (None, False)], ids=repr)
    def test_async_rtc_declared_spellings_select_the_pipeline_they_name(
        self, pacers: list[float], value: Any, enabled: bool
    ) -> None:
        """``None`` resolves from the mock policy, which is single-step, so synchronous."""
        result = _run(_Sim(), async_rtc=value)
        assert result["status"] == "success", _text(result)
        assert _json(result)["rtc_async_enabled"] is enabled


class TestTheDeclaredSpellingsStillRun:
    """Over-reach control: a boolean of either type is honoured, whatever the flag."""

    @pytest.mark.parametrize("flag", RUN_POLICY_FLAGS)
    @pytest.mark.parametrize("value", (True, False, np.True_, np.False_), ids=repr)
    def test_a_python_or_numpy_boolean_is_accepted(self, pacers: list[float], flag: str, value: Any) -> None:
        sim = _Sim()
        result = _run(sim, **{flag: value})
        assert result["status"] == "success", _text(result)
        assert sim.sends == 3

    def test_omitting_every_flag_keeps_the_defaults(self, pacers: list[float]) -> None:
        sim = _Sim()
        result = sim.run_policy("arm", policy_provider="mock", n_steps=3, control_frequency=50.0)
        assert result["status"] == "success", _text(result)
        # The defaults: paced, reset between episodes, shim installed, RTC from the policy.
        assert pacers == [pytest.approx(1.0 / 50.0)]
        assert sim.shim_installs == 1
        assert _json(result)["rtc_async_enabled"] is False


class TestTheRosterIsTheSignature:
    """A posture flag added to ``run_policy`` cannot skip the domain unnoticed."""

    def test_every_boolean_parameter_of_run_policy_is_in_the_roster(self) -> None:
        declared = {
            name
            for name, param in inspect.signature(SimEngine.run_policy).parameters.items()
            if param.annotation in ("bool", "bool | None", bool, bool | None)
        }
        assert declared, "no boolean parameter found - the roster read is broken"
        assert declared == set(RUN_POLICY_FLAGS), declared

    def test_the_helper_is_the_shared_domain_verbatim(self) -> None:
        for value in (*UNDECLARED, None, True, False, np.True_):
            err = SimEngine._validate_posture_flags("run_policy", fast_mode=value)
            expected = boolean_flag_error(value, "fast_mode", "run_policy")
            assert (None if err is None else _text(err)) == expected


class TestTheToolRefusesBeforeItStartsTheRecording:
    """The tool starts a recording with ``overwrite=True`` before the episode loop.

    The facade's refusal would therefore arrive after the dataset at
    ``dataset_root`` had been replaced with an empty one - the same reason the
    tool already checks the seed, the rates and the keyword bags up front.
    """

    @pytest.mark.parametrize("value", (*UNDECLARED, None), ids=repr)
    def test_a_non_boolean_fast_mode_is_refused_with_the_facades_words(self, value: Any, tmp_path: Path) -> None:
        from tests.tools.test_run_policy import _FakeSim

        sim = _FakeSim()
        result = dict(
            rp_mod.run_policy(sim, n_episodes=1, n_steps=4, fast_mode=value, dataset_root=str(tmp_path / "ds"))
        )
        assert result["status"] == "error"
        assert _text(result) == boolean_flag_error(value, "fast_mode", "run_policy")
        assert sim.start_recording_calls == [], "the refused call reached start_recording(overwrite=True)"
        assert sim.run_policy_calls == []

    @pytest.mark.parametrize("value", (True, False, np.False_), ids=repr)
    def test_a_boolean_fast_mode_is_forwarded_verbatim(self, value: Any) -> None:
        from tests.tools.test_run_policy import _FakeSim

        sim = _FakeSim()
        result = dict(rp_mod.run_policy(sim, n_episodes=1, n_steps=4, fast_mode=value))
        assert result["status"] == "success", _text(result)
        assert [call["fast_mode"] for call in sim.run_policy_calls] == [value]

    def test_every_boolean_parameter_of_the_tool_is_checked(self) -> None:
        flags = {
            name
            for name, param in inspect.signature(rp_mod.run_policy).parameters.items()
            if param.annotation in (bool, "bool")
        }
        assert flags == {"fast_mode"}, flags


_ARM_XML = """<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <geom type="box" size="0.05 0.05 0.1"/>
      <body name="link1" pos="0 0 0.1">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" limited="true" damping="2"/>
        <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.03"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a_shoulder" joint="shoulder" kp="30" ctrlrange="-2 2"/>
  </actuator>
</mujoco>
"""


class TestStartPolicyRefusesBeforeTheSubmit:
    """MuJoCo's ``start_policy`` runs the rollout on a worker.

    A refusal raised there is discarded with the future and the caller reads
    "started", which is why its sibling knobs are checked synchronously; the
    pacing posture is now one of them.
    """

    @pytest.fixture
    def sim(self, tmp_path: Path) -> Any:
        pytest.importorskip("mujoco")
        from strands_robots import Simulation

        arm_xml = tmp_path / "arm.xml"
        arm_xml.write_text(_ARM_XML, encoding="utf-8")
        sim = Simulation(backend="mujoco", tool_name="posture_flag_test", mesh=False)
        sim.create_world()
        sim.add_robot(name="arm", urdf_path=str(arm_xml))
        yield sim
        sim.cleanup()

    @pytest.mark.parametrize("value", (*UNDECLARED, None), ids=repr)
    def test_a_non_boolean_fast_mode_is_refused_without_reporting_started(self, sim: Any, value: Any) -> None:
        result = sim.start_policy(
            robot_name="arm", policy_provider="mock", n_steps=4, control_frequency=30.0, fast_mode=value
        )
        assert result["status"] == "error"
        assert _text(result) == boolean_flag_error(value, "fast_mode", "start_policy")
        assert "started" not in _text(result).lower()
        assert "No policies running" in _text(sim.list_policies_running()), "the refused call claimed the robot"

    def test_a_boolean_fast_mode_is_still_submitted(self, sim: Any) -> None:
        result = sim.start_policy(
            robot_name="arm", policy_provider="mock", n_steps=4, control_frequency=30.0, fast_mode=True
        )
        assert result["status"] == "success", _text(result)
        sim.stop_policy("arm")
