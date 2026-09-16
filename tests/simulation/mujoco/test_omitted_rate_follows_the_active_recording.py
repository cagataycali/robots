"""Through the tool router, an OMITTED rate follows the one already in force.

``start_recording`` defaults to 30 fps; every rollout defaults to
``control_frequency=50.0``. The recorder writes one frame per control step
with no decimation, so the two defaults cannot both be honored and the rate
guard refuses the rollout. Measured on ``Robot("so101", mode="sim")``: the
documented record-then-rollout sequence - ``start_recording(task=...)`` then
``run_policy(instruction=...)``, no rate named anywhere - was refused on the
first try, and the remedy cost the agent a second round-trip to type a number
the tool already knew.

A caller who omitted the rate expressed no preference between the defaults,
so the omitted one follows the recording (rollout after recording) or the
rollout (recording after rollout), and the result says so. A caller who
PASSED a rate is still refused on a mismatch. The rule belongs to the engine
rather than to the tool router, so a direct ``run_policy(...)`` on the Python
API follows the open recording exactly as a routed call does - an agent and a
script cannot disagree about what an absent rate means.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("lerobot")

from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine  # noqa: E402

_ARM = """<mujoco><worldbody><body name="l1">
<joint name="j1" type="hinge" axis="0 0 1" range="-1.5 1.5" damping="4"/>
<geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.02"/>
<body name="l2" pos="0.15 0 0">
<joint name="j2" type="hinge" axis="0 0 1" range="-1.5 1.5" damping="4"/>
<geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.02"/></body></body></worldbody>
<actuator><position name="a1" joint="j1" kp="30" ctrlrange="-1.5 1.5"/>
<position name="a2" joint="j2" kp="30" ctrlrange="-1.5 1.5"/></actuator></mujoco>"""


def _text(result) -> str:
    return " ".join(c["text"] for c in result["content"] if "text" in c)


@pytest.fixture
def sim(tmp_path):
    xml = tmp_path / "arm.xml"
    xml.write_text(_ARM)
    engine = MuJoCoSimEngine(tool_name="omitted_rate", mesh=False)
    engine.create_world()
    engine.add_robot(name="arm", urdf_path=str(xml))
    yield engine
    engine.cleanup(policy_stop_timeout=2.0)


def _wait_idle(sim, name: str, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    fut = sim._policy_threads.get(name)
    while fut is not None and not fut.done() and time.time() < deadline:
        time.sleep(0.02)


class TestARolloutAfterARecording:
    def test_the_documented_sequence_succeeds_first_try(self, sim, tmp_path):
        rec = sim._dispatch_action(
            "start_recording", {"repo_id": "local/a", "task": "hold", "root": str(tmp_path / "a")}
        )
        assert rec["status"] == "success"
        result = sim._dispatch_action("run_policy", {"robot_name": "arm", "policy_provider": "mock", "n_steps": 6})
        assert result["status"] == "success", result
        assert (
            "control_frequency=30 followed the active recording's 30 fps (no rate was passed); "
            "pass control_frequency= to choose." in _text(result)
        )
        stopped = sim._dispatch_action("stop_recording", {})
        assert stopped["status"] == "success"
        assert "6 frames" in _text(stopped)

    def test_the_recorded_timebase_is_the_capture_rate(self, sim, tmp_path):
        sim._dispatch_action(
            "start_recording", {"repo_id": "local/b", "task": "hold", "root": str(tmp_path / "b"), "fps": 25}
        )
        result = sim._dispatch_action("run_policy", {"robot_name": "arm", "policy_provider": "mock", "n_steps": 5})
        assert result["status"] == "success"
        assert "control_frequency=25 followed" in _text(result)
        assert sim.get_recording_status()["content"][0]["text"].count("5 steps") == 1

    def test_a_passed_rate_is_still_refused_on_a_mismatch(self, sim, tmp_path):
        sim._dispatch_action("start_recording", {"repo_id": "local/c", "task": "hold", "root": str(tmp_path / "c")})
        result = sim._dispatch_action(
            "run_policy", {"robot_name": "arm", "policy_provider": "mock", "n_steps": 5, "control_frequency": 50}
        )
        assert result["status"] == "error"
        assert "the active recording declares 30 fps but this rollout captures at control_frequency=50 Hz" in _text(
            result
        )

    def test_start_policy_follows_too(self, sim, tmp_path):
        sim._dispatch_action("start_recording", {"repo_id": "local/d", "task": "hold", "root": str(tmp_path / "d")})
        result = sim._dispatch_action("start_policy", {"robot_name": "arm", "policy_provider": "mock", "duration": 0.2})
        assert result["status"] == "success", result
        assert "control_frequency=30 followed" in _text(result)
        _wait_idle(sim, "arm")

    def test_no_recording_means_nothing_is_added_or_said(self, sim):
        result = sim._dispatch_action("run_policy", {"robot_name": "arm", "policy_provider": "mock", "n_steps": 3})
        assert result["status"] == "success"
        assert "followed" not in _text(result)

    def test_a_direct_python_call_follows_too(self, sim, tmp_path):
        """The engine owns the rule, so a script sees what an agent sees.

        The deferral lives in ``run_policy`` itself rather than in the tool
        router's reading of an absent field, so calling the Python API without
        a rate is not refused either, and the rollout is captured into the open
        recording at the rate that recording declared. A caller refused here
        would have to type a number the engine already knew. Naming the adopted
        rate back to the caller is the routed surface's line, pinned above.
        """
        sim.start_recording(repo_id="local/e", task="hold", root=str(tmp_path / "e"))
        result = sim.run_policy(robot_name="arm", policy_provider="mock", n_steps=3)
        assert result["status"] == "success", result
        assert "3 steps" in _text(sim.get_recording_status())


class TestARecordingAfterARollout:
    def test_an_omitted_fps_follows_the_running_rollout(self, sim, tmp_path):
        started = sim._dispatch_action(
            "start_policy", {"robot_name": "arm", "policy_provider": "mock", "duration": 3.0, "control_frequency": 40}
        )
        assert started["status"] == "success", started
        time.sleep(0.05)
        rec = sim._dispatch_action(
            "start_recording", {"repo_id": "local/f", "task": "hold", "root": str(tmp_path / "f")}
        )
        assert rec["status"] == "success", rec
        text = _text(rec)
        assert "@ 40fps" in text
        assert (
            "fps=40 followed the running rollout's control_frequency=40 (no fps was passed); pass fps= to choose."
            in text
        )
        sim.stop_policy("arm")
        _wait_idle(sim, "arm")

    def test_a_passed_fps_is_still_refused(self, sim, tmp_path):
        assert (
            sim._dispatch_action("start_policy", {"robot_name": "arm", "policy_provider": "mock", "duration": 3.0})[
                "status"
            ]
            == "success"
        )
        time.sleep(0.05)
        rec = sim._dispatch_action(
            "start_recording", {"repo_id": "local/g", "task": "hold", "root": str(tmp_path / "g"), "fps": 30}
        )
        assert rec["status"] == "error"
        sim.stop_policy("arm")
        _wait_idle(sim, "arm")

    def test_a_fractional_rollout_rate_is_not_followed(self, sim, tmp_path):
        assert (
            sim._dispatch_action(
                "start_policy",
                {"robot_name": "arm", "policy_provider": "mock", "duration": 3.0, "control_frequency": 12.5},
            )["status"]
            == "success"
        )
        time.sleep(0.05)
        rec = sim._dispatch_action(
            "start_recording", {"repo_id": "local/h", "task": "hold", "root": str(tmp_path / "h")}
        )
        assert rec["status"] == "error"
        assert "followed" not in _text(rec)
        # The refusal describes the caller's own default, not a rate the router
        # invented: truncating 12.5 to a whole 12 and filling it in would blame
        # them for a number they never typed ("would declare 12 fps").
        assert "this recording would declare 30 fps" in _text(rec)
        sim.stop_policy("arm")
        _wait_idle(sim, "arm")

    def test_two_rollouts_at_different_rates_are_not_followed(self, sim, tmp_path):
        """No single rate can describe two, so nothing is filled in."""
        sim.add_robot(name="arm2", urdf_path=str(tmp_path / "arm.xml"))
        for name, rate in (("arm", 40), ("arm2", 25)):
            started = sim._dispatch_action(
                "start_policy",
                {"robot_name": name, "policy_provider": "mock", "duration": 3.0, "control_frequency": rate},
            )
            assert started["status"] == "success", started
        time.sleep(0.05)
        assert set(sim._active_rollout_rates().values()) == {40.0, 25.0}
        rec = sim._dispatch_action(
            "start_recording", {"repo_id": "local/i", "task": "hold", "root": str(tmp_path / "i")}
        )
        assert rec["status"] == "error"
        assert "followed" not in _text(rec)
        # Neither rate is adopted: the refusal still reports the untouched
        # default and names both rollouts.
        text = _text(rec)
        assert "this recording would declare 30 fps" in text
        assert "'arm' at 40 Hz" in text and "'arm2' at 25 Hz" in text
        for name in ("arm", "arm2"):
            sim.stop_policy(name)
            _wait_idle(sim, name)
