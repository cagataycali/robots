"""A step that makes MuJoCo declare the physics unstable is reported, not a success.

``mj_step`` checks ``qpos``/``qvel``/``qacc`` before it integrates. When one is
NaN, inf or past ``mjMAXVAL``, MuJoCo resets the WHOLE state - every robot joint
and every object back to the model's initial pose, and ``data.time`` back to 0 -
and says so only as a ``WARNING`` on stderr. The input side is already guarded
(``test_a_qpos_write_beyond_mujocos_ceiling_is_refused``); the dynamics side was
not. Measured on ``Robot("so100")`` with a cube: after ``step(100)`` (t=0.200s),
``apply_force(cube, [1e12, 0, 0])`` then ``step(50)`` answered
``success  "+50 steps | t=0.1000s"`` - the clock had run BACKWARDS - and the
elbow the caller had set to 1.0 rad was back at its initial pose. A timestep of
``1e308`` (accepted with a warning) left ``qpos`` NaN under the same ``success``.

Pinned here: ``step``, ``send_action``, the three motion primitives and
``run_multi_policy`` each return ``status="error"`` naming the divergence, the
joint MuJoCo flagged and ``reset`` as the recovery; ``reset`` recovers; and an
ordinary long run is not flagged.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.mock import MockPolicy  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

from .test_motion_primitives import ARM_XML, REACHABLE  # noqa: E402

_SHOVE = [1e12, 0.0, 0.0]  # finite, and enough to blow the cube's qacc past mjMAXVAL


def _text(result: dict) -> str:
    return " ".join(str(block.get("text", "")) for block in result.get("content", []))


def _ok(result: dict, what: str) -> dict:
    if result["status"] != "success":
        raise AssertionError(f"{what} refused: {_text(result)}")
    return result


@pytest.fixture
def sim(tmp_path):
    path = tmp_path / "arm.xml"
    path.write_text(ARM_XML)
    s = Simulation(tool_name="test_diverged_world", mesh=False)
    _ok(s.create_world(), "create_world")
    _ok(s.add_robot("arm", urdf_path=str(path)), "add_robot")
    _ok(s.add_object("cube", shape="box", size=[0.02] * 3, position=[0.3, 0.0, 0.05]), "add_object")
    _ok(s.step(100), "step")
    yield s
    s.cleanup(policy_stop_timeout=2.0)


def _shove(sim) -> None:
    _ok(sim.apply_force("cube", force=_SHOVE), "apply_force")


def _assert_reports_divergence(result: dict, verb: str) -> None:
    text = _text(result)
    assert result["status"] == "error", f"{verb} answered success on a world MuJoCo reset: {text}"
    assert text.startswith(f"{verb}: the physics diverged"), text
    assert "cube" in text, f"the report should name the joint MuJoCo flagged: {text}"
    assert "reset" in text, f"the report should name the recovery: {text}"


class TestStep:
    def test_a_step_through_a_divergence_is_an_error(self, sim):
        t_before = sim._world._data.time
        _shove(sim)

        result = sim.step(50)

        _assert_reports_divergence(result, "step")
        assert "initial pose" in _text(result)
        assert sim._world._data.time < t_before, "premise: MuJoCo rewound the clock"

    def test_a_non_finite_state_is_an_error(self, sim):
        sim._world._model.opt.timestep = 1e308  # what set_timestep accepts with a warning

        result = sim.step(5)

        assert result["status"] == "error", _text(result)
        assert "physics diverged" in _text(result)

    def test_reset_recovers_and_the_next_step_succeeds(self, sim):
        _shove(sim)
        assert sim.step(50)["status"] == "error"

        _ok(sim.reset(), "reset")

        _ok(sim.step(50), "step after reset")

    def test_an_ordinary_long_run_is_not_flagged(self, sim):
        _ok(sim.step(5000), "step(5000)")


class TestSendAction:
    def test_send_action_through_a_divergence_is_an_error(self, sim):
        _shove(sim)

        result = sim.send_action({}, robot_name="arm", n_substeps=5)

        _assert_reports_divergence(result, "send_action")


class TestPrimitives:
    def test_move_to(self, sim):
        pytest.importorskip("mink")
        _shove(sim)
        _assert_reports_divergence(sim.move_to(robot_name="arm", position=REACHABLE, tol=0.002), "move_to")

    def test_set_gripper(self, sim):
        _shove(sim)
        _assert_reports_divergence(sim.set_gripper(robot_name="arm", state="close"), "set_gripper")

    def test_rotate_wrist(self, sim):
        _shove(sim)
        _assert_reports_divergence(sim.rotate_wrist(robot_name="arm", target_yaw=1.2), "rotate_wrist")


class TestRunMultiPolicy:
    def test_the_synchronized_loop_stops_and_reports(self, sim):
        _shove(sim)

        result = sim.run_multi_policy(
            policies={"arm": MockPolicy()}, instructions="hold", n_steps=50, control_frequency=500.0
        )

        _assert_reports_divergence(result, "run_multi_policy")
        assert "Stopped after 0 synchronized steps" in _text(result)


class _ShoveOnThirdQuery(MockPolicy):
    """A mock arm policy that shoves the cube mid-episode, from inside the rollout."""

    def __init__(self, sim, **kwargs):
        super().__init__(**kwargs)
        self._sim = sim
        self._queries = 0

    async def get_actions(self, *args, **kwargs):
        self._queries += 1
        if self._queries == 3:
            self._sim.apply_force("cube", force=_SHOVE)
        return await super().get_actions(*args, **kwargs)


class TestRollouts:
    def test_run_policy_stops_on_the_divergence(self, sim):
        """Before: ``success`` with "1/25 action steps reported coarse backend errors"."""
        _shove(sim)

        result = sim.run_policy(robot_name="arm", policy_provider="mock", instruction="wave", duration=0.2)

        assert result["status"] == "error", _text(result)
        assert "the physics diverged" in _text(result)
        assert "The rollout stopped at step 0" in _text(result)

    @pytest.mark.parametrize("async_rtc", [False, True])
    def test_eval_policy_does_not_score_a_reset_world(self, sim, async_rtc):
        """Before: ``Success: 2/2 (100.0%)`` over episodes MuJoCo had reset mid-run."""
        result = sim.eval_policy(
            robot_name="arm",
            policy_object=_ShoveOnThirdQuery(sim),
            n_episodes=2,
            max_steps=40,
            action_horizon=1,
            success_fn="contact",
            async_rtc=async_rtc,
        )

        assert result["status"] == "error", _text(result)
        assert "Stopped at episode 0, step 2" in _text(result)
        payload = result["content"][-1]["json"]
        assert payload["episodes_completed"] == 0
        assert "the physics diverged" in payload["physics_error"]
