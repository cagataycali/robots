"""Five places on the mjlab paths that answered a default instead of refusing.

Review of the backend PR found each one, all of the same class (a wrong or
missing input silently becomes a value the robot then follows):

* ``add_robot`` did not consult the frozen-schema guard the ABC requires on
  every backend, so a robot added mid-recording recorded zero rows.
* ``send_action`` read ``nan`` as its own "key absent" sentinel, so a diverged
  policy's ``nan`` kept the previous ctrl and the call reported success;
  ``send_action_batch`` wrote a non-finite block straight into ctrl.
* ``_resolve_key`` answered ``None`` (the zero pose) for an unknown keyframe,
  an index out of range, or any keyframe on a model without one.
* ``_SiteFK`` indexed ``jnt_qposadr`` with the ``-1`` ``mj_name2id`` returns for
  a joint the MJCF lacks, so it read the last joint's slot.
* ``vec_rollout`` zero-filled every action key the policy did not emit.

These tests run without mjlab (the pure helpers and the shared coercion are
exercised through a stand-in engine); the CUDA parity suite covers the rest.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.mjlab import simulation as mjlab_sim  # noqa: E402

_MJCF_WITH_KEYS = """
<mujoco>
  <worldbody>
    <body name="arm"><joint name="j0" type="hinge"/><geom size="0.05"/>
      <body name="tip"><joint name="j1" type="hinge"/><geom size="0.04"/><site name="ee"/></body>
    </body>
  </worldbody>
  <actuator><position name="j0" joint="j0"/><position name="j1" joint="j1"/></actuator>
  <keyframe><key name="home" qpos="0.1 0.2"/><key name="rest" qpos="0 0"/></keyframe>
</mujoco>
"""

_MJCF_NO_KEYS = """
<mujoco>
  <worldbody><body name="arm"><joint name="j0" type="hinge"/><geom size="0.05"/></body></worldbody>
  <actuator><position name="j0" joint="j0"/></actuator>
</mujoco>
"""


class TestAnUnknownKeyframeIsRefusedByName:
    def test_none_is_the_zero_pose_even_with_keyframes(self) -> None:
        model = mujoco.MjModel.from_xml_string(_MJCF_WITH_KEYS)
        assert mjlab_sim._resolve_key(model, None) is None

    def test_a_known_name_and_index_resolve(self) -> None:
        model = mujoco.MjModel.from_xml_string(_MJCF_WITH_KEYS)
        assert mjlab_sim._resolve_key(model, "rest") == 1
        assert mjlab_sim._resolve_key(model, 0) == 0

    def test_an_unknown_name_names_the_available_keyframes(self) -> None:
        model = mujoco.MjModel.from_xml_string(_MJCF_WITH_KEYS)
        with pytest.raises(KeyError, match=r"'hoem' not found\. Available: 'home', 'rest'"):
            mjlab_sim._resolve_key(model, "hoem")

    def test_an_index_out_of_range_is_refused(self) -> None:
        model = mujoco.MjModel.from_xml_string(_MJCF_WITH_KEYS)
        with pytest.raises(KeyError, match="index 7 out of range; the model has 2 keyframe"):
            mjlab_sim._resolve_key(model, 7)

    def test_any_keyframe_on_a_model_without_one_is_refused(self) -> None:
        model = mujoco.MjModel.from_xml_string(_MJCF_NO_KEYS)
        with pytest.raises(KeyError, match="Available: none"):
            mjlab_sim._resolve_key(model, "home")

    def test_inspect_mjcf_reads_the_requested_keyframe(self, tmp_path: Path) -> None:
        path = tmp_path / "arm.xml"
        path.write_text(_MJCF_WITH_KEYS)
        home, free, actuators, joints = mjlab_sim._inspect_mjcf(str(path), "home")
        assert home == {"j0": pytest.approx(0.1), "j1": pytest.approx(0.2)}
        assert not free and actuators == ["j0", "j1"] and joints == ["j0", "j1"]
        with pytest.raises(KeyError, match="'hoem' not found"):
            mjlab_sim._inspect_mjcf(str(path), "hoem")


class _StandInEngine(mjlab_sim.MjlabEngine):
    """The engine's ``send_action`` over a fake sim; no mjlab, no CUDA."""

    def __init__(self, actuator_names: list[str]) -> None:  # noqa: D107 - stand-in
        import threading

        self._lock = threading.RLock()
        self._robots = {
            "arm": mjlab_sim._RobotSpec(
                name="arm", path="", position=(0, 0, 0), orientation=(1, 0, 0, 0), keyframe=None
            )
        }
        self._robots["arm"].actuator_names = list(actuator_names)
        self._objects: dict[str, Any] = {}
        self.num_envs = 1
        self.device = "cpu"
        self._timestep = 0.002
        self._default_timestep = 0.002
        self._step_count = 0
        self._recording_manager = None
        self.written: list[Any] = []

    def _ensure_built(self) -> None:
        import torch

        if getattr(self, "_sim", None) is not None:
            return

        class _Data:
            ctrl = torch.zeros((1, 2))

        class _Sim:
            data = _Data()

            def step(self) -> None:
                pass

        class _Scene:
            def update(self, dt: float) -> None:
                pass

        self._sim = _Sim()
        self._scene = _Scene()

    def _actuator_ids(self, robot_name: str) -> list[int]:
        return [0, 1]


@pytest.fixture
def engine() -> _StandInEngine:
    return _StandInEngine(["j0", "j1"])


class TestSendActionRefusesWhatEveryBackendRefuses:
    def test_a_nan_in_a_dict_is_an_error_not_the_previous_ctrl(self, engine: _StandInEngine) -> None:
        res = engine.send_action({"j0": float("nan"), "j1": 0.1}, "arm")
        assert res["status"] == "error"
        assert "j0" in res["content"][0]["text"]

    def test_a_nan_in_a_full_vector_is_an_error(self, engine: _StandInEngine) -> None:
        assert engine.send_action([0.1, float("nan")], "arm")["status"] == "error"

    def test_a_boolean_is_an_error(self, engine: _StandInEngine) -> None:
        assert engine.send_action({"j0": True, "j1": 0.1}, "arm")["status"] == "error"

    def test_a_non_numeric_value_is_an_error_not_a_traceback(self, engine: _StandInEngine) -> None:
        bad: dict[str, Any] = {"j0": "open", "j1": 0.1}
        assert engine.send_action(bad, "arm")["status"] == "error"

    def test_a_partial_dict_still_keeps_the_other_ctrl(self, engine: _StandInEngine) -> None:
        engine._ensure_built()
        engine._sim.data.ctrl[0, 1] = 0.7
        res = engine.send_action({"j0": 0.3}, "arm")
        assert res["status"] == "success", res
        assert engine._sim.data.ctrl.tolist() == [[pytest.approx(0.3), pytest.approx(0.7)]]

    def test_a_batch_with_nan_is_refused(self, engine: _StandInEngine) -> None:
        res = engine.send_action_batch(np.array([[0.1, float("nan")]], dtype=np.float32), "arm")
        assert res["status"] == "error"
        assert "nan/inf" in res["content"][0]["text"]

    @pytest.mark.parametrize("bad", [0, -1, 2.7, True, "2"])
    def test_a_batch_with_a_non_positive_or_fractional_substep_count_is_refused_before_any_write(
        self, engine: _StandInEngine, bad: Any
    ) -> None:
        """Same floor as ``send_action`` (review on #4229): a success that advanced 0 steps leaves a
        target the world never integrates, and the vec_eval recorder would keep the row. ``2.7`` used
        to be floored to 2 silently; ``round(1 / (dt * control_hz))`` yields 0 above 250 Hz."""
        engine._ensure_built()
        before = engine._sim.data.ctrl.clone()
        res = engine.send_action_batch(np.array([[0.1, 0.2]], dtype=np.float32), "arm", n_substeps=bad)
        assert res["status"] == "error", res
        assert "n_substeps" in res["content"][0]["text"] and "send_action_batch" in res["content"][0]["text"]
        assert engine._sim.data.ctrl.tolist() == before.tolist(), "ctrl was written before the refusal"
        assert engine._step_count == 0

    def test_the_substep_guard_runs_before_the_shape_check(self, engine: _StandInEngine) -> None:
        """A wrong shape AND a bad count: the count is named (nothing was converted or compared)."""
        res = engine.send_action_batch(np.zeros((3, 7), dtype=np.float32), "arm", n_substeps=0)
        assert res["status"] == "error"
        assert "n_substeps" in res["content"][0]["text"]

    def test_a_batch_with_a_positive_count_still_advances(self, engine: _StandInEngine) -> None:
        res = engine.send_action_batch(np.array([[0.1, 0.2]], dtype=np.float32), "arm", n_substeps=3)
        assert res["status"] == "success", res
        assert engine._step_count == 3


def test_add_robot_consults_the_frozen_schema_guard_first(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _StandInEngine(["j0", "j1"])
    refusal = {"status": "error", "content": [{"text": "recording schema is frozen"}]}
    monkeypatch.setattr(engine, "_recording_schema_frozen_error", lambda verb, name: refusal)
    assert engine.add_robot("so101") is refusal


def test_site_fk_refuses_a_joint_the_mjcf_lacks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from strands_robots.policies.rsl_rl_onnx import policy as rsl

    path = tmp_path / "arm.xml"
    path.write_text(_MJCF_WITH_KEYS)
    monkeypatch.setattr("strands_robots.assets.resolve_model_path", lambda name: path)
    monkeypatch.setattr("strands_robots.assets.resolve_robot_name", lambda name: name)
    with pytest.raises(
        ValueError,
        match=r"joints \['shoulder'\] from the ONNX metadata are not in arm's MJCF; its joints are \['j0', 'j1'\]",
    ):
        rsl._SiteFK("arm", "ee", ["j0", "shoulder"])
    fk = rsl._SiteFK("arm", "ee", ["j1", "j0"])
    assert fk.qadr == [1, 0]


class _FakeVecEngine:
    num_envs = 2

    def __init__(self) -> None:
        self.blocks: list[np.ndarray] = []

    def reset(self) -> dict[str, Any]:
        return {"status": "success"}

    def robot_action_keys(self, robot_name: str) -> list[str]:
        return ["j0", "j1"]

    def physics_timestep(self) -> float:
        return 0.002

    def get_observation_batch(self, robot_name: str) -> dict[str, Any]:
        return {"joint_pos": np.zeros((2, 2), dtype=np.float32)}

    def send_action_batch(self, block: np.ndarray, robot_name: str, n_substeps: int = 1) -> dict[str, Any]:
        self.blocks.append(np.asarray(block))
        return {"status": "success"}


class _KeyPolicy:
    def __init__(self, keys: list[str]) -> None:
        self.keys = keys

    async def get_actions(self, observation: dict[str, Any], instruction: str = "", **kwargs: Any) -> list[dict]:
        return [{k: 0.5 for k in self.keys}]


def test_vec_rollout_refuses_a_policy_that_omits_an_action_key() -> None:
    from strands_robots.training.mjlab_tasks.vec_eval import vec_rollout

    engine = _FakeVecEngine()
    with pytest.raises(ValueError, match=r"lack keys \['j1'\] that 'arm' commands"):
        asyncio.run(vec_rollout(engine, _KeyPolicy(["j0"]), robot_name="arm", ticks=1))
    assert engine.blocks == [], "nothing was commanded before the refusal"
    result = asyncio.run(vec_rollout(engine, _KeyPolicy(["j0", "j1"]), robot_name="arm", ticks=1))
    assert result.ticks == 1 and engine.blocks[0].tolist() == [[0.5, 0.5], [0.5, 0.5]]
