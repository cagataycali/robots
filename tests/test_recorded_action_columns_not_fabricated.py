"""A recorded action column holds a command that was issued, or the frame is refused.

``DatasetRecorder.add_frame`` writes one value per declared action column. When
the frame's action dict does not carry a declared column, there is no command to
record, and every candidate placeholder misrepresents what happened:

* ``0.0`` is itself a command on an absolute-position action space, so
  :meth:`replay_episode` drives that joint to zero at servo speed.
* A joint's measured position is in different units from a normalized or
  tendon-driven actuator's command.
* The command standing on the actuator cannot be read back - the
  action-to-``ctrl`` map is deliberately not injective.

So the frame is refused. These tests pin the refusal, the scoping that keeps a
shared-scene recording working, and that no episode survives a refused rollout
for :meth:`replay_episode` to re-issue.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

from strands_robots.dataset_recorder import DatasetRecorder, unrecordable_action_columns_error
from strands_robots.policies import Policy

from .test_dataset_recorder import _CapturingDataset, _state_action_features

DECLARED = ["a_shoulder", "a_elbow", "a_grip"]


class TestUnrecordableActionColumnsError:
    """The rule itself, independent of any dataset."""

    def test_no_required_columns_declared_skips_the_check(self):
        """``None`` is the historical contract: the caller makes no claim."""
        assert unrecordable_action_columns_error({}, DECLARED, None) is None

    def test_every_required_column_present_is_accepted(self):
        action = dict.fromkeys(DECLARED, 0.3)
        assert unrecordable_action_columns_error(action, DECLARED, DECLARED) is None

    def test_a_missing_required_column_is_named(self):
        action = {"a_shoulder": 0.3}
        msg = unrecordable_action_columns_error(action, DECLARED, DECLARED)
        assert msg is not None
        assert "'a_elbow'" in msg and "'a_grip'" in msg
        assert "a_shoulder" not in msg

    def test_a_declared_column_outside_the_required_set_is_not_this_frames_job(self):
        """A shared scene declares columns for robots this rollout does not drive."""
        declared = [*DECLARED, "bob__a_shoulder"]
        action = dict.fromkeys(DECLARED, 0.3)
        assert unrecordable_action_columns_error(action, declared, DECLARED) is None

    def test_a_required_column_the_schema_never_declared_is_not_a_recorded_column(self):
        action = dict.fromkeys(DECLARED, 0.3)
        required = [*DECLARED, "a_phantom"]
        assert unrecordable_action_columns_error(action, DECLARED, required) is None

    def test_an_action_carrying_nothing_at_all_is_refused(self):
        msg = unrecordable_action_columns_error({}, DECLARED, DECLARED)
        assert msg is not None
        assert all(f"'{key}'" in msg for key in DECLARED)

    def test_the_message_explains_why_no_placeholder_is_correct_and_what_to_do(self):
        msg = unrecordable_action_columns_error({}, DECLARED, DECLARED)
        assert msg is not None
        # names the hazard, not just the symptom
        assert "travel to zero" in msg
        # and the two reasons a substitute cannot be synthesized
        assert "different units" in msg
        assert "not " in msg and "injective" in msg
        # and the remedy, pointing at the existing width diagnostic
        assert "diagnose_action_dim" in msg

    def test_the_reported_order_follows_the_declared_schema(self):
        msg = unrecordable_action_columns_error({}, DECLARED, DECLARED)
        assert msg is not None
        assert msg.index("'a_shoulder'") < msg.index("'a_elbow'") < msg.index("'a_grip'")


class TestAddFrameRefusesToFabricateAColumn:
    """The guard where the fabrication used to happen."""

    def _recorder(self):
        ds = _CapturingDataset(_state_action_features(["shoulder", "elbow", "grip"], DECLARED))
        return DatasetRecorder(dataset=ds, task="t"), ds

    def test_a_missing_required_column_raises(self):
        rec, _ds = self._recorder()
        with pytest.raises(ValueError, match="a_grip"):
            rec.add_frame(
                observation={"shoulder": 0.1, "elbow": 0.2, "grip": 0.3},
                action={"a_shoulder": 0.1, "a_elbow": 0.2},
                required_action_keys=DECLARED,
            )

    def test_nothing_is_written_for_a_refused_frame(self):
        """The refusal must not leave a half-built frame in the episode buffer."""
        rec, ds = self._recorder()
        with pytest.raises(ValueError):
            rec.add_frame(
                observation={"shoulder": 0.1, "elbow": 0.2, "grip": 0.3},
                action={"a_shoulder": 0.1},
                required_action_keys=DECLARED,
            )
        assert ds.frames == []

    def test_a_complete_frame_records_exactly_what_was_commanded(self):
        rec, ds = self._recorder()
        rec.add_frame(
            observation={"shoulder": 0.1, "elbow": 0.2, "grip": 0.3},
            action={"a_shoulder": 0.4, "a_elbow": 0.5, "a_grip": 0.6},
            required_action_keys=DECLARED,
        )
        assert len(ds.frames) == 1
        np.testing.assert_allclose(ds.frames[0]["action"], [0.4, 0.5, 0.6], atol=1e-6)

    def test_the_default_still_fills_unmatched_columns(self):
        """Unchanged for callers that make no claim - e.g. a shared-scene recording.

        This is the behaviour the ``required_action_keys`` scoping deliberately
        preserves: a rollout driving one robot in a two-robot scene supplies only
        its own columns, and the robots it does not drive are not its to report.
        """
        rec, ds = self._recorder()
        rec.add_frame(
            observation={"shoulder": 0.1, "elbow": 0.2, "grip": 0.3},
            action={"a_shoulder": 0.4},
        )
        np.testing.assert_allclose(ds.frames[0]["action"], [0.4, 0.0, 0.0], atol=1e-6)

    def test_an_actionless_frame_does_not_poison_the_declared_column_cache(self):
        """The declared columns are resolved from the schema, not from frame one.

        The guard has to run before the ``if action:`` branch so an empty action
        is refused too; that must not let the first frame cache an empty column
        list and silently drop every later action.
        """
        rec, ds = self._recorder()
        rec.add_frame(observation={"shoulder": 0.1, "elbow": 0.2, "grip": 0.3}, action={})
        rec.add_frame(
            observation={"shoulder": 0.1, "elbow": 0.2, "grip": 0.3},
            action={"a_shoulder": 0.4, "a_elbow": 0.5, "a_grip": 0.6},
            required_action_keys=DECLARED,
        )
        np.testing.assert_allclose(ds.frames[-1]["action"], [0.4, 0.5, 0.6], atol=1e-6)


class TestEveryRecordingHookDeclaresItsActionColumns:
    """No backend may forward a policy's action without scoping the columns.

    Read statically so the check covers the Isaac Sim and Newton backends, whose
    runtimes are not installed here. A backend that stops passing
    ``required_action_keys`` silently returns to fabricating the columns its
    policy did not produce.
    """

    HOOK_MODULES = [
        "strands_robots/simulation/mujoco/simulation.py",
        "strands_robots/simulation/isaac/recording.py",
        "strands_robots/simulation/newton/recording.py",
    ]

    @pytest.mark.parametrize("module_path", HOOK_MODULES)
    def test_every_add_frame_call_scopes_its_action_columns(self, module_path):
        source = (Path(__file__).resolve().parents[1] / module_path).read_text()
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_frame"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"rec", "recorder"}
        ]
        assert calls, f"{module_path}: no recorder.add_frame call found"
        for call in calls:
            keywords = {kw.arg for kw in call.keywords}
            assert "required_action_keys" in keywords, (
                f"{module_path}:{call.lineno} calls add_frame without required_action_keys, "
                "so a policy that omits an actuator would have that column fabricated."
            )


# -- End to end in sim: the record -> replay round trip ------------------------

_ARM_MJCF = """
<mujoco model="two_link">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <body name="upper" pos="0 0 0.5">
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="4"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.03"/>
      <body name="lower" pos="0 0 0.2">
        <joint name="elbow" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="4"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.025"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a_shoulder" joint="shoulder" kp="50"/>
    <position name="a_elbow" joint="elbow" kp="50"/>
  </actuator>
</mujoco>
"""


def _episode_parquets(root):
    """Recorded frame parquets only - LeRobot also writes meta/ parquets."""
    return sorted(path for path in root.rglob("*.parquet") if "data" in path.parts)


def _arm_sim(tmp_path, tool_name):
    from strands_robots import Simulation

    mjcf = tmp_path / "two_link.xml"
    mjcf.write_text(_ARM_MJCF)
    sim = Simulation(backend="mujoco", tool_name=tool_name, mesh=False)
    sim.create_world()
    assert sim.add_robot(name="arm", urdf_path=str(mjcf))["status"] == "success"
    assert sim.robot_action_keys(robot_name="arm") == ["a_shoulder", "a_elbow"]
    return sim


class _PartialPolicy(Policy):
    """A policy driving only the leading actuators, as a narrow checkpoint does."""

    def __init__(self, driven):
        self._driven = driven
        self.keys: list[str] = []

    @property
    def provider_name(self) -> str:
        return "partial"

    @property
    def requires_images(self) -> bool:
        return False

    def set_robot_state_keys(self, keys) -> None:
        self.keys = list(keys)

    async def get_actions(self, observation_dict, instruction, **kwargs):
        return [dict.fromkeys(self.keys[: self._driven], 0.4) for _ in range(8)]


def test_a_short_action_rollout_records_no_episode_to_replay(tmp_path):
    """The round-trip hazard, closed at its source.

    A recorded ``0.0`` for an actuator the policy never commanded is
    indistinguishable from a real command, and :meth:`replay_episode` re-issues
    it - driving that joint to zero at servo speed. The rollout is refused
    instead, so no episode reaches disk for a replay to re-issue.
    """
    pytest.importorskip("mujoco")
    pytest.importorskip("lerobot")

    sim = _arm_sim(tmp_path, "refuse_sim")
    try:
        started = sim.start_recording(repo_id="local/short_action_refused", task="t", fps=30, root=str(tmp_path / "ds"))
        assert started["status"] == "success"
        result = sim.run_policy(
            robot_name="arm",
            policy_object=_PartialPolicy(driven=1),
            instruction="t",
            n_steps=10,
            control_frequency=30.0,
        )
        assert result["status"] == "error"
        text = str(result)
        assert "a_elbow" in text
        assert "never issued" in text
        sim.stop_recording()
    finally:
        sim.cleanup()

    assert _episode_parquets(tmp_path / "ds") == [], "a refused rollout must not leave an episode on disk"


def test_a_complete_rollout_records_the_commands_that_were_issued(tmp_path):
    """The counterpart: a policy covering every actuator still records normally.

    Pins that the guard rejects only frames it cannot record faithfully - the
    recorded action column equals the command the policy actually issued, which
    is what makes replaying the episode reproduce the rollout.
    """
    pytest.importorskip("mujoco")
    pytest.importorskip("lerobot")
    pd = pytest.importorskip("pandas")

    sim = _arm_sim(tmp_path, "record_sim")
    try:
        assert (
            sim.start_recording(repo_id="local/full_action_ok", task="t", fps=30, root=str(tmp_path / "ds"))["status"]
            == "success"
        )
        result = sim.run_policy(
            robot_name="arm",
            policy_object=_PartialPolicy(driven=2),
            instruction="t",
            n_steps=10,
            control_frequency=30.0,
        )
        assert result["status"] == "success"
        assert sim.stop_recording()["status"] == "success"
    finally:
        sim.cleanup()

    parquets = _episode_parquets(tmp_path / "ds")
    assert parquets, "a complete rollout must record an episode"
    actions = np.stack(pd.concat([pd.read_parquet(p) for p in parquets])["action"].to_numpy())
    assert actions.shape[1] == 2
    # Both columns carry the issued command, not a fabricated placeholder.
    np.testing.assert_allclose(actions, 0.4, atol=1e-5)


def test_a_recording_never_falls_back_to_fabricating_when_the_columns_are_unknown(tmp_path, monkeypatch):
    """The scope is load-bearing for a recording, not best-effort.

    ``robot_action_keys`` is deliberately best-effort for the runner's fail-fast
    probe - see ``test_policy_runner_action_keys_probe_failsoft`` - which is why
    the recording hook resolves it lazily and only when a recorder is attached.
    But a recording cannot proceed without it: not knowing which columns the
    rollout owes is not a licence to fill them in. The rollout must fail rather
    than persist a frame it could not check.
    """
    pytest.importorskip("mujoco")

    class _Recorder:
        def __init__(self):
            self.frames = 0

        def add_frame(self, observation, action, task="", required_action_keys=None):
            self.frames += 1

        def save_episode(self):
            return {"status": "success"}

    sim = _arm_sim(tmp_path, "unknown_cols_sim")
    rec = _Recorder()
    try:
        assert sim._world is not None
        sim._world._backend_state["recording"] = True
        sim._world._backend_state["trajectory"] = []
        sim._world._backend_state["dataset_recorder"] = rec

        def _boom(robot_name: str) -> list[str]:
            raise RuntimeError("action keys unavailable")

        monkeypatch.setattr(sim, "robot_action_keys", _boom)
        result = sim.run_policy(
            robot_name="arm",
            policy_object=_PartialPolicy(driven=2),
            instruction="t",
            n_steps=6,
            control_frequency=50.0,
        )
        assert result["status"] == "error"
    finally:
        sim.cleanup()

    assert rec.frames == 0, "no frame may be recorded when the owed columns are unknown"
