"""An undriven robot's declared ``observation.state`` columns are measurements.

``start_recording`` declares ``observation.state`` over every robot in the
scene, prefixing each column with its robot's name. A single-policy rollout's
recording hook supplies only the driven robot's observation, so before this was
fixed every column belonging to another robot fell through to
``DatasetRecorder.add_frame``'s ``0.0`` fill - recorded in the same column, with
the same dtype, as a measurement, for every frame, under ``status="success"``.

Nothing downstream can tell that zero from a real reading: a policy trained on
the dataset learns the other robot is permanently at its zero pose. The
disagreement is unbounded, because an undriven robot keeps whatever pose it was
placed in.

The *action* half of the same frame is a separate question and deliberately not
touched here - no command was issued to a robot this rollout does not drive, so
there is no truthful value to write (#1715). A state column has one: the robot
is in the scene and its joint positions are readable at that instant.

``run_multi_policy``'s synchronized loop already merged every robot's state into
one frame per step, so this was the three single-policy hooks disagreeing with
the one path that had it right.
"""

from __future__ import annotations

import ast
import inspect
import json
import pathlib
import textwrap
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("lerobot")

from strands_robots.simulation.recording import undriven_robot_state  # noqa: E402

_ROBOT_XML = """
<mujoco model="probe_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.04 0.04" rgba="0.3 0.3 0.8 1"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <body name="link" pos="0 0 0.08">
        <geom type="capsule" fromto="0 0 0 0 0 0.12" size="0.02" rgba="0.8 0.3 0.3 1"/>
        <joint name="elbow" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="80"/>
    <position name="elbow_act" joint="elbow" kp="80"/>
  </actuator>
</mujoco>
"""

#: The pose the undriven robot is parked in. Every component is far from ``0.0``
#: so a recorded zero cannot be mistaken for this pose - the whole point of the
#: fixture. Asserted against the fill in ``test_the_parked_pose_is_not_the_fill``.
_PARKED = {"shoulder_pan": 0.9, "elbow": -0.7}

_SETTLE_SUBSTEPS = 600


@pytest.fixture
def two_robot_recording(tmp_path: pathlib.Path) -> Any:
    """A two-robot scene with ``bob`` parked away from ``alice`` and settled."""
    from strands_robots.simulation import Simulation

    xml = tmp_path / "probe_arm.xml"
    xml.write_text(_ROBOT_XML, encoding="utf-8")

    sim = Simulation()
    sim.create_world()
    sim.add_robot("alice", urdf_path=str(xml), position=[0.0, 0.0, 0.0])
    # Far enough that no contact couples the two, so the parked pose holds and
    # the last recorded frame can be compared against a post-rollout reading.
    sim.add_robot("bob", urdf_path=str(xml), position=[2.0, 0.0, 0.0])
    assert sim.send_action(_PARKED, robot_name="bob", n_substeps=_SETTLE_SUBSTEPS)["status"] == "success"
    return sim


def _record_one_episode(sim: Any, root: pathlib.Path) -> dict[str, Any]:
    """Drive ``alice`` for a few steps with a recording open, then read the parquet."""
    assert (
        sim.start_recording(
            repo_id="local/undriven_state_probe",
            task="probe",
            fps=10,
            root=str(root),
            cameras=["default"],
            overwrite=True,
        )["status"]
        == "success"
    )
    assert (
        sim.run_policy(
            robot_name="alice",
            policy_provider="mock",
            instruction="probe",
            n_steps=5,
            control_frequency=10.0,
            fast_mode=True,
        )["status"]
        == "success"
    )
    assert sim.stop_recording()["status"] == "success"

    # Read back through the public reader rather than the parquet, so this is
    # the round trip a consumer of the dataset actually performs.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    names = info["features"]["observation.state"]["names"]
    dataset = LeRobotDataset("local/undriven_state_probe", root=str(root))
    assert len(dataset) > 0, f"reopened dataset at {root} is empty"
    state = np.stack([np.asarray(dataset[i]["observation.state"], dtype=np.float64) for i in range(len(dataset))])
    return {"names": names, "state": state}


class TestAnUndrivenRobotsStateIsRecordedAsMeasured:
    """The regression: a declared state column holds a reading, not the fill."""

    def test_the_parked_pose_is_not_the_fill(self) -> None:
        """Premise: a recorded zero would be indistinguishable from this pose."""
        assert all(abs(v) > 0.1 for v in _PARKED.values()), (
            f"the parked pose {_PARKED} must differ from add_frame's 0.0 fill in every "
            "component, or a recording that fabricates zeros passes these tests"
        )

    def test_the_schema_declares_the_undriven_robots_columns(self, two_robot_recording: Any, tmp_path: Any) -> None:
        """Premise: without declared ``bob__*`` columns there is nothing to fill."""
        recorded = _record_one_episode(two_robot_recording, tmp_path / "ds_schema")
        assert [n for n in recorded["names"] if n.startswith("bob__")] == ["bob__shoulder_pan", "bob__elbow"]

    def test_the_undriven_robots_columns_are_not_the_zero_fill(self, two_robot_recording: Any, tmp_path: Any) -> None:
        """The defect: every ``bob__*`` value on disk used to be ``0.0``."""
        recorded = _record_one_episode(two_robot_recording, tmp_path / "ds_zero")
        idx = [i for i, n in enumerate(recorded["names"]) if n.startswith("bob__")]
        column = recorded["state"][:, idx]
        assert not np.allclose(column, 0.0), (
            "every declared observation.state column of the undriven robot 'bob' was recorded "
            f"as the 0.0 fill while bob is parked at {_PARKED}: {column.tolist()}"
        )

    def test_the_undriven_robots_columns_match_the_engines_own_reading(
        self, two_robot_recording: Any, tmp_path: Any
    ) -> None:
        """The recorded value is the measurement, not merely non-zero."""
        sim = two_robot_recording
        recorded = _record_one_episode(sim, tmp_path / "ds_match")
        truth = sim.get_observation(robot_name="bob", skip_images=True)
        for i, name in enumerate(recorded["names"]):
            if not name.startswith("bob__"):
                continue
            joint = name.removeprefix("bob__")
            assert float(recorded["state"][-1][i]) == pytest.approx(float(truth[joint]), abs=1e-3), (
                f"{name} on disk disagrees with the engine's own reading of bob's {joint}"
            )

    def test_the_driven_robots_columns_still_track_a_real_trajectory(
        self, two_robot_recording: Any, tmp_path: Any
    ) -> None:
        """Control: filling the undriven columns must not flatten the driven ones."""
        recorded = _record_one_episode(two_robot_recording, tmp_path / "ds_driven")
        idx = [i for i, n in enumerate(recorded["names"]) if n.startswith("alice__")]
        assert idx, "premise: the driven robot must have declared columns too"
        column = recorded["state"][:, idx]
        assert np.ptp(column, axis=0).max() > 1e-4, (
            f"the driven robot's own columns no longer vary across the episode: {column.tolist()}"
        )

    def test_a_driven_column_is_never_overwritten_by_the_undriven_fill(
        self, two_robot_recording: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The precedence the fill depends on: this frame's own reading wins.

        The undriven fill and the driven observation are merged into one frame,
        so a fill that claimed a driven robot's column would silently replace a
        real per-step reading with a second read taken through a different path.
        Forced here by making the fill claim every column it must not own.
        """
        import strands_robots.simulation.mujoco.simulation as mujoco_sim

        sentinel = -12345.0
        monkeypatch.setattr(
            mujoco_sim,
            "undriven_robot_state",
            lambda engine, driven, names: {f"{driven}__shoulder_pan": sentinel, "bob__shoulder_pan": sentinel},
        )
        recorded = _record_one_episode(two_robot_recording, tmp_path / "ds_precedence")
        driven = recorded["names"].index("alice__shoulder_pan")
        assert not np.any(recorded["state"][:, driven] == sentinel), (
            "the undriven fill overwrote the driven robot's own per-step reading"
        )
        # The premise: the fill really was consulted, so the assertion above is
        # about precedence rather than about a patch that never ran.
        bystander = recorded["names"].index("bob__shoulder_pan")
        assert np.all(recorded["state"][:, bystander] == sentinel), "premise: the patched fill was not consulted"


class TestTheHelperReadsOnlyWhatItShould:
    """``undriven_robot_state`` is the single owner, so its rule is pinned here."""

    class _Engine:
        def __init__(self, states: dict[str, dict[str, Any]], raises: set[str] | None = None) -> None:
            self.states = states
            self.raises = raises or set()
            self.asked: list[str] = []

        def get_observation(self, robot_name: str | None = None, skip_images: bool = False) -> dict[str, Any]:
            assert skip_images is True, "a per-robot state read must not pay for a render"
            self.asked.append(str(robot_name))
            if robot_name in self.raises:
                raise RuntimeError("transient read failure")
            return self.states[str(robot_name)]

    def test_the_driven_robot_is_not_re_read(self) -> None:
        """The hook already carries the driven robot's own observation."""
        engine = self._Engine({"alice": {"j": 1.0}, "bob": {"j": 2.0}})
        assert undriven_robot_state(engine, "alice", ["alice", "bob"]) == {"bob__j": 2.0}
        assert engine.asked == ["bob"]

    def test_a_single_robot_scene_yields_nothing(self) -> None:
        """A one-robot schema is not prefixed, so there is nothing to fill."""
        engine = self._Engine({"alice": {"j": 1.0}})
        assert undriven_robot_state(engine, "alice", ["alice"]) == {}
        assert engine.asked == []

    def test_array_values_are_left_out(self) -> None:
        """A camera frame is keyed by the camera, never by a robot."""
        engine = self._Engine({"bob": {"j": 2.0, "wrist": np.zeros((2, 2, 3), dtype=np.uint8)}})
        assert undriven_robot_state(engine, "alice", ["alice", "bob"]) == {"bob__j": 2.0}

    def test_a_failed_read_does_not_lose_the_episode(self) -> None:
        """The driven robot's columns are the rollout's primary product."""
        engine = self._Engine({"bob": {"j": 2.0}, "carol": {"j": 3.0}}, raises={"bob"})
        assert undriven_robot_state(engine, "alice", ["alice", "bob", "carol"]) == {"carol__j": 3.0}


class TestEverySinglePolicyHookConsultsTheSharedOwner:
    """A fourth backend cannot reintroduce the fill by forgetting the helper."""

    #: Each backend's single-policy recording hook, as ``(module, attribute)``.
    HOOKS = [
        ("strands_robots.simulation.mujoco.simulation", "MuJoCoSimEngine"),
        ("strands_robots.simulation.isaac.recording", "IsaacRecordingMixin"),
        ("strands_robots.simulation.newton.recording", "NewtonRecordingMixin"),
    ]

    def _hook_source(self, module_name: str, attr: str) -> str:
        import importlib

        module = importlib.import_module(module_name)
        owner = getattr(module, attr)
        return inspect.getsource(owner._make_run_policy_hook)

    def test_the_hooks_under_survey_all_exist(self) -> None:
        """Premise: a renamed backend would make every check below vacuous."""
        for module_name, attr in self.HOOKS:
            assert self._hook_source(module_name, attr), f"{module_name}.{attr} has no readable hook source"

    @pytest.mark.parametrize(("module_name", "attr"), HOOKS)
    def test_the_hook_fills_undriven_columns_from_the_shared_owner(self, module_name: str, attr: str) -> None:
        """Each hook resolves the undriven columns through the one helper."""
        source = self._hook_source(module_name, attr)
        called = {
            node.func.id
            for node in ast.walk(ast.parse(textwrap.dedent(source)))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "undriven_robot_state" in called, (
            f"{module_name}.{attr}._make_run_policy_hook does not call undriven_robot_state, so a "
            "multi-robot recording made through it writes the other robots' declared "
            "observation.state columns as add_frame's 0.0 fill"
        )
