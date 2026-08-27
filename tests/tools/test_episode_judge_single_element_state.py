"""A recording whose state has one component is sampled, not raised on.

LeRobot stores a feature of shape ``[1]`` unwrapped: the ``observation.state``
parquet column holds a scalar float where a wider state holds a list, while
``meta/info.json`` declares ``shape [1]`` and one joint name either way.
:func:`strands_robots.tools.episode_judge.sample_frames` read every frame's
state by iterating that value, so a single-DOF recording raised
``TypeError: 'float' object is not iterable`` - out through the ``@tool``
boundary, because ``TypeError`` is not among the exceptions the tool converts
into an error envelope. The module promises the opposite in its own docstring:
"Every tool returns the structured ``{"status", "content"}`` envelope and never
raises - a judge run over a hundred episodes must report the one episode it
could not read, not die on it". The recording is healthy, so there was nothing
to report either; the state is the one-element vector its metadata declares.

The break also reached further than the state fields: it happened before
``include_images`` was consulted, so the whole multimodal path was unreachable
for such a dataset.

Covers:

* the real writer's shapes, measured rather than assumed - one component is
  stored as a scalar, two as a list, both declaring their width in
  ``meta/info.json``;
* a single-DOF recording is sampled end to end: one-element state vectors, a
  populated motion summary, and decoded camera blocks;
* a wider recording is unchanged (the control that scopes the fix);
* the column shape is read the same way with no simulator installed, over a
  hand-written dataset, so the rule is graded wherever pyarrow is;
* ``load_episode`` and ``sample_frames`` agree on how wide the state is - the
  asymmetry the break produced, where the tool that describes an episode
  reported one joint name and the tool that samples it raised.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

import strands_robots.tools.episode_judge as M

pq = pytest.importorskip("pyarrow.parquet", reason="the hand-written dataset fixture writes parquet")
import pyarrow as pa  # noqa: E402

os.environ.setdefault("MUJOCO_GL", "egl")

# The tools are wrapped by the Strands @tool decorator; call the raw functions.
_load_episode = getattr(M.load_episode, "__wrapped__", None) or M.load_episode
_sample_frames = getattr(M.sample_frames, "__wrapped__", None) or M.sample_frames

#: A one-joint arm carrying its own camera. A single-DOF scene is the ordinary
#: shape for a gripper, a linear stage or a pan unit, and it is what the state
#: column collapses to a scalar for.
_ONE_JOINT_XML = """
<mujoco model="one_joint">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.8 1"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <camera name="wrist" pos="0.25 0 0.05" xyaxes="0 -1 0 0 0 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
  </actuator>
</mujoco>
"""

#: The same arm with a second joint, so the state is stored as a list. The
#: control for every rule below: the fix must not move this path.
_TWO_JOINT_XML = """
<mujoco model="two_joint">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.8 1"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <body name="link" pos="0 0 0.1">
        <geom type="capsule" size="0.02 0.05" rgba="0.8 0.3 0.3 1"/>
        <joint name="elbow_flex" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
    <position name="elbow_flex_act" joint="elbow_flex" kp="50"/>
  </actuator>
</mujoco>
"""

_FRAMES = 8


def _text(result: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in result.get("content", []) if "text" in block)


def _payload(result: dict[str, Any]) -> dict[str, Any]:
    return next((block["json"] for block in result.get("content", []) if "json" in block), {})


def _state_column(root: Path) -> list[Any]:
    """The recorded ``observation.state`` column, exactly as parquet holds it."""
    table = pq.read_table(sorted((root / "data").glob("**/*.parquet"))[0])
    return list(table.to_pydict()["observation.state"])


def _declared_state(root: Path) -> dict[str, Any]:
    """The ``observation.state`` feature spec from ``meta/info.json``."""
    info = json.loads((root / "meta" / "info.json").read_text())
    return info["features"]["observation.state"]


def _declared_cameras(root: Path) -> list[str]:
    """The camera keys the recording declared, in the order the tool sorts them."""
    info = json.loads((root / "meta" / "info.json").read_text())
    return sorted(
        key.removeprefix("observation.images.") for key in info["features"] if key.startswith("observation.images.")
    )


def _record(tmp_path: Path, xml: str, name: str) -> Path:
    """Record one episode of a mock rollout over ``xml`` and return the root."""
    from strands_robots.simulation import Simulation

    model = tmp_path / f"{name}.xml"
    model.write_text(xml)
    root = tmp_path / "dataset"
    sim = Simulation()
    sim.create_world()
    sim.add_robot("arm", urdf_path=str(model))
    try:
        started = sim.start_recording(repo_id=f"local/{name}", root=str(root), task="pan the arm", overwrite=True)
        assert started["status"] == "success", _text(started)
        rollout = sim.run_policy(robot_name="arm", policy_provider="mock", n_steps=_FRAMES, control_frequency=30.0)
        assert rollout["status"] == "success", _text(rollout)
        stopped = sim.stop_recording()
        assert stopped["status"] == "success", _text(stopped)
    finally:
        sim.destroy()
    return root


def _write_hand_built_dataset(root: Path, state_column: list[Any], names: list[str] | None) -> None:
    """Write a LeRobot-v3-shaped dataset whose state column is given verbatim.

    Lets the column shape be chosen directly, so the reader's rule is graded
    on an install with no simulator and no lerobot - the recorded fixtures
    above establish that these shapes are the ones the real writer produces.
    """
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "total_episodes": 1,
                "total_frames": len(state_column),
                "features": {"observation.state": {"dtype": "float32", "shape": [len(names or [])], "names": names}},
            }
        )
    )
    pq.write_table(
        pa.table({"episode_index": [0], "length": [len(state_column)]}),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": [0] * len(state_column),
                "frame_index": list(range(len(state_column))),
                "timestamp": [frame / 30.0 for frame in range(len(state_column))],
                "observation.state": state_column,
            }
        ),
        root / "data" / "chunk-000" / "file-000.parquet",
    )


@pytest.fixture(scope="module")
def one_joint_root(tmp_path_factory):
    pytest.importorskip("mujoco", reason="recording a scene needs the mujoco backend")
    # The dataset stack, not the top-level package: lerobot imports fine while
    # lerobot.datasets.lerobot_dataset does not on an install whose optional
    # dataset dependencies are missing, which is exactly what recording needs.
    pytest.importorskip(
        "lerobot.datasets.lerobot_dataset", reason="writing a LeRobot dataset needs lerobot's dataset stack"
    )
    return _record(tmp_path_factory.mktemp("one_joint"), _ONE_JOINT_XML, "one_joint")


@pytest.fixture(scope="module")
def two_joint_root(tmp_path_factory):
    pytest.importorskip("mujoco", reason="recording a scene needs the mujoco backend")
    # The dataset stack, not the top-level package: lerobot imports fine while
    # lerobot.datasets.lerobot_dataset does not on an install whose optional
    # dataset dependencies are missing, which is exactly what recording needs.
    pytest.importorskip(
        "lerobot.datasets.lerobot_dataset", reason="writing a LeRobot dataset needs lerobot's dataset stack"
    )
    return _record(tmp_path_factory.mktemp("two_joint"), _TWO_JOINT_XML, "two_joint")


class TestThePremise:
    """The two widths really are stored differently by the real writer.

    Without this the rules below could be satisfied by a hand-written fixture
    of an invented shape: the scalar column is what LeRobot actually writes,
    not a shape chosen to make a reader fail.
    """

    def test_one_component_is_stored_as_a_scalar(self, one_joint_root):
        column = _state_column(one_joint_root)
        assert column, "the recording wrote no frames"
        assert all(isinstance(value, float) for value in column), column[:3]

    def test_two_components_are_stored_as_a_list(self, two_joint_root):
        column = _state_column(two_joint_root)
        assert column, "the recording wrote no frames"
        assert all(isinstance(value, list) and len(value) == 2 for value in column), column[:3]

    def test_both_declare_their_width_in_the_metadata(self, one_joint_root, two_joint_root):
        # The scalar is not an undeclared state: the dataset says how wide it
        # is, which is why reading it as a one-element vector is a reading and
        # not a repair.
        assert _declared_state(one_joint_root)["shape"] == [1], _declared_state(one_joint_root)
        assert _declared_state(one_joint_root)["names"] == ["shoulder_pan"], _declared_state(one_joint_root)
        assert _declared_state(two_joint_root)["shape"] == [2], _declared_state(two_joint_root)


class TestASingleDofRecordingIsSampled:
    """The regression: the tool reads the recording instead of raising."""

    def test_every_sample_carries_the_one_element_vector(self, one_joint_root):
        result = _sample_frames(str(one_joint_root), 0, n_frames=3)
        assert result["status"] == "success", _text(result)
        samples = _payload(result)["samples"]
        assert samples, result
        assert [sample["state"] for sample in samples] == [
            [value] for value in (_state_column(one_joint_root)[sample["frame_index"]] for sample in samples)
        ], samples

    def test_the_motion_summary_is_a_number_rather_than_absent(self, one_joint_root):
        # max_state_delta and rms_state_jerk both walk the state vectors, so a
        # width the reader cannot express takes the summary with it.
        payload = _payload(_sample_frames(str(one_joint_root), 0, n_frames=3))
        assert isinstance(payload["max_state_delta"], float), payload
        assert isinstance(payload["rms_state_jerk"], float), payload

    def test_the_camera_frames_decode(self, one_joint_root):
        # The state read happens before include_images is consulted, so the
        # whole multimodal path was unreachable for this dataset.
        cameras = _declared_cameras(one_joint_root)
        assert cameras, "the recording declared no camera column"
        result = _sample_frames(str(one_joint_root), 0, n_frames=2, include_images=True)
        assert result["status"] == "success", _text(result)
        blocks = [block for block in result["content"] if "image" in block]
        assert len(blocks) == 2 * len(cameras), (len(blocks), cameras, _text(result))
        assert all(block["image"]["source"]["bytes"] for block in blocks)


class TestAWiderRecordingIsUnchanged:
    """The control: the fix is scoped to the width that was unreadable."""

    def test_a_two_component_state_is_still_a_two_vector(self, two_joint_root):
        result = _sample_frames(str(two_joint_root), 0, n_frames=3)
        assert result["status"] == "success", _text(result)
        states = [sample["state"] for sample in _payload(result)["samples"]]
        assert states and all(len(state) == 2 for state in states), states


class TestTheColumnShapeIsReadWithNoSimulator:
    """The same rule over a hand-written dataset, so pyarrow alone grades it."""

    def test_a_scalar_column_is_a_one_element_vector(self, tmp_path):
        root = tmp_path / "scalar"
        root.mkdir()
        _write_hand_built_dataset(root, [0.0, 0.25, 0.5, 0.75, 1.0], ["gripper"])
        result = _sample_frames(str(root), 0, n_frames=5)
        assert result["status"] == "success", _text(result)
        assert [sample["state"] for sample in _payload(result)["samples"]] == [
            [0.0],
            [0.25],
            [0.5],
            [0.75],
            [1.0],
        ], _payload(result)["samples"]

    def test_a_list_column_is_read_component_wise(self, tmp_path):
        root = tmp_path / "vector"
        root.mkdir()
        _write_hand_built_dataset(root, [[0.0, 1.0], [0.5, 1.5]], ["pan", "lift"])
        result = _sample_frames(str(root), 0, n_frames=2)
        assert result["status"] == "success", _text(result)
        assert [sample["state"] for sample in _payload(result)["samples"]] == [[0.0, 1.0], [0.5, 1.5]]

    def test_a_dataset_with_no_state_column_still_reports_no_state(self, tmp_path):
        # The boundary the reader already had: absent is absent, and a scalar
        # is not read as absent.
        root = tmp_path / "stateless"
        root.mkdir()
        _write_hand_built_dataset(root, [0.0, 1.0], ["gripper"])
        table = pq.read_table(sorted((root / "data").glob("**/*.parquet"))[0])
        pq.write_table(table.drop_columns(["observation.state"]), sorted((root / "data").glob("**/*.parquet"))[0])
        result = _sample_frames(str(root), 0, n_frames=2)
        assert result["status"] == "success", _text(result)
        assert [sample["state"] for sample in _payload(result)["samples"]] == [None, None]

    def test_the_two_tools_agree_on_how_wide_the_state_is(self, tmp_path):
        # The asymmetry the break produced: load_episode reported one joint
        # name off the metadata while sample_frames raised on the column that
        # metadata describes.
        root = tmp_path / "agreement"
        root.mkdir()
        _write_hand_built_dataset(root, [0.0, 0.5], ["gripper"])
        described = _load_episode(str(root), 0)
        assert described["status"] == "success", _text(described)
        names = _payload(described)["state_names"]
        sampled = _sample_frames(str(root), 0, n_frames=2)
        assert sampled["status"] == "success", _text(sampled)
        assert all(len(sample["state"]) == len(names) for sample in _payload(sampled)["samples"]), names
