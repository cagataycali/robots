"""A robot cannot be added into a live recording's frozen feature schema.

A recorder declares its features once, from the robots attached at
``start_recording``, and a LeRobotDataset cannot gain a column afterwards.
Before this refusal, ``add_robot`` succeeded and the first recorded rollout
ended one of two silent ways: against a camera-only schema it died inside
lerobot ("Feature mismatch in `frame` dictionary: Extra features: {'action',
'observation.state'}") after 0 frames and left a ``meta/info.json`` shell, and
against a 6-wide so101 schema a 9-joint panda rollout raised nothing at all -
12 frames whose state and action were all zeros, the frozen robot's values
rather than the moved robot's.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("lerobot")

from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine


def _text(result: dict) -> str:
    return "\n".join(c["text"] for c in result["content"] if isinstance(c, dict) and "text" in c)


@pytest.fixture
def sim():
    engine = MuJoCoSimEngine(tool_name="add_robot_recording", mesh=False)
    engine.create_world()
    try:
        yield engine
    finally:
        engine.cleanup()


@pytest.mark.parametrize(
    ("attached_first", "expected_robots", "frozen_shape", "schema_phrase"),
    [
        pytest.param(
            None,
            [],
            [],
            "it declares no observation.state/action columns at all (a camera-only recording)",
            id="camera-only-schema",
        ),
        pytest.param(
            "so101",
            ["so101"],
            [6],
            "it declares observation.state/action shaped [6]",
            id="six-wide-so101-schema",
        ),
    ],
)
def test_add_robot_during_a_recording_is_refused(
    sim, tmp_path, attached_first, expected_robots, frozen_shape, schema_phrase
):
    if attached_first:
        assert sim.add_robot(name=attached_first, data_config="so101")["status"] == "success"
    assert sim.start_recording(repo_id="lab/frozen", root=str(tmp_path / "frozen"), fps=30)["status"] == "success"

    refusal = sim.add_robot(name="panda", data_config="panda", position=[0.6, 0, 0])

    assert refusal["status"] == "error"
    text = _text(refusal)
    assert "a dataset recording is open on 'lab/frozen'" in text
    assert schema_phrase in text
    assert text.endswith(
        "Call stop_recording first - it saves the buffered frames - then add_robot, then start_recording again."
    )
    assert refusal["content"][1]["json"] == {
        "recording": True,
        "repo_id": "lab/frozen",
        "frozen_state_shape": frozen_shape,
        "requested_robot": "panda",
    }
    # The world is left exactly as the frozen schema describes it.
    assert sim.list_robots() == expected_robots


def test_the_sequence_the_refusal_names_records_both_robots(sim, tmp_path):
    """stop_recording, add_robot, start_recording again - and the columns follow."""
    assert sim.add_robot(name="so101", data_config="so101")["status"] == "success"
    assert sim.start_recording(repo_id="lab/one", root=str(tmp_path / "one"), fps=30)["status"] == "success"
    assert (
        sim.run_policy(robot_name="so101", policy_provider="mock", duration=0.3, control_frequency=30)["status"]
        == "success"
    )
    assert sim.stop_recording()["status"] == "success"

    assert sim.add_robot(name="panda", data_config="panda", position=[0.6, 0, 0])["status"] == "success"
    root = tmp_path / "two"
    assert sim.start_recording(repo_id="lab/two", root=str(root), fps=30)["status"] == "success"
    assert (
        sim.run_policy(robot_name="panda", policy_provider="mock", duration=0.3, control_frequency=30)["status"]
        == "success"
    )
    assert sim.stop_recording()["status"] == "success"

    # The second dataset was declared with BOTH robots in the world, so it
    # reopens carrying columns for each - 6 so101 joints plus 9 panda ones - the
    # width the first dataset's frozen schema could never have grown to.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id="lab/two", root=str(root))
    assert dataset.num_frames > 0
    assert tuple(dataset.features["observation.state"]["shape"]) == (6 + 9,)
    assert any(abs(float(v)) > 0 for v in dataset[dataset.num_frames - 1]["observation.state"])
