"""``Policy("<dataset dir>[?episode=n]")`` replays a recorded episode: record, replay, the same trajectory."""

from __future__ import annotations

import json

import numpy as np
import pytest

from strands_robots import Policy, Robot, Simulation
from strands_robots.policy import SequencePolicy

pytest.importorskip("lerobot.datasets.lerobot_dataset")
import pyarrow.parquet as pq  # noqa: E402  (lerobot[dataset] brings it)


def _states(root, episode):
    rows = [pq.read_table(f).to_pydict() for f in sorted((root / "data").glob("**/*.parquet"))]
    got = [
        (f, s)
        for r in rows
        for e, f, s in zip(r["episode_index"], r["frame_index"], r["observation.state"], strict=True)
        if e == episode
    ]
    return np.asarray([s for _, s in sorted(got)])


@pytest.mark.parametrize("units,amplitude", [("rad", 0.4), ("lerobot", 20.0)])  # radians; LeRobot degrees / percent
def test_a_recorded_episode_replays_to_the_same_trajectory(tmp_path, units, amplitude):
    root = tmp_path / "wave"
    with Simulation() as sim:
        arm = Robot("so101", sim=sim)
        acts = [amplitude * np.sin(i / 10 + np.arange(len(arm.action_keys))) for i in range(40)]
        arm.run(SequencePolicy(acts), steps=40, hz=50, record=root, units=units)
        sim.reset()
        replay = Policy(str(root))
        assert (replay.units, replay.hz, repr(replay)) == (units, 50.0, f"Policy('{root}?episode=0')")
        arm.run(replay, steps=40, record=root)  # the dataset's fps, the dataset's units, a second episode
    assert json.loads((root / "meta" / "units.json").read_text())["units"] == units
    gap = float(np.abs(_states(root, 1) - _states(root, 0)).max())
    print(f"[measured] replay in {units!r}: max |state_replay - state_recorded| = {gap:.2e}")
    assert gap < 1e-4
    assert Policy(f"{root}?episode=1").actions[0].shape == (len(arm.action_keys),)


_REFUSED = {
    "a dir that is neither": (lambda root, empty: str(empty), "neither a checkpoint nor a dataset"),
    "an episode it lacks": (lambda root, empty: f"{root}?episode=7", r"no episode 7; it has episodes \[0\]"),
    "a non-integer episode": (lambda root, empty: f"{root}?episode=last", "episode must be an integer"),
}


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    root = tmp_path_factory.mktemp("ds") / "one"
    Robot("so101").run(lambda obs: {"elbow_flex": 0.4}, steps=5, hz=30, record=root)
    return root


@pytest.mark.parametrize("spec,match", _REFUSED.values(), ids=_REFUSED.keys())
def test_a_local_spec_is_refused_locally_never_sent_to_the_hub(recorded, tmp_path, spec, match):
    with pytest.raises(ValueError, match=match):
        Policy(spec(recorded, tmp_path))


def test_a_dataset_that_does_not_say_its_units_needs_them_from_the_caller(recorded, tmp_path):
    foreign = tmp_path / "foreign"
    (foreign / "meta").mkdir(parents=True)
    (foreign / "data").symlink_to(recorded / "data", target_is_directory=True)
    (foreign / "meta" / "info.json").write_text((recorded / "meta" / "info.json").read_text())
    with pytest.raises(ValueError, match="pass units="):
        Policy(str(foreign))
    assert Policy(str(foreign), units="rad").units == "rad"


def test_a_replay_on_a_robot_with_keys_the_dataset_lacks_is_refused_by_name(recorded):
    with pytest.raises(ValueError, match="has no action for"):
        Robot("g1").run(Policy(str(recorded)), steps=1)
