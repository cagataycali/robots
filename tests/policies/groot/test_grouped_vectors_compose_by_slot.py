"""A GR00T grouped vector is reachable from a robot's per-joint readings.

Every GR00T SO-ARM embodiment declares grouped modality keys - ``state.single_arm``
is five joints wide and ``action.single_arm`` five columns - while a robot
publishes one reading per joint: a MuJoCo SO-101 observation carries ``'1'`` ..
``'6'``, a driver carries its own names. The mapping is a 1:1 name map, so
before slots the two shapes could not be bridged and both directions failed
silently:

* several robot keys naming ``state.single_arm`` sent a one-wide vector holding
  whichever reading dict order reached last, and the model saw a state of the
  wrong width with no line saying so;
* one actuator mapped to ``action.single_arm`` received the whole five-wide row
  as its command, and the other four joints were never addressed.

Naming the slot (``"single_arm[3]"``) states the composition in the caller's own
mapping, in both directions and in either inference mode. A mapping that cannot
be composed is refused by name rather than honoured as something narrower, since
a zero or a dropped component is indistinguishable from a real reading.
"""

import tracemalloc

import numpy as np
import pytest

from strands_robots.policies.groot.policy import Gr00tPolicy

#: Five joint scalars into ``single_arm``, the sixth into ``gripper`` - the
#: binding an SO-101 sim observation needs against any SO-ARM data config.
SO101_OBS_MAPPING = {
    "room": "video.room",
    "wrist": "video.wrist",
    **{f"{i}": f"state.single_arm[{i - 1}]" for i in range(1, 6)},
    "6": "state.gripper[0]",
}
SO101_ACTION_MAPPING = {
    **{f"action.single_arm[{i - 1}]": f"{i}" for i in range(1, 6)},
    "action.gripper[0]": "6",
}


def _observation() -> dict:
    obs: dict = {f"{i}": i / 10 for i in range(1, 7)}
    obs["room"] = np.zeros((8, 8, 3), np.uint8)
    obs["wrist"] = np.zeros((8, 8, 3), np.uint8)
    return obs


def _policy(**kwargs) -> Gr00tPolicy:
    return Gr00tPolicy(data_config="so101_dualcam", groot_version="n1.7", **kwargs)


def test_slotted_state_keys_compose_one_vector_per_model_key():
    payload = _policy(observation_mapping=SO101_OBS_MAPPING)._prepare_observation(_observation(), "pick up the cube")

    single_arm = payload["state"]["single_arm"]
    assert single_arm.shape == (1, 1, 5)
    assert single_arm.ravel().tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
    assert payload["state"]["gripper"].shape == (1, 1, 1)
    assert payload["state"]["gripper"].ravel().tolist() == pytest.approx([0.6])


@pytest.mark.parametrize("unpack", ["_unpack_actions", "_unpack_service_actions"])
def test_a_grouped_action_column_reaches_the_joint_its_slot_names(unpack):
    chunk = {
        "action.single_arm": np.arange(16 * 5, dtype=np.float32).reshape(1, 16, 5),
        "action.gripper": np.arange(16, dtype=np.float32).reshape(1, 16, 1),
    }
    steps = getattr(_policy(action_mapping=SO101_ACTION_MAPPING), unpack)(chunk)

    assert len(steps) == 16
    assert steps[0] == {"1": 0.0, "2": 1.0, "3": 2.0, "4": 3.0, "5": 4.0, "6": 0.0}
    assert steps[1] == {"1": 5.0, "2": 6.0, "3": 7.0, "4": 8.0, "5": 9.0, "6": 1.0}


@pytest.mark.parametrize(
    ("mapping", "at_construction", "names"),
    [
        ({"1": "state.single_arm", "2": "state.single_arm"}, False, "only one of them can be sent"),
        ({"1": "state.single_arm[0]", "3": "state.single_arm[2]"}, False, r"leave \[1\] unfilled"),
        ({"1": "state.single_arm[0]", "2": "state.single_arm[0]"}, False, "is named twice"),
        ({"1": "state.single_arm[0]", "2": "state.single_arm"}, False, "both as a whole vector and by slot"),
        ({"absent": "state.single_arm[0]"}, False, "refused rather than zero-filled"),
        ({"room": "state.single_arm[0]"}, False, "A slot carries one component"),
        ({"room": "video.room[0]"}, True, "names a slot of a video key"),
        ({"1": "state.single_arm[x]"}, True, "is not a model key or a slot of one"),
        ({"1": "state.single_arm[-1]"}, True, "is not a model key or a slot of one"),
    ],
)
def test_a_mapping_that_cannot_be_composed_is_refused_by_name(mapping, at_construction, names):
    with pytest.raises(ValueError, match=names):
        policy = _policy(observation_mapping=mapping)
        assert not at_construction, "expected the refusal at construction"
        policy._prepare_observation(_observation(), "pick up the cube")


def test_a_slot_the_chunk_does_not_carry_is_refused_with_its_width():
    policy = _policy(action_mapping={"action.gripper[3]": "6"})
    with pytest.raises(ValueError, match="the model emitted 1 column"):
        policy._unpack_service_actions({"action.gripper": np.zeros((1, 4, 1), np.float32)})


def test_an_absurd_slot_index_is_refused_without_an_allocation_its_own_size():
    """The slot index is caller input - ``run_policy`` forwards a ``policy_config``
    verbatim - and a gap refusal that materialised ``range(width)`` sized its
    allocation by the largest index named rather than by the mapping, so
    ``single_arm[10**12]`` exhausted memory on the inference path before the
    refusal could name it. The check is by count, and the message previews a
    few of the gaps rather than listing every one."""
    policy = _policy(observation_mapping={"1": "state.single_arm[0]", "2": "state.single_arm[1000000]"})

    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match=r"leave \[1, 2, 3, 4, 5, 6, 7, 8\] and 999991 more unfilled"):
            policy._prepare_observation(_observation(), "pick up the cube")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 1 << 20, f"a gap refusal allocated {peak} bytes for a mapping of two entries"
