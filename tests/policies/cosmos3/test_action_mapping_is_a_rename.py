"""``action_mapping`` renames action columns; it cannot merge two of them.

:class:`~strands_robots.policies.cosmos3.policy.Cosmos3Policy` unpacks a
``[T, D]`` action chunk into one dict entry per column, keyed by the column's
name after the caller's ``action_mapping`` rename. A dict holds one value per
key, so two columns arriving at one actuator name collapse into a single entry:
the column written last wins and the other column's command is dropped. The
chunk still unpacks and the step dict is still well-formed, so the only visible
trace is an actuator that holds position while the model asked it to move.

The construction-time check on mapping *keys* exists for the neighbouring
mistake ("a typo'd rename can't silently emit a key the robot never consumes").
These cells hold the target side of the same dict to the same standard, and pin
that a rename which does not collide still delivers every column unchanged.
"""

import asyncio

import numpy as np
import pytest

from strands_robots.policies.cosmos3 import Cosmos3Policy
from strands_robots.policies.cosmos3.embodiments import ROBOT_ACTION_MAPPINGS, get_embodiment

# The released DROID joint_pos layout: 7 joints plus a gripper.
DROID_JOINT_POS = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
WIDTH = len(DROID_JOINT_POS)


class RecordingClient:
    """Records every ``infer`` call so a refusal can be shown to precede one."""

    def __init__(self, action: np.ndarray) -> None:
        self._action = action
        self.infer_calls = 0

    def infer(self, observation: dict) -> dict:
        self.infer_calls += 1
        return {"action": self._action, "server_timing": {}}

    def reset(self) -> None:
        pass

    def get_server_metadata(self) -> dict:
        return {}


def _chunk() -> np.ndarray:
    """A chunk whose value states its own column, so a swap is readable."""
    return np.tile(np.arange(WIDTH, dtype=np.float32), (4, 1))


def _observation() -> dict:
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    obs: dict = {
        key: image
        for key in (
            "observation/wrist_image_left",
            "observation/exterior_image_1_left",
            "observation/exterior_image_2_left",
        )
    }
    for i in range(7):
        obs[f"joint_{i}"] = float(i) * 0.1
    obs["gripper"] = 0.5
    return obs


def _rollout(mapping: dict[str, str] | None) -> tuple[list[dict], RecordingClient]:
    client = RecordingClient(_chunk())
    policy = Cosmos3Policy(
        embodiment="droid",
        # A recording stand-in for the WebSocket client (dependency injection).
        client=client,  # type: ignore[arg-type]
        action_mapping=mapping,
    )
    policy.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])
    return asyncio.run(policy.get_actions(_observation(), "pick up the cube")), client


# (mapping, the columns that would arrive together, the name they collide on)
COLLIDING = {
    "two entries share one target": ({"joint_0": "j6", "joint_1": "j6"}, ["joint_0", "joint_1"], "j6"),
    "one entry onto a sibling column's own name": ({"joint_0": "joint_1"}, ["joint_0", "joint_1"], "joint_1"),
    "one entry onto the gripper column's name": ({"joint_6": "gripper"}, ["joint_6", "gripper"], "gripper"),
    # The reverse of the row above, and the sharper one: the surviving entry is
    # keyed on a JOINT and carries the GRIPPER's value, so the joint is
    # mis-commanded rather than merely uncommanded.
    "the gripper column onto a joint's name": ({"gripper": "joint_6"}, ["joint_6", "gripper"], "joint_6"),
    "a whole layout folded onto one actuator": (
        dict.fromkeys(DROID_JOINT_POS, "everything"),
        DROID_JOINT_POS,
        "everything",
    ),
}

NOT_COLLIDING = {
    "no mapping at all": None,
    "distinct fresh targets": {"joint_0": "shoulder_pan", "gripper": "grip"},
    # A shift of *every* column is a bijection, so it is a legitimate rename
    # even though each individual target is another column's source name.
    "a shift of every column": {f"joint_{i}": f"joint_{i + 1}" for i in range(7)},
}


class TestACollidingRenameIsRefused:
    """A mapping that would merge two columns is refused, naming both."""

    @pytest.mark.parametrize(("mapping", "columns", "target"), COLLIDING.values(), ids=list(COLLIDING))
    def test_construction_refuses_and_names_both_columns_and_the_target(self, mapping, columns, target):
        with pytest.raises(ValueError, match="action_mapping is not a rename") as excinfo:
            Cosmos3Policy(embodiment="droid", client=RecordingClient(_chunk()), action_mapping=mapping)
        message = str(excinfo.value)
        for column in columns:
            assert column in message, f"the refusal does not name the colliding column {column!r}"
        assert repr(target) in message, "the refusal does not name the target the columns collide on"

    def test_the_refusal_precedes_any_inference(self):
        """Refused client-side, before a request reaches the server."""
        client = RecordingClient(_chunk())
        with pytest.raises(ValueError, match="action_mapping is not a rename"):
            Cosmos3Policy(embodiment="droid", client=client, action_mapping={"joint_6": "gripper"})
        assert client.infer_calls == 0


class TestARenameThatDoesNotCollideDeliversEveryColumn:
    """The control: an injective rename is untouched by the refusal."""

    @pytest.mark.parametrize("mapping", NOT_COLLIDING.values(), ids=list(NOT_COLLIDING))
    def test_every_column_of_the_chunk_reaches_its_own_key(self, mapping):
        steps, client = _rollout(mapping)
        assert client.infer_calls == 1
        for step in steps:
            assert len(step) == WIDTH, f"a {WIDTH}-column chunk unpacked to {len(step)} keys: {sorted(step)}"
            expected = {
                (mapping or {}).get(column, column): float(index) for index, column in enumerate(DROID_JOINT_POS)
            }
            assert step == expected

    def test_every_built_in_robot_mapping_is_injective(self):
        """Derived floor: a built-in mapping added later is graded on arrival."""
        assert ROBOT_ACTION_MAPPINGS, "no built-in mappings to grade"
        for robot in ROBOT_ACTION_MAPPINGS:
            steps, _ = _rollout(dict(ROBOT_ACTION_MAPPINGS[robot]))
            assert len(steps[0]) == WIDTH, f"the built-in {robot!r} mapping merges columns: {sorted(steps[0])}"


def test_the_droid_layout_this_module_reasons_about_is_the_shipped_one():
    """Anchor the hand-written layout to the registry it stands in for."""
    assert get_embodiment("droid").action_layouts["joint_pos"] == DROID_JOINT_POS
