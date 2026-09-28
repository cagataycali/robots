"""LayaPolicy: the state text, the primitive -> joint mapping and the gate, with a fake router.

No network, no model: the fake ``Router`` returns whatever answers the test
scripts, in the exact shape ``laya.Router.predict`` produces (``answers`` per
question with ``choice`` / ``score`` / ``noul``, ``probabilities``,
``confidence``, ``answer_confidence``). What is under test is everything the
provider does around the model.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from strands_robots.policies.laya import (
    HOLD,
    SO101_SIM_GRIPPER_RANGE_RAD,
    LayaPolicy,
    Primitive,
    apply_primitive,
    build_questions,
    decode_answers,
    labels_for_keys,
    serialize_state,
    size_from_score,
)

SIM_KEYS = ["1", "2", "3", "4", "5", "6"]
LABELS = labels_for_keys(SIM_KEYS)
ARM = tuple(label for label in LABELS.values() if label != "gripper")
REST = {"1": 0.0, "2": math.radians(-94.0), "3": math.radians(85.0), "4": math.radians(70.0), "5": 0.0, "6": 0.0}


def _choice(choice: str, options: tuple[str, ...], p: float = 0.6) -> dict[str, Any]:
    rest = (1.0 - p) / max(len(options) - 1, 1)
    probs = {o: (p if o == choice else rest) for o in options}
    return {"type": "choice", "choice": choice, "probabilities": probs, "confidence": p - rest, "answer_confidence": p}


def _score(value: float) -> dict[str, Any]:
    return {"type": "score", "score": value, "confidence": 0.4}


def _noul(p: float) -> dict[str, Any]:
    return {"type": "noul", "noul": p, "confidence": max(p, 1 - p)}


class FakeRouter:
    """Scripted stand-in for ``laya.Router``: records calls, replays answers."""

    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers
        self.calls: list[dict[str, Any]] = []

    def predict(
        self, state: Any, questions: dict[str, Any], model: str | None = None, max_len: int | None = None
    ) -> dict:
        self.calls.append({"state": state, "questions": questions, "model": model, "max_len": max_len})
        return {"model": model, "answers": {k: v for k, v in self.answers.items() if k in questions}, "routing": {}}


def _answers(
    joint: str, direction: str = "positive", score: float = 1.0, progress: float = 0.9, jaws: float = 0.1
) -> dict:
    return {
        "joint": _choice(joint, ARM + ("gripper", "none")),
        "direction": _choice(direction, ("positive", "negative")),
        "size": _score(score),
        "progress_ok": _noul(progress),
        "cube_in_jaws": _noul(jaws),
    }


# --- state text -----------------------------------------------------------------


def test_labels_for_sim_and_hardware_keys() -> None:
    assert LABELS == {
        "1": "shoulder_pan",
        "2": "shoulder_lift",
        "3": "elbow_flex",
        "4": "wrist_flex",
        "5": "wrist_roll",
        "6": "gripper",
    }
    hw = labels_for_keys(["shoulder_pan.pos", "gripper.pos"])
    assert hw == {"shoulder_pan.pos": "shoulder_pan", "gripper.pos": "gripper"}
    with pytest.raises(ValueError, match="not distinct"):
        labels_for_keys(["a", "b"], {"a": "same", "b": "same"})


def test_state_text_is_degrees_percent_and_metres_with_the_instruction() -> None:
    world = {"gripper_xyz_m": [0.05, -0.2, 0.15], "cube_xyz_m": [0.02, -0.32, 0.02], "contacts": ["cube:table"]}
    state = serialize_state(instruction="reach the cube", joints_rad=REST, labels=LABELS, gripper_pct=5.0, world=world)
    assert state["instruction"] == "reach the cube"
    assert state["joints_deg"] == {
        "shoulder_pan": 0.0,
        "shoulder_lift": -94.0,
        "elbow_flex": 85.0,
        "wrist_flex": 70.0,
        "wrist_roll": 0.0,
    }
    assert "gripper" not in state["joints_deg"]
    assert state["gripper_open_pct"] == 5.0
    assert state["gripper_to_cube_m"] == [-0.03, -0.12, -0.13]
    assert state["distance_to_cube_m"] == pytest.approx(0.1794, abs=1e-3)
    assert state["contacts"] == ["cube:table"]


def test_state_text_without_a_reader_says_so_instead_of_inventing_poses() -> None:
    state = serialize_state(instruction="x", joints_rad=REST, labels=LABELS, gripper_pct=None, world=None)
    assert "cube_xyz_m" not in state and "gripper_open_pct" not in state
    assert "not observed" in state["world"]


def test_questions_offer_every_arm_joint_plus_gripper_and_hold() -> None:
    qs = build_questions(ARM)
    assert list(qs) == ["joint", "direction", "size"]
    assert set(qs["joint"]["criteria"]) == set(ARM) | {"gripper", "none"}
    assert qs["size"]["criteria"] == ["small", "medium", "large"]
    gated = build_questions(ARM, "joint_direction_size_gated")
    assert {"progress_ok", "cube_in_jaws"} <= set(gated)
    with pytest.raises(ValueError, match="unknown profile"):
        build_questions(ARM, "nope")


# --- primitives -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "label"),
    [
        (-1.0, "small"),
        (0.0, "small"),
        (0.49, "small"),
        (0.6, "medium"),
        (1.4, "medium"),
        (1.6, "large"),
        (9.0, "large"),
    ],
)
def test_score_maps_to_the_nearest_size_label(score: float, label: str) -> None:
    assert size_from_score(score) == label


def test_score_refuses_nan() -> None:
    with pytest.raises(ValueError):
        size_from_score(float("nan"))


def test_arm_primitive_moves_one_joint_by_the_step_and_keeps_the_rest() -> None:
    action = apply_primitive(
        Primitive("elbow_flex", -1.0, "large"), REST, LABELS, gripper_range=SO101_SIM_GRIPPER_RANGE_RAD
    )
    assert set(action) == set(REST)
    assert action["3"] == pytest.approx(math.radians(75.0))
    for key in ("1", "2", "4", "5", "6"):
        assert action[key] == REST[key]
    assert all(type(v) is float for v in action.values())


def test_arm_primitive_is_clipped_to_the_ctrl_bound() -> None:
    action = apply_primitive(
        Primitive("shoulder_pan", 1.0, "large"),
        REST,
        LABELS,
        gripper_range=SO101_SIM_GRIPPER_RANGE_RAD,
        ctrl_bounds={"1": (-0.1, 0.1)},
    )
    assert action["1"] == pytest.approx(0.1)


def test_gripper_primitive_steps_in_percent_of_its_range() -> None:
    low, high = SO101_SIM_GRIPPER_RANGE_RAD
    closed = dict(REST, **{"6": low})
    opened = apply_primitive(
        Primitive("gripper", 1.0, "small"), closed, LABELS, gripper_range=(low, high), gripper_step_pct=20.0
    )
    assert opened["6"] == pytest.approx(low + 0.2 * (high - low))
    shut = apply_primitive(
        Primitive("gripper", -1.0, "small"), closed, LABELS, gripper_range=(low, high), gripper_step_pct=20.0
    )
    assert shut["6"] == pytest.approx(low)  # clipped at closed


def test_hold_returns_the_current_state() -> None:
    assert apply_primitive(HOLD, REST, LABELS, gripper_range=SO101_SIM_GRIPPER_RANGE_RAD) == REST


def test_unknown_joint_or_size_is_refused() -> None:
    with pytest.raises(ValueError, match="not one of the labelled"):
        apply_primitive(Primitive("knee", 1.0, "small"), REST, LABELS, gripper_range=SO101_SIM_GRIPPER_RANGE_RAD)
    with pytest.raises(ValueError, match="unknown size"):
        apply_primitive(Primitive("elbow_flex", 1.0, "huge"), REST, LABELS, gripper_range=SO101_SIM_GRIPPER_RANGE_RAD)


def test_decode_reads_the_primitive_and_the_gate_probabilities() -> None:
    primitive, conf = decode_answers(_answers("wrist_flex", "negative", 1.8, progress=0.25, jaws=0.7), ARM)
    assert primitive == Primitive("wrist_flex", -1.0, "large")
    assert conf["joint_p"] == pytest.approx(0.6)
    assert conf["progress_ok"] == pytest.approx(0.25)
    assert conf["cube_in_jaws"] == pytest.approx(0.7)
    with pytest.raises(ValueError, match="chose joint"):
        decode_answers(_answers("knee"), ARM)


# --- the policy -----------------------------------------------------------------


def _policy(answers: dict[str, Any], **cfg: Any) -> tuple[LayaPolicy, FakeRouter]:
    router = FakeRouter(answers)
    policy = LayaPolicy(router=router, **cfg)
    policy.set_robot_state_keys(SIM_KEYS)
    return policy, router


def test_policy_returns_one_action_per_tick_and_records_the_tick() -> None:
    policy, router = _policy(_answers("shoulder_lift", "positive", 1.0), model="typed-decisions")
    actions = policy.get_actions_sync(dict(REST), "reach the cube")
    assert len(actions) == 1
    assert actions[0]["2"] == pytest.approx(REST["2"] + math.radians(5.0))
    assert router.calls[0]["model"] == "typed-decisions"
    assert list(router.calls[0]["questions"]) == ["joint", "direction", "size"]
    assert router.calls[0]["state"]["joints_deg"]["shoulder_lift"] == -94.0
    tick = policy.last_tick
    assert tick is not None and tick["applied"] == {"joint": "shoulder_lift", "direction": 1.0, "size": "medium"}
    assert tick["gated"] is False and tick["latency_ms"] >= 0.0


def test_policy_feeds_the_world_reader_into_the_state() -> None:
    policy, router = _policy(_answers("none"))
    policy.set_world_reader(lambda: {"gripper_xyz_m": [0.0, 0.0, 0.1], "cube_xyz_m": [0.0, 0.0, 0.0]})
    policy.get_actions_sync(dict(REST), "reach")
    assert router.calls[0]["state"]["distance_to_cube_m"] == pytest.approx(0.1)


def test_policy_without_a_reader_serializes_joints_only(caplog: pytest.LogCaptureFixture) -> None:
    policy, router = _policy(_answers("none"))
    with caplog.at_level("WARNING"):
        policy.get_actions_sync(dict(REST), "reach")
        policy.get_actions_sync(dict(REST), "reach")
    assert "no world reader" in caplog.text
    assert caplog.text.count("no world reader") == 1
    assert "cube_xyz_m" not in router.calls[0]["state"]


def test_gate_holds_when_progress_probability_is_below_the_threshold() -> None:
    policy, _ = _policy(
        _answers("elbow_flex", progress=0.3), confidence_gate=0.5, questions_profile="joint_direction_size_gated"
    )
    actions = policy.get_actions_sync(dict(REST), "reach")
    assert actions[0] == REST
    assert policy.last_tick["gated"] is True
    assert policy.last_tick["primitive"]["joint"] == "elbow_flex"  # what Laya wanted is still recorded
    assert policy.last_tick["applied"] == HOLD.as_dict()


def test_gate_falls_back_to_the_joint_probability_without_a_progress_question() -> None:
    policy, _ = _policy(_answers("elbow_flex"), confidence_gate=0.7)  # joint_p is 0.6
    actions = policy.get_actions_sync(dict(REST), "reach")
    assert actions[0] == REST and policy.last_tick["gated"] is True
    policy.confidence_gate = 0.5
    assert policy.get_actions_sync(dict(REST), "reach")[0]["3"] != REST["3"]


def test_policy_declares_no_images_and_reads_the_instruction() -> None:
    policy, _ = _policy(_answers("none"))
    assert policy.requires_images is False
    assert LayaPolicy.reads_instruction is True
    assert policy.provider_name == "laya"
    assert "LayaPolicy(" in repr(policy)


def test_config_refusals() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        LayaPolicy(model="gpt", router=object())
    with pytest.raises(ValueError, match="confidence_gate"):
        LayaPolicy(confidence_gate=1.5, router=object())
    with pytest.raises(ValueError, match="missing size"):
        LayaPolicy(step_deg={"small": 1.0}, router=object())
    policy = LayaPolicy(router=object())
    with pytest.raises(ValueError):
        policy.set_robot_state_keys("123")  # a string is not a list of names
    with pytest.raises(TypeError):
        policy.set_world_reader(42)


def test_reset_clears_the_tick_record() -> None:
    policy, _ = _policy(_answers("none"))
    policy.get_actions_sync(dict(REST), "reach")
    policy.reset(seed=3)
    assert policy.last_tick is None and policy.tick_index == 0
