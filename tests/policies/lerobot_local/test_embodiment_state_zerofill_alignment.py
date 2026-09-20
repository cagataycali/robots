# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Declarative-embodiment state build must zero-fill a missing state_key IN PLACE.

This is the embodiment-path sibling of the generic-path fix pinned in
``test_partial_state_key_alignment.py``. ``PackStateProcessorStep.observation()``
(the step a declarative embodiment installs on the preprocessor) previously
appended a value only for state_keys *found* in the observation and skipped the
absent ones, so a missing key DROPPED its slot and shifted every following joint
up one index before the trailing pad -- the model received a garbage
``observation.state`` while the run reported success.

The canonical trigger is the aloha bimanual embodiment: its 14 actuators follow
the gym-aloha / LeRobot ACT convention ``[6 arm + 1 gripper] x 2`` with the
gripper ACTUATORS ``left/gripper`` / ``right/gripper`` at indices 6 and 13, but
the sim observation exposes the finger JOINTS (``left/left_finger`` ...), never a
``*/gripper`` value. With the old skip logic the right-arm joints slid into the
left gripper slot; and because the shipped config declared the 16 finger-JOINT
names (not the 14 actuators), a canonical 14-D ACT crashed outright
(``observation.state dim 16 > model expected 14``).

These tests pin the corrected behaviour: a missing state_key is zero-filled IN
PLACE (present joints keep their model index), the degradation is warned once,
a fully-present key set is unchanged, an all-missing set is left untouched for a
clearer downstream error, and the aloha embodiment declares the 14 actuator keys.

The warning was the ONLY report: the step packs inside LeRobot's pipeline and had
no route back to the policy, so ``missing_state_keys_used`` - the flag
``run_policy`` reports and a collection loop gates on - read False for every
embodiment-driven run, and ``strict_keys=True`` packed the zero anyway. Measured
on this same trigger with the real checkpoint ``lerobot/act_aloha_sim_transfer_
cube_human``: two of fourteen dims carrying no reading under a ``success``
envelope whose binding flags all read healthy. :class:`TestTheZeroFillIsReported`
pins the two surfaces the generic path already honours.
"""

import numpy as np
import pytest

pytest.importorskip("lerobot")

import strands_robots.policies.lerobot_local.embodiment as E

# The aloha bimanual actuator convention: 6 arm + 1 gripper actuator per side,
# gripper actuators interspersed at indices 6 and 13.
ALOHA_14 = [
    "left/waist",
    "left/shoulder",
    "left/elbow",
    "left/forearm_roll",
    "left/wrist_angle",
    "left/wrist_rotate",
    "left/gripper",
    "right/waist",
    "right/shoulder",
    "right/elbow",
    "right/forearm_roll",
    "right/wrist_angle",
    "right/wrist_rotate",
    "right/gripper",
]
LEFT_ARM = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
RIGHT_ARM = [108.0, 109.0, 110.0, 111.0, 112.0, 113.0]
ARM_KEYS_L = ALOHA_14[0:6]
ARM_KEYS_R = ALOHA_14[7:13]


def _sim_obs_no_gripper() -> dict[str, float]:
    """A sim observation exposing the arm joints + finger joints but NOT the
    gripper actuator keys ``left/gripper`` / ``right/gripper`` (mirrors what the
    real MuJoCo aloha's ``get_observation`` returns)."""
    obs: dict[str, float] = {}
    for k, v in zip(ARM_KEYS_L, LEFT_ARM):
        obs[k] = v
    for k, v in zip(ARM_KEYS_R, RIGHT_ARM):
        obs[k] = v
    # finger joints present but NOT declared as state_keys (so they are ignored):
    obs["left/left_finger"] = 0.02
    obs["left/right_finger"] = 0.02
    obs["right/left_finger"] = 0.02
    obs["right/right_finger"] = 0.02
    return obs


def _step(state_keys, expected_dim, **kw):
    Step = E.register_pack_state_step()
    assert Step is not None, "lerobot processor framework unavailable"
    return Step(state_keys=list(state_keys), expected_dim=expected_dim, dim_policy=kw.pop("dim_policy", "pad"), **kw)


class TestPackStateZeroFillInPlace:
    def test_missing_gripper_keys_zero_filled_in_place(self):
        """Arm joints keep their canonical model index; the two absent gripper
        actuator slots read 0.0. FAILS pre-fix: the right arm slid into the
        left-gripper slot and both zeros landed at the tail."""
        E._WARNED_STATE_KEY_MISMATCH.clear()
        out = _step(ALOHA_14, 14).observation(_sim_obs_no_gripper())
        state = out["observation.state"].numpy()
        assert len(state) == 14
        np.testing.assert_allclose(state[0:6], LEFT_ARM, atol=1e-5)
        assert state[6] == 0.0  # left/gripper slot held in place
        np.testing.assert_allclose(state[7:13], RIGHT_ARM, atol=1e-5)  # <- shifted pre-fix
        assert state[13] == 0.0  # right/gripper slot held in place

    def test_missing_keys_warn_once(self, caplog):
        """The degradation is surfaced once (naming the absent keys), then
        deduplicated across the hot control loop."""
        E._WARNED_STATE_KEY_MISMATCH.clear()
        step = _step(ALOHA_14, 14)
        import logging

        with caplog.at_level(logging.WARNING, logger="strands_robots.policies.lerobot_local.embodiment"):
            step.observation(_sim_obs_no_gripper())
            step.observation(_sim_obs_no_gripper())
        # Selected by the missing-key report's stable subject rather than by a
        # phrase: the wording is shared with the generic robot_state_keys path,
        # so it belongs to that message's contract and not to this test's.
        warns = [r for r in caplog.records if "declared state_keys" in r.getMessage()]
        assert len(warns) == 1
        assert "left/gripper" in warns[0].getMessage() and "right/gripper" in warns[0].getMessage()

    def test_all_present_unchanged(self):
        """A fully-present key set is packed verbatim with no zero-fill (the
        so101-style path); this must not regress."""
        E._WARNED_STATE_KEY_MISMATCH.clear()
        keys = ["a", "b", "c"]
        out = _step(keys, 3).observation({"a": 1.0, "b": 2.0, "c": 3.0})
        state = out["observation.state"].numpy()
        np.testing.assert_allclose(state, [1.0, 2.0, 3.0], atol=1e-5)
        assert not E._WARNED_STATE_KEY_MISMATCH  # no missing -> no warn recorded

    def test_all_missing_passthrough(self):
        """When NONE of the declared keys are present, leave the observation
        untouched so a clearer downstream error can fire (do not emit all-zero)."""
        E._WARNED_STATE_KEY_MISMATCH.clear()
        obs = {"unrelated": 5.0}
        out = _step(["a", "b"], 2).observation(dict(obs))
        assert "observation.state" not in out
        assert out == obs


class TestTheZeroFillIsReported:
    """The caller's flag and the caller's strict posture see what the step filled."""

    @staticmethod
    def _bridge(state_keys, expected_dim, **applied):
        """A bridge carrying only the injected pack-state step, plus that step.

        ``applied`` goes to ``apply_embodiment``, so a case that does not depend
        on the strict posture passes nothing and reads the default.
        """
        from types import SimpleNamespace

        from strands_robots.policies.lerobot_local.processor import ProcessorBridge

        bridge = ProcessorBridge(preprocessor=SimpleNamespace(steps=[]))
        bridge.apply_embodiment(
            E.EmbodimentMap(name="pack_test", state_keys=list(state_keys), dim_policy="pad"),
            input_features={
                "observation.state": SimpleNamespace(type=SimpleNamespace(name="STATE"), shape=(expected_dim,))
            },
            **applied,
        )
        packers = [s for s in bridge._preprocessor.steps if getattr(s, "_registry_name", None) == "strands_pack_state"]
        assert len(packers) == 1, bridge._preprocessor.steps
        return bridge, packers[0]

    @staticmethod
    def _policy():
        """A policy with no model loaded, read only for its telemetry flag."""
        from unittest.mock import patch

        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

        with patch.object(LerobotLocalPolicy, "_load_model"):
            return LerobotLocalPolicy(pretrained_name_or_path=None, policy_type="act")

    def test_the_policy_flag_reports_the_filled_dims(self):
        """FAILS pre-fix: the flag read False while two dims carried no reading."""
        E._WARNED_STATE_KEY_MISMATCH.clear()
        bridge, step = self._bridge(ALOHA_14, 14)
        step.observation(_sim_obs_no_gripper())
        policy = self._policy()
        policy._processor_bridge = bridge
        assert policy.missing_state_keys_used is True
        assert bridge.state_missing_keys == ("left/gripper", "right/gripper")

    def test_strict_keys_refuses_instead_of_filling(self):
        """strict_keys is one posture on every state path, not two."""
        E._WARNED_STATE_KEY_MISMATCH.clear()
        _bridge, step = self._bridge(ALOHA_14, 14, strict_keys=True)
        with pytest.raises(ValueError) as exc:
            step.observation(_sim_obs_no_gripper())
        msg = str(exc.value)
        assert "strict_keys=True" in msg
        assert "left/gripper" in msg and "right/gripper" in msg
        assert "set_robot_state_keys" in msg

    def test_a_fully_bound_observation_packs_and_reports_nothing(self):
        """The control: strict_keys set, every declared key present -> no raise, flag clear."""
        E._WARNED_STATE_KEY_MISMATCH.clear()
        keys = ["a", "b", "c"]
        bridge, step = self._bridge(keys, 3, strict_keys=True)
        out = step.observation({"a": 1.0, "b": 2.0, "c": 3.0})
        np.testing.assert_allclose(out["observation.state"].numpy(), [1.0, 2.0, 3.0], atol=1e-5)
        assert bridge.state_missing_keys == ()
        policy = self._policy()
        policy._processor_bridge = bridge
        assert policy.missing_state_keys_used is False


class TestAlohaEmbodimentActuatorConvention:
    def test_aloha_declares_14_actuator_keys(self):
        """The shipped aloha embodiment uses the 14 actuator convention (matching
        the model's actuators / robot_action_keys), not the 16 finger-JOINT
        names that crash/mis-align a canonical 14-D ACT."""
        emb = E.load_embodiment("aloha")
        assert emb.state_keys == ALOHA_14
        assert emb.action_keys == ALOHA_14
        # the old 16-finger-joint layout is gone
        assert "left/left_finger" not in emb.state_keys
        assert "left/left_finger" not in emb.action_keys

    def test_aloha_state_build_is_canonically_aligned(self):
        """End-to-end: build observation.state from a gripper-less sim obs through
        the real aloha embodiment config; arm stays aligned, grippers zero-filled."""
        E._WARNED_STATE_KEY_MISMATCH.clear()
        emb = E.load_embodiment("aloha")
        step = _step(emb.state_keys, 14, dim_policy=emb.dim_policy)
        state = step.observation(_sim_obs_no_gripper())["observation.state"].numpy()
        expected = LEFT_ARM + [0.0] + RIGHT_ARM + [0.0]
        np.testing.assert_allclose(state, expected, atol=1e-5)
