# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``Cosmos3Policy.preflight`` refuses a configuration whose actions no actuator would receive.

Measured on main 9e4f0a3d0 with a fake RoboLab ``(16, 8)`` chunk and
``set_robot_state_keys(["joint1", ..., "joint7", "gripper"])``:
``sorted(actions[0]) == ['gripper', 'joint_0', ..., 'joint_6']``. The policy keys
actions by the DROID layout names, not by the robot's actuator names, so without
``robot=`` or ``action_mapping`` the sim dropped every command while
``run_policy`` reported it ran; nothing checked the two against each other.

The provider now overrides :meth:`Policy.preflight`, the hook every rollout
surface runs before ``create_policy``: when neither ``action_mapping`` nor
``robot`` is given and no name of the active layout is among the observation
keys, the configuration is refused before the server is dialled, naming both
remedies. The hook is local metadata only (the embodiment registry and the
keys it is handed); nothing here opens a socket.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.policies import policy_overrides_preflight, preflight_reason
from strands_robots.policies._state_keys import FLAT_STATE_KEY
from strands_robots.policies.cosmos3 import Cosmos3Policy
from strands_robots.policies.cosmos3.embodiments import get_embodiment, list_robot_action_mappings

PANDA_SIM = {f"joint{i}" for i in range(1, 8)} | {"finger_joint1"}
CAMERAS = {"observation/wrist_image_left", "observation/exterior_image_1_left", "observation/exterior_image_2_left"}
DROID_NAMES = {f"joint_{i}" for i in range(7)} | {"gripper"}


def _reason(keys: set[str], **config: Any) -> str | None:
    return preflight_reason("cosmos3", lambda: keys, **config)


class TestTheHookIsWired:
    def test_cosmos3_overrides_preflight(self) -> None:
        assert policy_overrides_preflight("cosmos3") is True


class TestTheMeasuredConfigurationIsRefused:
    def test_a_panda_without_a_mapping_is_refused_before_the_server(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            Cosmos3Policy.preflight(PANDA_SIM | CAMERAS, embodiment="droid")
        text = str(excinfo.value)
        assert "joint_0" in text and "gripper" in text
        assert "joint1" in text and "finger_joint1" in text
        assert "robot=<name>" in text
        for name in list_robot_action_mappings():
            assert name in text
        assert "action_mapping=" in text
        assert "observation/" not in text.split("observation key of this robot")[1].split(")")[0]

    def test_through_the_shared_preflight_reason(self) -> None:
        """The text the simulation and the physical arm answer as their envelope."""
        reason = _reason(PANDA_SIM | CAMERAS, embodiment="droid", host="127.0.0.1", port=8000)
        assert reason is not None
        assert "send_action would drop every command" in reason

    def test_the_default_embodiment_is_droid(self) -> None:
        assert _reason(PANDA_SIM) is not None
        assert get_embodiment("droid").name in str(_reason(PANDA_SIM))


class TestTheCallerWhoAnsweredIsNotAsked:
    def test_robot_sugar_passes(self) -> None:
        assert _reason(PANDA_SIM | CAMERAS, embodiment="droid", robot="panda") is None

    def test_an_explicit_action_mapping_passes(self) -> None:
        mapping = {f"joint_{i}": f"joint{i + 1}" for i in range(7)} | {"gripper": "finger_joint1"}
        assert _reason(PANDA_SIM | CAMERAS, embodiment="droid", action_mapping=mapping) is None

    def test_an_unknown_robot_is_left_to_the_constructor(self) -> None:
        """The hook passes; the constructor refuses the name in its own words."""
        assert _reason(PANDA_SIM, embodiment="droid", robot="not_a_robot") is None
        stand_in: Any = object()
        with pytest.raises(ValueError, match="Unknown robot 'not_a_robot'"):
            Cosmos3Policy(embodiment="droid", robot="not_a_robot", client=stand_in)


class TestARobotNamedLikeTheLayoutPasses:
    def test_layout_names_in_the_observation_pass(self) -> None:
        assert _reason(DROID_NAMES | CAMERAS, embodiment="droid") is None

    def test_one_shared_name_is_enough(self) -> None:
        """A partial overlap is a mapping question the rollout can still act on; only a total miss is refused."""
        assert _reason({"gripper", "joint1", "joint2"} | CAMERAS, embodiment="droid") is None

    def test_the_action_space_selects_the_layout(self) -> None:
        midtrain = set(get_embodiment("droid").action_layouts["midtrain"])
        assert _reason(midtrain | CAMERAS, embodiment="droid", action_space="midtrain") is None
        assert _reason(DROID_NAMES | CAMERAS, embodiment="droid", action_space="midtrain") is not None

    def test_the_diffusers_backend_reads_the_raw_layout(self) -> None:
        raw = set(get_embodiment("droid").raw_action_layout)
        assert _reason(raw | CAMERAS, embodiment="droid", backend="diffusers") is None
        assert _reason(PANDA_SIM | CAMERAS, embodiment="droid", backend="diffusers") is not None


class TestNothingToJudgeAgainstPasses:
    def test_an_empty_observation_passes(self) -> None:
        assert _reason(set(), embodiment="droid") is None

    def test_a_flat_state_only_observation_passes(self) -> None:
        assert _reason({FLAT_STATE_KEY}, embodiment="droid") is None

    def test_an_unknown_embodiment_is_left_to_the_constructor(self) -> None:
        assert _reason(PANDA_SIM, embodiment="not_an_embodiment") is None

    def test_an_unknown_action_space_is_left_to_the_constructor(self) -> None:
        assert _reason(PANDA_SIM, embodiment="droid", action_space="not_a_space") is None
