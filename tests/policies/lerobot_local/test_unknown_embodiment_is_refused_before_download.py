# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An ``embodiment=`` the registry does not know is refused before any weights move.

Measured on main 40eec5bee with ``run_policy(..., policy_config={"embodiment":
"so102", ...})``: ``[18.00s] RAISED RuntimeError: Failed to load embodiment
'so102': Unknown embodiment 'so102'. Available: [...]`` (2.13 s with the
checkpoint cached), while a misspelt keyword on the same call was refused in
0.09 s as a ``status=error`` envelope. The embodiment was resolved by
``_configure_embodiment``, which runs after ``_load_model``, and
``LerobotLocalPolicy.preflight`` (the rollout surfaces' pre-download hook)
swallowed the resolution failure "for create_policy to report".

Now one rule, :func:`embodiment_spec_error`, is read by both ``preflight`` (so
the rollout surfaces answer the envelope before the download) and the
constructor (so a caller who builds the policy directly is refused before
``_load_model``). The words are ``load_embodiment``'s own: the unknown name and
the registered names.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from strands_robots.policies import preflight_reason
from strands_robots.policies.lerobot_local.embodiment import EMBODIMENT_MAP, EmbodimentMap, load_embodiment
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy, embodiment_spec_error

_UNKNOWN = "so102"


def _keys() -> set[str]:
    return {"1", "2", "3", "4", "5", "6", "front", "wrist"}


class TestTheRule:
    def test_none_is_not_a_spec(self) -> None:
        assert embodiment_spec_error(None) is None

    def test_a_registered_name_resolves(self) -> None:
        for name in EMBODIMENT_MAP:
            assert embodiment_spec_error(name) is None, name

    def test_an_instance_resolves(self) -> None:
        assert embodiment_spec_error(load_embodiment(next(iter(EMBODIMENT_MAP)))) is None

    def test_an_unknown_name_names_the_registered_ones(self) -> None:
        reason = embodiment_spec_error(_UNKNOWN)
        assert reason is not None
        assert reason.startswith("lerobot_local: Unknown embodiment 'so102'.")
        for name in EMBODIMENT_MAP:
            assert name in reason

    def test_a_wrong_type_is_refused(self) -> None:
        reason = embodiment_spec_error(42)
        assert reason is not None
        assert "embodiment must be str | dict | EmbodimentMap" in reason

    def test_a_dict_with_a_field_the_map_does_not_declare_is_refused(self) -> None:
        reason = embodiment_spec_error({"state_keys": ["1"], "no_such_field": 1})
        assert reason is not None
        assert "no_such_field" in reason

    def test_a_dict_the_map_accepts_resolves(self) -> None:
        fields = {"obs_rename": {}, "state_keys": ["1", "2"], "action_keys": ["1", "2"]}
        assert isinstance(load_embodiment(fields), EmbodimentMap)
        assert embodiment_spec_error(fields) is None


class TestPreflight:
    """The hook the rollout surfaces run before the download refuses the spec itself."""

    def test_an_unknown_embodiment_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown embodiment 'so102'") as excinfo:
            LerobotLocalPolicy.preflight(_keys(), embodiment=_UNKNOWN)
        assert "Available:" in str(excinfo.value)

    def test_through_the_shared_preflight_reason(self) -> None:
        """What ``SimEngine._preflight_policy_config`` and the physical arm read."""
        reads = 0

        def read_keys() -> set[str]:
            nonlocal reads
            reads += 1
            return _keys()

        reason = preflight_reason(
            "lerobot_local", read_keys, pretrained_name_or_path="robotfuel/act_so101_t16b", embodiment=_UNKNOWN
        )
        assert reason is not None
        assert "Unknown embodiment 'so102'" in reason
        assert reads == 1

    def test_a_known_embodiment_still_passes_to_the_camera_check(self) -> None:
        """The spec check is a gate in front of the existing routing check, not a replacement."""
        LerobotLocalPolicy.preflight(_keys(), embodiment="so101")


class TestTheConstructor:
    def test_an_unknown_embodiment_is_refused_before_load_model(self) -> None:
        with patch.object(LerobotLocalPolicy, "_load_model") as load_model:
            with pytest.raises(ValueError, match="Unknown embodiment 'so102'"):
                LerobotLocalPolicy(pretrained_name_or_path="robotfuel/act_so101_t16b", embodiment=_UNKNOWN)
        load_model.assert_not_called()

    def test_a_known_embodiment_reaches_load_model(self) -> None:
        with patch.object(LerobotLocalPolicy, "_load_model") as load_model:
            LerobotLocalPolicy(pretrained_name_or_path="robotfuel/act_so101_t16b", embodiment="so101")
        load_model.assert_called_once()


class TestTheRolloutSurface:
    """The measured call, on a live so101 scene: an envelope, no download."""

    @pytest.fixture
    def sim(self):
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

        engine = MuJoCoSimEngine()
        engine.create_world()
        engine.add_robot("so101")
        yield engine
        engine.cleanup()

    def test_run_policy_answers_the_envelope_without_loading(self, sim: Any) -> None:
        config = {"pretrained_name_or_path": "robotfuel/act_so101_t16b", "embodiment": _UNKNOWN}
        with patch.object(LerobotLocalPolicy, "_load_model") as load_model:
            result = sim.run_policy(
                robot_name="so101", policy_provider="lerobot_local", policy_config=config, duration=0.2
            )
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "Unknown embodiment 'so102'" in text
        assert "Available:" in text
        load_model.assert_not_called()
