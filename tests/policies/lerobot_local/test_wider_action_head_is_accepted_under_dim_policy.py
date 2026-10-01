# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A wider action head drives an embodiment's leading columns under ``dim_policy`` pad/truncate, judged before the download.

Measured on main 9e4f0a3d0 with ``lerobot/pi0_base`` (a 32-wide padded action
head shipped for every embodiment) and a native six-key embodiment with
``dim_policy="pad"``: ``create_policy`` took 126.41 s, then
``EmbodimentMap.validate`` refused ``6 action_keys but model action dim is 32``
whatever ``dim_policy`` said, and the pipeline was discarded for the raw
obs/action fallback. The state side already followed ``dim_policy`` (``pad``
widens a six-value state to 32); the action side did not, although
``align_action_values`` already consumes a long vector by its leading columns.

Now ``EmbodimentMap.action_dim_error`` is the one rule: ``strict`` wants the
exact width; ``pad`` / ``truncate`` accept a wider head (leading columns drive
the actuators) and still refuse a narrower one (no policy widens a head).
``validate`` reads it after the load, and ``LerobotLocalPolicy.preflight`` reads
it before the download from the width ``config.json`` declares
(``declared_action_dim``), so a too-narrow head is refused in the envelope
before any weights move. The Hub is not dialled here: the width reader is
replaced by one that answers what pi0_base's config.json answers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import strands_robots.policies.lerobot_local.policy as policy_mod
from strands_robots.policies import preflight_reason
from strands_robots.policies.lerobot_local.embodiment import EmbodimentMap
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy
from strands_robots.policies.lerobot_local.resolution import declared_action_dim

SIX = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class _Feat:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


def _features(action_dim: int) -> tuple[dict[str, Any], dict[str, Any]]:
    return {"observation.state": _Feat((32,))}, {"action": _Feat((action_dim,))}


def _embodiment(dim_policy: str) -> EmbodimentMap:
    return EmbodimentMap(name=f"six_{dim_policy}", state_keys=SIX, action_keys=SIX, dim_policy=dim_policy)


class TestTheRule:
    @pytest.mark.parametrize("dim_policy", ["strict", "pad", "truncate"])
    def test_an_exact_head_passes(self, dim_policy: str) -> None:
        assert _embodiment(dim_policy).action_dim_error(6) is None

    @pytest.mark.parametrize("dim_policy", ["pad", "truncate"])
    def test_a_wider_head_is_accepted_when_adaptation_is_opted_in(self, dim_policy: str) -> None:
        assert _embodiment(dim_policy).action_dim_error(32) is None

    def test_strict_still_refuses_a_wider_head_and_names_the_way_in(self) -> None:
        reason = _embodiment("strict").action_dim_error(32)
        assert reason is not None
        assert "6 action_keys but model action dim is 32" in reason
        assert "dim_policy='pad' or 'truncate'" in reason

    @pytest.mark.parametrize("dim_policy", ["strict", "pad", "truncate"])
    def test_a_narrower_head_is_refused_under_every_policy(self, dim_policy: str) -> None:
        reason = _embodiment(dim_policy).action_dim_error(4)
        assert reason is not None
        assert "6 action_keys but model action dim is 4" in reason

    def test_no_action_keys_means_nothing_to_judge(self) -> None:
        assert EmbodimentMap(name="stateless", state_keys=SIX).action_dim_error(32) is None


class TestValidateReadsTheRule:
    def test_pi0_base_shape_passes_under_pad(self) -> None:
        inp, out = _features(32)
        _embodiment("pad").validate(inp, out)

    def test_a_wider_head_is_refused_under_strict(self) -> None:
        """With the state side exact, so the action verdict is the one reported."""
        inp, out = {"observation.state": _Feat((6,))}, {"action": _Feat((32,))}
        with pytest.raises(ValueError, match="6 action_keys but model action dim is 32"):
            _embodiment("strict").validate(inp, out)

    def test_a_narrow_head_is_refused_under_pad(self) -> None:
        inp, out = _features(4)
        with pytest.raises(ValueError, match="narrower than the actuators"):
            _embodiment("pad").validate(inp, out)


class TestTheWidthIsReadFromConfigJson:
    def test_a_local_checkpoint_directory_declares_its_width(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text(
            json.dumps({"type": "pi0", "output_features": {"action": {"type": "ACTION", "shape": [32]}}}),
            encoding="utf-8",
        )
        assert declared_action_dim(str(tmp_path)) == 32

    @pytest.mark.parametrize(
        "config",
        [
            {"type": "pi0"},
            {"output_features": "not a dict"},
            {"output_features": {"action": {"shape": []}}},
            {"output_features": {"action": {"shape": ["32"]}}},
            {"output_features": {"action": {"shape": [0]}}},
            {"output_features": {"action": {"shape": [True]}}},
        ],
        ids=["no features", "features not a dict", "empty shape", "string width", "zero width", "bool width"],
    )
    def test_an_unreadable_width_is_unknown_not_a_number(self, tmp_path: Path, config: dict[str, Any]) -> None:
        (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        assert declared_action_dim(str(tmp_path)) is None


class TestPreflightJudgesTheWidthBeforeTheDownload:
    """What the rollout surfaces run first, fed the width the Hub's config.json declares."""

    def test_pi0_base_with_a_six_key_pad_embodiment_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(policy_mod, "declared_action_dim", lambda ref, rev=None: 32)
        monkeypatch.setattr(policy_mod, "declared_image_features", lambda ref, rev=None: None)
        LerobotLocalPolicy.preflight(
            set(SIX) | {"front"}, pretrained_name_or_path="lerobot/pi0_base", embodiment=_embodiment("pad")
        )

    def test_a_too_narrow_head_is_refused_before_the_download(self, monkeypatch: pytest.MonkeyPatch) -> None:
        asked: list[tuple[str, Any]] = []

        def width(ref: str, rev: Any = None) -> int:
            asked.append((ref, rev))
            return 4

        monkeypatch.setattr(policy_mod, "declared_action_dim", width)
        with pytest.raises(ValueError, match="6 action_keys but model action dim is 4"):
            LerobotLocalPolicy.preflight(
                set(SIX), pretrained_name_or_path="acme/four-wide", revision="v2", embodiment=_embodiment("pad")
            )
        assert asked == [("acme/four-wide", "v2")]

    def test_strict_refuses_the_wider_head_before_the_download(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(policy_mod, "declared_action_dim", lambda ref, rev=None: 32)
        with pytest.raises(ValueError, match="dim_policy='pad' or 'truncate'"):
            LerobotLocalPolicy.preflight(
                set(SIX), pretrained_name_or_path="lerobot/pi0_base", embodiment=_embodiment("strict")
            )

    def test_an_unknown_width_is_left_to_validate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(policy_mod, "declared_action_dim", lambda ref, rev=None: None)
        monkeypatch.setattr(policy_mod, "declared_image_features", lambda ref, rev=None: None)
        LerobotLocalPolicy.preflight(set(SIX), pretrained_name_or_path="acme/no-config", embodiment=_embodiment("pad"))

    def test_no_checkpoint_reference_reads_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def never(ref: str, rev: Any = None) -> int:
            raise AssertionError("no reference, no read")

        monkeypatch.setattr(policy_mod, "declared_action_dim", never)
        LerobotLocalPolicy.preflight(set(SIX), embodiment=_embodiment("pad"))

    def test_through_the_shared_preflight_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The text the simulation and the physical arm answer as their envelope."""
        monkeypatch.setattr(policy_mod, "declared_action_dim", lambda ref, rev=None: 4)
        reason = preflight_reason(
            "lerobot_local", lambda: set(SIX), pretrained_name_or_path="acme/four-wide", embodiment=_embodiment("pad")
        )
        assert reason is not None
        assert "6 action_keys but model action dim is 4" in reason
