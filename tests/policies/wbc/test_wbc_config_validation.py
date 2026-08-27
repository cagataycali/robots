"""Fail-fast validation and file-loading contracts for :class:`WBCConfig`.

WBC drives a walking humanoid, so a config paired with the wrong checkpoint (a
bad dimension, a truncated per-joint vector) would destabilise the robot at
runtime. :meth:`WBCConfig.__post_init__` therefore rejects impossible
dimensions at construction rather than warn-and-continue, and
:meth:`WBCConfig.from_file` surfaces malformed/unsupported files as a loud
``ValueError`` instead of returning a half-built config. These tests pin those
contracts through the public surface (the constructor and ``from_file``) so a
regression that silently accepts a broken config is caught.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from strands_robots.policies.wbc import WBCConfig


class TestDimensionFailFast:
    """__post_init__ rejects sub-minimal dimensions with an actionable message.

    Widened from three fields to all five when the dimensions moved onto the
    shared count domain: ``command_dim`` and ``n_obs_joints`` were refused at
    zero only as a side effect of their own floor and relation, so neither had
    a cell of its own here. The message is the shared domain's, which names the
    field - a zero is not a small count, it is not a count at all.
    """

    @pytest.mark.parametrize(
        "dimension_name",
        ["num_actions", "obs_history_len", "single_obs_dim", "command_dim", "n_obs_joints"],
    )
    def test_a_dimension_of_zero_is_rejected(self, dimension_name: str) -> None:
        # Typed Any: the dataclass mixes field types, so a homogeneous splat is
        # an [arg-type] against whichever field mypy resolves the mapping to.
        override: dict[str, Any] = {dimension_name: 0}
        with pytest.raises(ValueError, match=rf"{dimension_name} must be a positive integer"):
            WBCConfig(policy_path="p.onnx", **override)


class TestFromFileErrorPaths:
    """from_file distinguishes not-found, malformed, and unsupported inputs."""

    def test_malformed_json_raises_value_error_naming_the_file(self, tmp_path: Path) -> None:
        bad = tmp_path / "wbc.json"
        bad.write_text("{not valid json")
        with pytest.raises(ValueError, match=r"is not valid JSON"):
            WBCConfig.from_file(bad)

    def test_unsupported_extension_is_rejected(self, tmp_path: Path) -> None:
        cfg = tmp_path / "wbc.txt"
        cfg.write_text("policy_path: p.onnx")
        with pytest.raises(ValueError, match=r"unsupported extension '\.txt'"):
            WBCConfig.from_file(cfg)

    def test_non_mapping_document_is_rejected(self, tmp_path: Path) -> None:
        # A JSON list parses fine but is not a config mapping; reject it loudly
        # rather than crash later on attribute access.
        cfg = tmp_path / "wbc.json"
        cfg.write_text(json.dumps(["not", "a", "mapping"]))
        with pytest.raises(ValueError, match=r"must contain a mapping"):
            WBCConfig.from_file(cfg)

    def test_valid_json_round_trips_through_from_file(self, tmp_path: Path) -> None:
        cfg = tmp_path / "wbc.json"
        cfg.write_text(json.dumps({"policy_path": "g1.onnx", "num_actions": 15}))
        loaded = WBCConfig.from_file(cfg)
        assert loaded.policy_path == "g1.onnx"
        assert loaded.num_actions == 15
