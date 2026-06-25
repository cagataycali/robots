"""Direct regression tests for the training LLM-input-safety gate.

``strands_robots.training._validate.validate_train_inputs`` is the single
source of input-safety truth for every training backend: each concrete
:class:`~strands_robots.training.base.Trainer` calls it (fail-closed) before a
``TrainSpec`` reaches backend config / argv-parity helpers. Per AGENTS.md >
Review Learnings (#92) > "LLM Input Safety", the path fields and the free-form
``extra`` dict are untrusted input that reaches backend internals, so this gate
is a security boundary.

Until now the gate was exercised only *indirectly* (through each backend's
``Trainer.validate``), and those orchestration tests skip/fail when an optional
backend dep (e.g. ``lerobot``) is absent. These tests pin each documented
attack vector directly against the dependency-free validator, so a regression
in the boundary surfaces immediately and independently of any backend.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.training import TrainSpec
from strands_robots.training._validate import validate_train_inputs


def _spec(tmp_path, **overrides) -> TrainSpec:
    """A clean, fully-valid TrainSpec with the given fields overridden.

    Defaults resolve to real tmp paths so the audited path check passes,
    letting each test isolate a single vector via ``overrides``.
    """
    base: dict[str, Any] = {
        "dataset_root": str(tmp_path / "data"),
        "base_model": "lerobot/act_aloha",
        "output_dir": str(tmp_path / "out"),
        "embodiment": "so101",
    }
    base.update(overrides)
    return TrainSpec(**base)


class TestCleanSpec:
    def test_clean_spec_has_no_problems(self, tmp_path):
        assert validate_train_inputs(_spec(tmp_path)) == []

    def test_optional_fields_unset_is_clean(self, tmp_path):
        # embodiment is optional (None) and must be skipped, not flagged.
        spec = _spec(tmp_path, embodiment=None, extra={})
        assert validate_train_inputs(spec) == []

    def test_dotted_extra_keys_are_clean(self, tmp_path):
        # The documented escape hatch: dotted/underscored lowercase keys.
        spec = _spec(
            tmp_path,
            extra={"policy_type": "act", "dataset.episodes": [0, 1], "model.x.y": 1},
        )
        assert validate_train_inputs(spec) == []


class TestPathFields:
    def test_null_byte_in_path_rejected(self, tmp_path):
        spec = _spec(tmp_path, dataset_root="/tmp/data\x00/evil")
        problems = validate_train_inputs(spec)
        assert len(problems) == 1
        assert "invalid characters" in problems[0]

    def test_traversal_in_path_rejected(self, tmp_path):
        spec = _spec(tmp_path, output_dir="../../etc")
        problems = validate_train_inputs(spec)
        assert len(problems) == 1
        assert "traversal" in problems[0]

    def test_protected_system_dir_rejected(self, tmp_path):
        spec = _spec(tmp_path, dataset_root="/etc/passwd")
        problems = validate_train_inputs(spec)
        assert len(problems) == 1
        assert "protected system directory" in problems[0]


class TestFlagBoundDashGuard:
    @pytest.mark.parametrize("field", ["base_model", "embodiment"])
    def test_leading_dash_rejected_on_non_path_fields(self, tmp_path, field):
        # base_model / embodiment are flag-bound but NOT path-checked, so a
        # leading dash is the only vector and must trip exactly one problem.
        spec = _spec(tmp_path, **{field: "--config_path=/etc/passwd"})
        problems = validate_train_inputs(spec)
        assert len(problems) == 1
        assert field in problems[0]
        assert "must not start with '-'" in problems[0]

    def test_path_field_leading_dash_is_dash_only(self, tmp_path):
        # output_dir is on BOTH the path list and the flag list. A safe value
        # that merely starts with '-' clears the path check (no null, no '..',
        # not a protected dir) and trips ONLY the dash check.
        spec = _spec(tmp_path, output_dir="-o")
        problems = validate_train_inputs(spec)
        assert len(problems) == 1
        assert "output_dir" in problems[0]
        assert "must not start with '-'" in problems[0]


class TestExtraKeyAllowlist:
    @pytest.mark.parametrize(
        "key",
        ["policy_type", "dataset.episodes", "model.x.y", "a", "lr_scheduler"],
    )
    def test_allowed_extra_keys(self, tmp_path, key):
        assert validate_train_inputs(_spec(tmp_path, extra={key: 1})) == []

    @pytest.mark.parametrize(
        "key",
        [
            "--steps",  # leading dash -> stray flag
            "Steps",  # uppercase
            "policy=act",  # embedded '='
            "my key",  # whitespace
            "1step",  # leading digit
            ".leading_dot",  # leading dot
            "",  # empty
        ],
    )
    def test_rejected_extra_keys(self, tmp_path, key):
        spec = _spec(tmp_path, extra={key: 1})
        problems = validate_train_inputs(spec)
        assert len(problems) == 1
        assert repr(key) in problems[0]
        assert "not allowed" in problems[0]


class TestAccumulation:
    def test_independent_problems_accumulate(self, tmp_path):
        # A bad path AND a bad extra key are independent checks; both report.
        spec = _spec(tmp_path, dataset_root="/etc/passwd", extra={"--bad": 1})
        problems = validate_train_inputs(spec)
        assert len(problems) == 2
        assert any("protected system directory" in p for p in problems)
        assert any("not allowed" in p for p in problems)
