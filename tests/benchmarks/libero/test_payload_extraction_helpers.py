"""Unit tests for the LIBERO adapter's payload-extraction helpers.

These module-level helpers parse two independent payload shapes without
touching MuJoCo:

* ``_extract_init_targets`` walks a parsed BDDL goal/init tree and returns
  the first-argument body name of every leaf predicate - the "subject" body
  the adapter may jitter at episode start.
* ``_extract_position`` / ``_extract_pose`` pull Cartesian state out of a
  ``get_body_state`` status-dict payload, tolerating missing/malformed
  blocks by returning ``None`` rather than raising.
* ``_fmt_state_value`` renders heterogeneous state values into the rounded,
  greppable string form used by the ``STATE_LOG`` diagnostic stream.

All four are pure functions, so the tests assert outputs directly against
hand-built payloads (no sim, no model).
"""

from __future__ import annotations

import numpy as np

from strands_robots.benchmarks.libero.adapter import (
    _extract_init_targets,
    _extract_pose,
    _extract_position,
    _fmt_state_value,
)
from strands_robots.benchmarks.libero.bddl_parser import And, Not, Or, Pred


class TestExtractInitTargets:
    """``_extract_init_targets`` collects leaf-predicate subject bodies."""

    def test_predicate_returns_first_arg(self):
        targets = _extract_init_targets(Pred(name="on", args=("cube_1", "table_1")))
        assert targets == ["cube_1"]

    def test_predicate_without_args_returns_empty(self):
        assert _extract_init_targets(Pred(name="ready", args=())) == []

    def test_and_flattens_every_clause_in_order(self):
        node = And(clauses=(Pred("on", ("a", "table")), Pred("upright", ("b",))))
        assert _extract_init_targets(node) == ["a", "b"]

    def test_or_flattens_every_clause(self):
        node = Or(clauses=(Pred("on", ("d", "table")), Pred("on", ("e", "shelf"))))
        assert _extract_init_targets(node) == ["d", "e"]

    def test_not_recurses_into_wrapped_clause(self):
        assert _extract_init_targets(Not(clause=Pred("on", ("c", "table")))) == ["c"]

    def test_nested_combinators_are_traversed_depth_first(self):
        node = And(
            clauses=(
                Pred("on", ("cube", "table")),
                Or(clauses=(Pred("upright", ("bottle",)), Not(clause=Pred("open", ("drawer",))))),
            )
        )
        assert _extract_init_targets(node) == ["cube", "bottle", "drawer"]


class TestExtractPosition:
    """``_extract_position`` pulls a 3-vector from a status-dict payload."""

    def test_valid_position_block_returns_floats(self):
        payload = {"content": [{"json": {"position": [1, 2, 3]}}]}
        assert _extract_position(payload) == [1.0, 2.0, 3.0]

    def test_wrong_length_position_returns_none(self):
        assert _extract_position({"content": [{"json": {"position": [1, 2]}}]}) is None

    def test_non_numeric_component_returns_none(self):
        assert _extract_position({"content": [{"json": {"position": [1, "x", 3]}}]}) is None

    def test_missing_content_returns_none(self):
        assert _extract_position({}) is None

    def test_first_valid_block_wins(self):
        payload = {
            "content": [
                {"text": "ignored, not a json block"},
                {"json": {"position": [4, 5, 6]}},
            ]
        }
        assert _extract_position(payload) == [4.0, 5.0, 6.0]


class TestExtractPose:
    """``_extract_pose`` returns ``(position, quaternion_wxyz)`` or ``None``s."""

    def test_non_dict_state_returns_none_pair(self):
        assert _extract_pose(None) == (None, None)

    def test_non_success_status_returns_none_pair(self):
        assert _extract_pose({"status": "error", "content": []}) == (None, None)

    def test_position_and_quaternion_both_parsed(self):
        payload = {
            "status": "success",
            "content": [{"json": {"position": [1, 2, 3], "quaternion": [1, 0, 0, 0]}}],
        }
        assert _extract_pose(payload) == ([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0])

    def test_position_only_leaves_quaternion_none(self):
        payload = {"status": "success", "content": [{"json": {"position": [1, 2, 3]}}]}
        assert _extract_pose(payload) == ([1.0, 2.0, 3.0], None)

    def test_malformed_quaternion_length_is_dropped(self):
        payload = {
            "status": "success",
            "content": [{"json": {"position": [1, 2, 3], "quaternion": [1, 0, 0]}}],
        }
        assert _extract_pose(payload) == ([1.0, 2.0, 3.0], None)

    def test_non_dict_blocks_are_skipped(self):
        payload = {
            "status": "success",
            "content": ["not-a-dict", {"json": {"quaternion": [0, 1, 0, 0]}}],
        }
        assert _extract_pose(payload) == (None, [0.0, 1.0, 0.0, 0.0])


class TestFmtStateValue:
    """``_fmt_state_value`` renders rounded, greppable STATE_LOG strings."""

    def test_none_renders_literal_none(self):
        assert _fmt_state_value(None) == "None"

    def test_float_rounded_to_six_decimals(self):
        assert _fmt_state_value(0.123456789) == "0.123457"

    def test_int_rendered_as_float(self):
        assert _fmt_state_value(2) == "2.000000"

    def test_bool_is_not_treated_as_float(self):
        # bool is an int subclass; the helper must exclude it from the
        # numeric branch and fall through to repr.
        assert _fmt_state_value(True) == "True"

    def test_list_rounds_numeric_elements_preserving_non_numeric(self):
        assert _fmt_state_value([0.111111111, "a"]) == "[0.111111, 'a']"

    def test_ndarray_rounded_elementwise(self):
        assert _fmt_state_value(np.array([0.1234567, 0.2])) == "[0.123457, 0.2]"

    def test_unknown_type_falls_back_to_repr(self):
        assert _fmt_state_value({"k": 1}) == "{'k': 1}"
