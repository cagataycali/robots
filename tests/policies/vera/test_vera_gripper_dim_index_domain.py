"""``gripper_dim_index`` selects one column of the action chunk, so it takes a domain.

:func:`~strands_robots.policies.vera.sim_ik.decode_vera_delta_chunk_to_targets`
splits the gripper column off the chunk before IK with two statements::

    gidx = gripper_dim_index if gripper_dim_index >= 0 else D - 1
    gripper = action_chunk[:, gidx].copy()
    pose_block = np.delete(action_chunk, gidx, axis=1)

That ``>= 0`` test is a *selector* as well as the only check: whatever fails it is
read as the ``-1`` "trailing column" sentinel. Its two siblings in the same
signature were already held to a domain - ``translation_scale`` in the function's
first statement, ``rotation_dim`` in its second - and the ``Raises`` section spells
out why ``rotation_dim`` in particular cannot be left to the arithmetic that
consumes it: "it indexes the rotation block of every step, so a non-integral or
non-numeric width raised ``TypeError: slice indices must be integers`` out of
that slice - naming neither the parameter nor this function". The gripper index
indexes a *column* of every step and did exactly that.

Measured on ``67ca3ac4``, one real decode per row over a 7-wide
``[trans(3), rot(3), gripper(1)]`` chunk whose columns are distinguishable, so
the column actually read is observable:

| ``gripper_dim_index=`` | outcome | gripper read from | cost |
| --- | --- | --- | --- |
| ``-1`` / ``6`` / ``0`` | accepted | col 6 / col 6 / col 0 | honored (unchanged) |
| ``-5`` / ``-99`` | **accepted** | col 6 | silently read as the sentinel |
| ``nan`` | **accepted** | col 6 | silently read as the sentinel |
| ``2.7`` / ``6.0`` / ``inf`` | ``IndexError`` | - | ``only integers, slices ...`` |
| ``True`` | ``ValueError`` | - | ``boolean array argument obj to delete ...`` |
| ``99`` | ``IndexError`` | - | ``index 99 is out of bounds for axis 1`` |
| ``'6'`` / ``None`` / ``[6]`` | ``TypeError`` | - | ``'>=' not supported between ...`` |

Two things that table shows.

**Three values were answered with the default.** ``nan >= 0`` is ``False``, and so
is any negative, so ``-5``, ``-99`` and ``nan`` each resolved to ``D - 1`` - the
*same* column the ``-1`` sentinel names. A request no column satisfies became the
documented default, with nothing logged and nothing in the result to say so, so
the value the caller meant and the value that was used differ and both decodes
return identical clean joint targets. That is the one outcome a caller cannot
detect, and it is the reason the check belongs where the value arrives rather than
in the selector that consumes it.

**Eight escaped naming neither the parameter nor the function.** The index reached
``action_chunk[:, gidx]`` or ``np.delete``, so the refusal came from numpy about
an axis, or from the ``>=`` comparison about two types - not from the surface that
received the value. ``IndexError`` and ``TypeError`` also miss the ``ValueError``
channel this function documents, so an ``except ValueError`` around the decode did
not catch them.

The value is not only a caller's: the provider reads it from the policy server's
metadata (``int(meta.get("gripper_dim_index", -1))``) and forwards it here, so a
misconfigured server sending ``-5`` reached the silent half and one sending ``99``
reached the numpy half.

It now routes through
:func:`~strands_robots.policies.vera.sim_ik.coerce_gripper_dim_index`, beside the
sibling that owns the encoding width, which delegates numeric-ness, ``bool``
rejection, finiteness and the float64 range to
:func:`~strands_robots.utils.finite_number_error` and decides only the sentinel
and the sign. Two consequences worth pinning: an integral float now *decodes*
instead of failing to index - ``6.0`` is what a config read produces, and it is
what ``int()`` produced on the provider path - and an in-range index that names no
column of *this* chunk is reported against the chunk's width, which needs the
width and so is checked where the width is known.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import numpy as np

from strands_robots.policies.vera.sim_ik import (
    GRIPPER_INDEX_LAST,
    coerce_gripper_dim_index,
    decode_vera_delta_chunk_to_targets,
)
from strands_robots.utils import finite_number_error

#: Width of the chunk every behavioural test below decodes: 3 translation, 3
#: rotation, 1 gripper.
WIDTH = 7

#: Indices a 7-wide chunk can be read with, in the spellings a caller reaches
#: them by. Integral floats are included deliberately: an index read from JSON or
#: a YAML config arrives as ``6.0``, and ``int(meta["gripper_dim_index"])`` on the
#: provider path produced one too.
USABLE: list[Any] = [GRIPPER_INDEX_LAST, 0, 3, 6, 6.0, np.int64(6), np.float64(0.0)]

#: Indices that name no column: a negative other than the sentinel, and a
#: fractional one. Refused by the sentinel/sign half of the rule.
NOT_AN_INDEX: list[Any] = [-2, -5, -99, 2.7, -0.5]

#: Values that are not a usable number at all. Refused by the shared numeric
#: domain rather than by anything decided here.
NOT_A_NUMBER: list[Any] = [
    True,
    False,
    "6",
    "abc",
    [6],
    {},
    None,
    float("nan"),
    float("inf"),
    float("-inf"),
    10**400,
]

UNUSABLE: list[Any] = NOT_AN_INDEX + NOT_A_NUMBER

#: In-range for the rule but not for a 7-wide chunk. Refused where the width is
#: known rather than by the rule, so it is kept out of :data:`UNUSABLE`.
PAST_THE_LAST_COLUMN: list[Any] = [7, 8, 99]


class _StubBridge:
    """Duck-typed IK bridge: the decoder touches only these three members."""

    def __init__(self, nq: int = 3) -> None:
        self.model = type("_M", (), {"nq": nq})()

    def ee_pose(self, q: Any) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = np.asarray(q, dtype=np.float64)[:3]
        return pose

    def solve(self, target: Any, q: Any) -> np.ndarray:
        return np.asarray(target, dtype=np.float64)[:3, 3].copy()


def _chunk(width: int = WIDTH, steps: int = 3) -> np.ndarray:
    """A chunk whose every column carries a distinct value.

    That is what makes "which column was read as the gripper" observable: the
    returned gripper series equals exactly one column of the input.
    """
    row = np.arange(1, width + 1, dtype=np.float64) * 0.01
    return np.tile(row, (steps, 1))


def _decode(gripper_dim_index: Any, *, width: int = WIDTH, has_gripper: bool = True) -> Any:
    """Run the real decoder over a ``width``-wide chunk.

    ``Any`` for the index because these tests deliberately supply values outside
    the declared ``int``; the bridge is duck-typed, so the decode path needs
    neither ``mink`` nor ``mujoco``.
    """
    bridge: Any = _StubBridge()
    return decode_vera_delta_chunk_to_targets(
        _chunk(width),
        bridge,
        np.zeros(3, dtype=np.float64),
        rotation_dim=3,
        has_gripper=has_gripper,
        gripper_dim_index=gripper_dim_index,
    )


def _column_read_as_gripper(result: Any, width: int = WIDTH) -> int:
    """Index of the input column the decode returned as the gripper series."""
    got = np.asarray(result["gripper"], dtype=np.float64)
    chunk = _chunk(width)
    for col in range(width):
        if np.allclose(got, chunk[:, col]):
            return col
    raise AssertionError(f"gripper series {got!r} matches no column of the chunk")


class TestTheGripperColumnIndexDomain:
    """The rule itself, with no chunk, no bridge and no sim stack."""

    def test_an_index_a_chunk_can_be_read_with_is_accepted(self) -> None:
        for value in USABLE:
            assert coerce_gripper_dim_index(value, "gripper_dim_index", "ctx") == (int(value), None), value

    def test_an_accepted_index_is_normalized_to_an_int(self) -> None:
        for value in USABLE:
            index, error = coerce_gripper_dim_index(value, "gripper_dim_index", "ctx")
            assert error is None
            assert type(index) is int, (value, type(index))

    def test_a_value_naming_no_column_is_refused(self) -> None:
        for value in UNUSABLE:
            index, error = coerce_gripper_dim_index(value, "gripper_dim_index", "ctx")
            assert index is None, value
            assert error, value

    def test_the_refusal_names_the_surface_the_parameter_and_the_value(self) -> None:
        for value in UNUSABLE:
            _, error = coerce_gripper_dim_index(value, "gripper_dim_index", "decode_x")
            assert error is not None
            assert error.startswith("decode_x: gripper_dim_index "), error
            assert repr(value) in error, error

    def test_a_number_outside_the_accepted_set_is_told_what_the_set_is(self) -> None:
        for value in NOT_AN_INDEX:
            _, error = coerce_gripper_dim_index(value, "gripper_dim_index", "ctx")
            assert error is not None
            assert f"must be {GRIPPER_INDEX_LAST} (the trailing column) or a column index >= 0" in error, error

    def test_only_the_sentinel_and_the_sign_are_decided_here(self) -> None:
        """Everything else is the shared numeric domain's, verbatim.

        This is what stops the two rules drifting apart: the local helper adds a
        membership decision on top of :func:`finite_number_error` and nothing else,
        so a value that domain refuses is refused here with that domain's message.
        """
        divergent = []
        for value in [*USABLE, *UNUSABLE, *PAST_THE_LAST_COLUMN]:
            shared = finite_number_error(value, "gripper_dim_index", "ctx")
            _, local = coerce_gripper_dim_index(value, "gripper_dim_index", "ctx")
            if shared is not None and local != shared:
                divergent.append((value, shared, local))
        assert divergent == [], divergent

    def test_the_sentinel_is_not_reachable_by_an_equal_looking_value(self) -> None:
        """``-1.0`` compares equal to the sentinel and is a legitimate spelling.

        ``True``/``False`` also compare equal to ``1``/``0`` but are refused by
        the shared domain, so a boolean can never arrive as a column index.
        """
        assert coerce_gripper_dim_index(-1.0, "gripper_dim_index", "ctx") == (GRIPPER_INDEX_LAST, None)
        assert coerce_gripper_dim_index(True, "gripper_dim_index", "ctx")[0] is None
        assert coerce_gripper_dim_index(False, "gripper_dim_index", "ctx")[0] is None


class TestTheDecoderRefusesAnIndexItCannotRead:
    """The behavioural half: a real decode per value."""

    def test_an_unusable_index_is_refused_as_a_value_error(self) -> None:
        for value in UNUSABLE:
            try:
                _decode(value)
            except ValueError:
                continue
            except Exception as exc:  # noqa: BLE001 - the escape is the finding
                raise AssertionError(f"{value!r} raised {type(exc).__name__}, not ValueError: {exc}") from exc
            raise AssertionError(f"{value!r} was accepted")

    def test_the_refusal_names_this_function_and_the_parameter(self) -> None:
        for value in UNUSABLE:
            try:
                _decode(value)
            except ValueError as exc:
                text = str(exc)
                assert text.startswith("decode_vera_delta_chunk_to_targets: gripper_dim_index "), text
            else:  # pragma: no cover - covered by the sibling test
                raise AssertionError(f"{value!r} was accepted")

    def test_a_negative_other_than_the_sentinel_no_longer_reads_the_last_column(self) -> None:
        """The silent half of the defect, pinned as a refusal.

        ``-5`` used to resolve to ``D - 1``, the same column the sentinel names,
        so an index no column satisfies was answered with the default and the
        substitution was reported nowhere.
        """
        assert _column_read_as_gripper(_decode(GRIPPER_INDEX_LAST)) == WIDTH - 1
        for value in (-2, -5, -99):
            try:
                _decode(value)
            except ValueError as exc:
                assert "gripper_dim_index" in str(exc)
            else:
                raise AssertionError(f"{value!r} was accepted")

    def test_a_non_finite_index_no_longer_reads_the_last_column(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            try:
                _decode(value)
            except ValueError as exc:
                assert "gripper_dim_index" in str(exc)
            else:
                raise AssertionError(f"{value!r} was accepted")

    def test_a_usable_index_still_reads_the_column_it_names(self) -> None:
        assert _column_read_as_gripper(_decode(GRIPPER_INDEX_LAST)) == WIDTH - 1
        assert _column_read_as_gripper(_decode(WIDTH - 1)) == WIDTH - 1
        assert _column_read_as_gripper(_decode(0)) == 0
        assert _column_read_as_gripper(_decode(3)) == 3

    def test_an_integral_float_index_now_decodes_instead_of_failing_to_index(self) -> None:
        """The capability the normalization adds, mirroring ``rotation_dim``.

        ``6.0`` reached ``action_chunk[:, 6.0]`` and raised ``IndexError: only
        integers, slices ...`` before this change.
        """
        result = _decode(6.0)
        assert _column_read_as_gripper(result) == WIDTH - 1
        assert np.asarray(result["qpos"]).shape == (3, 3)

    def test_a_numpy_integer_index_is_still_accepted(self) -> None:
        """No narrowing: numpy indexing honors it, so the guard must too."""
        assert _column_read_as_gripper(_decode(np.int64(6))) == WIDTH - 1


class TestAnIndexPastTheLastColumnIsReportedAgainstTheChunkWidth:
    """The half that needs the chunk, so it is checked where the width is known."""

    def test_an_index_past_the_last_column_is_refused(self) -> None:
        for value in PAST_THE_LAST_COLUMN:
            try:
                _decode(value)
            except ValueError as exc:
                assert "addresses no column" in str(exc), str(exc)
            else:
                raise AssertionError(f"{value!r} was accepted")

    def test_the_refusal_names_the_parameter_the_value_and_the_width(self) -> None:
        try:
            _decode(99)
        except ValueError as exc:
            text = str(exc)
        else:  # pragma: no cover - covered by the sibling test
            raise AssertionError("99 was accepted")
        assert text.startswith("decode_vera_delta_chunk_to_targets: gripper_dim_index 99 "), text
        assert f"{WIDTH}-wide action chunk" in text, text
        assert f"columns 0..{WIDTH - 1}" in text, text

    def test_the_same_index_is_usable_on_a_wider_chunk(self) -> None:
        """The bound is the chunk's, not the rule's - so it moves with the chunk."""
        assert _column_read_as_gripper(_decode(7, width=8), width=8) == 7


class TestTheIndexIsReadOnlyWhenAGripperColumnIsClaimed:
    """``has_gripper=False`` means nothing reads it, so nothing may refuse it."""

    def test_an_unusable_index_is_inert_without_a_gripper_column(self) -> None:
        for value in UNUSABLE + PAST_THE_LAST_COLUMN:
            result = _decode(value, has_gripper=False)
            assert result["gripper"] is None, value

    def test_the_pose_width_check_still_reports_a_server_mismatch(self) -> None:
        """A chunk too narrow for the encoding is still the error it always was."""
        try:
            _decode(GRIPPER_INDEX_LAST, width=4)
        except ValueError as exc:
            assert "pose dims" in str(exc), str(exc)
        else:
            raise AssertionError("a 4-wide chunk was accepted for a 3+3 encoding")


class TestTheRuleCostsNoDependency:
    """Reaching the domain must not drag the sim stack into the light base env."""

    def test_reaching_the_rule_loads_no_heavy_module(self) -> None:
        probe = (
            "import sys;"
            "from strands_robots.policies.vera.sim_ik import coerce_gripper_dim_index as c;"
            "assert c(-5, 'gripper_dim_index', 'ctx')[0] is None;"
            "assert c(6.0, 'gripper_dim_index', 'ctx')[0] == 6;"
            "print(sorted(m for m in ('mink', 'mujoco', 'torch') if m in sys.modules))"
        )
        out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=180)
        assert out.stdout.strip() == "[]", out.stdout
