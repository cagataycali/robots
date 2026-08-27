"""``history_length`` sizes the history window the tracker actually reads.

``ProtoMotionsPolicy.__init__`` takes ``history_length`` as the number of past
action frames that feed the ``historical_processed_actions`` ONNX input, then
normalises it with ``int()`` to shape the rolling buffer. The parameter is one
of the provider's advertised ``config_keys``, so it arrives from a JSON or YAML
policy config as readily as from a keyword.

A bare ``value < 1`` test in front of that ``int()`` covers the floor and not
the domain, and the two spellings it lets through are the ones a config carries:
``2.7`` becomes a two-frame window and ``true`` a one-frame window, each
reporting a successfully built policy. That is the same laundering
:meth:`ProtoMotionsConfig.__post_init__` documents for its own body indices,
which is why this module grades the ordering as well as the values - the shared
domain has to run *before* the conversion, not after it.

The controls here are the spellings the pre-fix code already honored: an ``int``,
an integral float read from a config, a NumPy integer, and the ``0`` / negative
floor. Widening or narrowing past those is the failure this file exists to catch.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import math
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.factory import create_policy
from strands_robots.policies.protomotions import (
    GTP_G1_JOINT_NAMES,
    ProtoMotionsConfig,
    ProtoMotionsPolicy,
)
from strands_robots.policies.protomotions import policy as policy_module

#: The window lengths a caller can legitimately ask for, and the buffer depth
#: each must produce. Stated here rather than derived from the guard so the
#: expectation does not follow a change to it.
ACCEPTED_SPELLINGS: list[tuple[str, Any, int]] = [
    ("int", 4, 4),
    ("the documented default", 1, 1),
    ("an integral float from a config", 4.0, 4),
    ("a numpy integer", np.int64(4), 4),
    ("a numpy integral float", np.float32(4.0), 4),
]

#: Values that read as a window length and cannot be honored as one. Each is
#: refused by the shared domain rather than coerced into a different window.
LAUNDERED_SPELLINGS: list[tuple[str, Any]] = [
    ("a fractional count", 2.7),
    ("a numpy fractional count", np.float64(2.7)),
    ("a boolean that acts as 1", True),
    ("not-a-number", float("nan")),
    ("infinity", float("inf")),
    ("a digit string", "3"),
    ("nothing at all", None),
    ("a one-element list", [2]),
    ("a count past any buffer", 10**400),
]

#: Below the floor. The pre-fix code refused these too; they are here so a fix
#: to the domain cannot quietly drop the floor with it.
BELOW_FLOOR: list[tuple[str, Any]] = [
    ("zero", 0),
    ("negative", -3),
    ("a boolean that acts as 0", False),
]


class _RecordingSession:
    """A tracker stub that records the shape of every input it is fed."""

    def __init__(self) -> None:
        self.feed_shapes: dict[str, tuple[int, ...]] = {}

    def run(self, output_names: list[str] | None, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.feed_shapes = {name: array.shape for name, array in inputs.items()}
        num_dofs = len(GTP_G1_JOINT_NAMES)
        zeros = np.zeros(num_dofs, dtype=np.float32)
        return [
            zeros.reshape(1, num_dofs),
            zeros.reshape(1, num_dofs),
            np.full((1, num_dofs), 40.0, dtype=np.float32),
            np.full((1, num_dofs), 2.5, dtype=np.float32),
        ]


def _flat_motion_cache(num_frames: int = 20) -> dict[str, Any]:
    num_bodies, num_dofs = 33, len(GTP_G1_JOINT_NAMES)
    return {
        "dof_pos": np.zeros((num_frames, num_dofs), dtype=np.float32),
        "dof_vel": np.zeros((num_frames, num_dofs), dtype=np.float32),
        "body_rot": np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (num_frames, num_bodies, 1)),
        "body_pos": np.zeros((num_frames, num_bodies, 3), dtype=np.float32),
        "body_vel": np.zeros((num_frames, num_bodies, 3), dtype=np.float32),
        "body_ang_vel": np.zeros((num_frames, num_bodies, 3), dtype=np.float32),
        "control_dt": 0.02,
        "num_frames": num_frames,
    }


def _minimal_obs_kwargs() -> dict[str, np.ndarray]:
    num_dofs = len(GTP_G1_JOINT_NAMES)
    return {
        "anchor_rot_xyzw": np.array([0, 0, 0, 1], dtype=np.float32),
        "root_ang_vel_local": np.zeros(3, dtype=np.float32),
        "dof_pos": np.zeros(num_dofs, dtype=np.float32),
        "dof_vel": np.zeros(num_dofs, dtype=np.float32),
    }


def _init_source() -> str:
    return inspect.getsource(ProtoMotionsPolicy.__init__)


def _build(**kwargs: Any) -> ProtoMotionsPolicy:
    """Construct the policy through one ``Any``-typed funnel.

    Several cells below hand ``history_length`` a value its annotation forbids,
    because that off-domain spelling is the subject under test. Routing every
    construction through here keeps the deliberate case out of the type
    checker's way without annotating it away at each call site.
    """
    return ProtoMotionsPolicy(**kwargs)


class TestThePremises:
    """What the rest of this file relies on, stated so a drift is visible."""

    def test_history_length_is_an_advertised_config_key(self) -> None:
        """The value arrives from a policy config, not only from a keyword."""
        from strands_robots.registry.policies import get_policy_provider

        provider = get_policy_provider("protomotions")
        assert provider is not None
        assert "history_length" in provider["config_keys"]

    def test_the_window_length_is_a_buffer_dimension(self) -> None:
        """``history_length`` is read as an array shape, so it must be whole."""
        source = inspect.getsource(policy_module)
        assert "np.zeros(\n                (self._history_length" in source or (
            "self._history_length" in source and "np.zeros(" in source
        )
        assert "reshape(1, self._history_length, self._config.num_dofs)" in source

    def test_the_sibling_config_guards_before_it_normalises(self) -> None:
        """The same package already resolves this ordering the other way.

        ``ProtoMotionsConfig.__post_init__`` sends each body index through the
        shared whole-number domain and only then calls ``int()``, and says why.
        The policy's constructor is held to that same order below; if this
        premise ever fails, the citation for it has gone stale.
        """
        source = inspect.getsource(ProtoMotionsConfig.__post_init__)
        domain_at = source.index("non_negative_whole_number_error(")
        coerce_at = source.index("int(raw)")
        assert domain_at < coerce_at
        prose = " ".join((ProtoMotionsConfig.__post_init__.__doc__ or "").split())
        assert "BEFORE the ``int()`` normalisation" in prose

    def test_every_laundered_spelling_is_distinct_from_every_accepted_one(self) -> None:
        """The two tables do not overlap, so neither cell set is vacuous."""
        accepted = [value for _, value, _ in ACCEPTED_SPELLINGS]
        for _, bad in LAUNDERED_SPELLINGS:
            for good in accepted:
                # A bool compares equal to 1, so identity is the right test.
                assert bad is not good


class TestAWindowLengthThatCannotBeHonoredIsRefused:
    """The regression: a spelling that reads as a count is not coerced into one."""

    @pytest.mark.parametrize(
        ("label", "value"),
        LAUNDERED_SPELLINGS,
        ids=[label.replace(" ", "-") for label, _ in LAUNDERED_SPELLINGS],
    )
    def test_the_constructor_refuses_it_by_name(self, label: str, value: Any) -> None:
        with pytest.raises(ValueError, match=r"^ProtoMotionsPolicy: history_length must be"):
            _build(session=_RecordingSession(), history_length=value)

    @pytest.mark.parametrize(
        ("label", "value"),
        LAUNDERED_SPELLINGS,
        ids=[label.replace(" ", "-") for label, _ in LAUNDERED_SPELLINGS],
    )
    def test_the_public_factory_refuses_it_by_name(self, label: str, value: Any) -> None:
        """The advertised route a policy config travels refuses it too."""
        with pytest.raises(ValueError, match=r"^ProtoMotionsPolicy: history_length must be"):
            create_policy("protomotions", session=_RecordingSession(), history_length=value)

    def test_a_fractional_count_does_not_reach_the_tracker_as_a_shorter_window(self) -> None:
        """The headline: 2.7 must not feed the ONNX graph a two-frame window."""
        session = _RecordingSession()
        with pytest.raises(ValueError, match=r"history_length must be a positive whole number"):
            policy = create_policy(
                "protomotions",
                session=session,
                history_length=2.7,
                motion=_flat_motion_cache(),
            )
            asyncio.run(policy.get_actions({}, "", **_minimal_obs_kwargs()))
        assert session.feed_shapes == {}, "the tracker was fed a window the caller never asked for"

    def test_a_boolean_does_not_reach_the_tracker_as_a_single_frame_window(self) -> None:
        """``true`` from a config is not a window length of one."""
        session = _RecordingSession()
        with pytest.raises(ValueError, match=r"history_length must be a positive whole number"):
            create_policy("protomotions", session=session, history_length=True)
        assert session.feed_shapes == {}

    def test_the_refusal_names_the_value_it_read(self) -> None:
        """The reason quotes what arrived, so a config typo is findable."""
        with pytest.raises(ValueError, match=r"got 2\.7"):
            _build(session=_RecordingSession(), history_length=2.7)

    def test_the_refusal_names_the_class_that_refused(self) -> None:
        with pytest.raises(ValueError, match=r"^ProtoMotionsPolicy: history_length"):
            _build(session=_RecordingSession(), history_length=2.7)

    def test_a_fractional_count_is_refused_for_not_being_whole(self) -> None:
        """The reason states which property of a window length failed."""
        with pytest.raises(ValueError, match=r"must be a positive whole number"):
            _build(session=_RecordingSession(), history_length=2.7)

    def test_a_count_past_any_buffer_is_refused_on_magnitude(self) -> None:
        """The domain owns the ceiling too, so the reason names the parameter.

        Without it the value reaches NumPy, which refuses the allocation with a
        message that names neither the parameter nor the caller.
        """
        with pytest.raises(ValueError, match=r"history_length must be within the range of a 64-bit float"):
            _build(session=_RecordingSession(), history_length=10**400)


class TestTheGuardPrecedesTheNormalisation:
    """Structural: the domain must see the raw value, not the converted one."""

    def test_the_shared_domain_runs_before_the_int_conversion(self) -> None:
        source = _init_source()
        domain_at = source.index("positive_whole_number_error(")
        coerce_at = source.index("int(history_length)")
        assert domain_at < coerce_at, "int() would launder the value before the domain sees it"

    def test_the_constructor_does_not_reimplement_the_domain(self) -> None:
        """A local comparison beside the shared guard is how the two drift."""
        tree = ast.parse(inspect.cleandoc(_init_source()))
        compares = [
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Compare) and "history_length" in ast.unparse(node)
        ]
        assert compares == [], f"history_length is compared directly: {compares}"

    def test_the_domain_comes_from_the_shared_module(self) -> None:
        from strands_robots.utils import positive_whole_number_error

        assert policy_module.positive_whole_number_error is positive_whole_number_error


class TestTheAcceptedWindowLengthsAreUnchanged:
    """Over-reach guard: every spelling the pre-fix code honored still works."""

    @pytest.mark.parametrize(
        ("label", "value", "depth"),
        ACCEPTED_SPELLINGS,
        ids=[label.replace(" ", "-") for label, _, _ in ACCEPTED_SPELLINGS],
    )
    def test_it_builds_and_sizes_the_buffer(self, label: str, value: Any, depth: int) -> None:
        policy = _build(session=_RecordingSession(), history_length=value)
        assert policy._action_history.shape == (depth, len(GTP_G1_JOINT_NAMES))

    @pytest.mark.parametrize(
        ("label", "value", "depth"),
        ACCEPTED_SPELLINGS,
        ids=[label.replace(" ", "-") for label, _, _ in ACCEPTED_SPELLINGS],
    )
    def test_the_tracker_reads_the_window_that_was_asked_for(self, label: str, value: Any, depth: int) -> None:
        session = _RecordingSession()
        policy = create_policy(
            "protomotions",
            session=session,
            history_length=value,
            motion=_flat_motion_cache(),
        )
        asyncio.run(policy.get_actions({}, "", **_minimal_obs_kwargs()))
        assert session.feed_shapes["historical_processed_actions"] == (1, depth, len(GTP_G1_JOINT_NAMES))

    @pytest.mark.parametrize(
        ("label", "value"),
        BELOW_FLOOR,
        ids=[label.replace(" ", "-") for label, _ in BELOW_FLOOR],
    )
    def test_the_floor_still_refuses_it(self, label: str, value: Any) -> None:
        with pytest.raises(ValueError, match=r"history_length"):
            _build(session=_RecordingSession(), history_length=value)

    def test_an_integral_float_is_the_config_case_the_domain_exists_for(self) -> None:
        """A YAML ``4.0`` is integral and meant, so it is honored, not refused."""
        assert math.isclose(4.0, 4)
        policy = _build(session=_RecordingSession(), history_length=4.0)
        assert policy._history_length == 4
        assert isinstance(policy._history_length, int)
