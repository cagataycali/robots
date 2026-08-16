"""Arity contract for the WBC ``target_orientation`` command override.

:meth:`WBCPolicy._resolve_command` builds the observation's command block, and
two of its components are VECTORS the caller may override per call. They were
held to different arity rules:

* ``target_velocity`` goes through :meth:`WBCPolicy._validate_velocity`, which
  refuses fewer than three entries -- ``target_velocity must have at least 3
  elements [vx, vy, omega], got 2``.
* ``target_orientation`` went through :meth:`WBCPolicy._validate_orientation`,
  which checked every component's VALUE and not the count. The write is
  ``command[4 : 4 + n_rpy] = rpy[:n_rpy]`` with ``n_rpy = min(c - 4,
  rpy.shape[0])`` into a ZERO-INITIALISED block, so a short sequence left the
  axes it did not supply at ``0.0`` -- not at the ``rpy_cmd`` value that applies
  when the kwarg is omitted entirely.

So with ``rpy_cmd=[0.7, 0.8, 0.9]``, omitting the kwarg commanded
``[0.7, 0.8, 0.9]`` while ``target_orientation=[0.5]`` commanded
``[0.5, 0.0, 0.0]`` -- silently commanding zero roll and yaw for axes the caller
never mentioned, discarding targets they DID configure, under a ``success``
result. ``target_orientation=[]`` discarded all three. Every component of those
inputs is a usable finite number, so the per-component value rule cannot see it:
this is an arity question, and it is the one class of caller mistake the sibling
vector component of the same block already refused.

The asymmetry with a LONG sequence is deliberate and is pinned below. A longer
orientation is truncated because every component the block has room for is
honored and only the surplus is dropped -- which is exactly what
:meth:`_validate_velocity` does with a packed velocity (``vel_full[:3]``). A
shorter one is the opposite: components the caller never mentioned are
overwritten with a value they did not choose.

Both ``_resolve_command`` implementations -- the 7-wide one here and
:meth:`WBCGaitPolicy._resolve_command`, whose ``freq_cmd`` slot pushes rpy to
``[5:8]`` -- call the same inherited validator, so both surfaces are pinned.

Out of scope: ``config.rpy_cmd`` itself also accepts a short vector. That is a
different question rather than the same defect -- a short CONFIG default has no
other source for the axes it omits, so zero is the only value those axes could
take, whereas a short per-call override discards a value the caller did supply.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.wbc.policy import WBCPolicy

from .test_command_override_domains import _config, _gait, _resolve

# A distinctive configured orientation, so a discarded axis is unmistakable: a
# zero left behind by a short override cannot be confused with a configured 0.0.
RPY_CMD: list[float] = [0.7, 0.8, 0.9]

# Sequences carrying fewer than the three components roll/pitch/yaw needs. Every
# component in each is a usable finite number, so the per-component value rule
# accepts them all -- the count is the whole defect. The scalar spellings are the
# same shape reached through a different type: ``np.asarray(0.5).ravel()`` is a
# well-formed one-element array.
SHORT_ORIENTATIONS: list[Any] = [
    [0.5, 0.6],
    [0.5],
    (0.5, 0.6),
    0.5,
    np.float64(0.5),
    np.array([0.5]),
    [],
    (),
    np.array([]),
]

# Lengths the block honors. Three is exact; more is truncated to what fits, which
# is deliberate and matches the sibling velocity component.
HONORED_ORIENTATIONS: list[Any] = [
    [0.1, 0.2, 0.3],
    [0.1, 0.2, 0.3, 0.4],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    np.array([0.1, 0.2, 0.3]),
]

# The message shape both vector components of the block share.
_ARITY_MESSAGE = re.compile(r"must have at least 3 elements \[[^]]+\], got \d+")


def _main_policy() -> Any:
    """A non-gait policy whose rpy slots are ``command[4:7]``."""
    return WBCPolicy(config=_config(rpy_cmd=RPY_CMD), walk=False, allow_missing_models=True)


def _gait_policy() -> Any:
    """A gait policy whose ``freq_cmd`` slot pushes rpy to ``command[5:8]``."""
    return _gait(config=_config(rpy_cmd=RPY_CMD, command_dim=8, single_obs_dim=95))


# (label, policy factory, first rpy slot) for the two independent block builders.
SURFACES: list[tuple[str, Any, int]] = [
    ("WBCPolicy", _main_policy, 4),
    ("WBCGaitPolicy", _gait_policy, 5),
]
_SURFACE_IDS = [label for label, _, _ in SURFACES]


class TestAShortOrientationIsRefused:
    """A partial ``target_orientation`` names the parameter instead of zeroing."""

    @pytest.mark.parametrize("value", SHORT_ORIENTATIONS, ids=[repr(v) for v in SHORT_ORIENTATIONS])
    @pytest.mark.parametrize(("label", "factory", "first"), SURFACES, ids=_SURFACE_IDS)
    def test_refused_naming_the_parameter_and_the_count(self, label: str, factory: Any, first: int, value: Any) -> None:
        with pytest.raises(ValueError, match=r"target_orientation must have at least 3 elements") as excinfo:
            _resolve(factory(), target_orientation=value)
        # The count the caller actually supplied is what makes it actionable.
        supplied = np.asarray(value, dtype=np.float64).ravel().shape[0]
        assert f"got {supplied}" in str(excinfo.value)

    def test_the_refusal_carries_the_axis_names(self) -> None:
        """``[roll, pitch, yaw]`` tells the caller what the three components are."""
        with pytest.raises(ValueError, match=r"\[roll, pitch, yaw\]"):
            _resolve(_main_policy(), target_orientation=[0.5])

    @pytest.mark.parametrize(("label", "factory", "first"), SURFACES, ids=_SURFACE_IDS)
    def test_no_partial_override_reaches_the_block(self, label: str, factory: Any, first: int) -> None:
        """The refusal precedes the write, so no axis is left holding a zero.

        This is the behavioural heart: pre-fix the call returned a block whose
        unsupplied axes held ``0.0`` in place of the configured target.
        """
        policy = factory()
        with pytest.raises(ValueError):
            _resolve(policy, target_orientation=[0.5])
        # The same policy still resolves the configured orientation afterwards -
        # the refused call left no state behind.
        command, _ = _resolve(policy, target_orientation=None)
        assert command[first : first + 3] == pytest.approx(RPY_CMD)


class TestTheHonoredOrientationsAreUnchanged:
    """The guard refuses only the arity nothing can honor."""

    @pytest.mark.parametrize(("label", "factory", "first"), SURFACES, ids=_SURFACE_IDS)
    def test_omitting_it_still_uses_the_configured_orientation(self, label: str, factory: Any, first: int) -> None:
        command, _ = _resolve(factory(), target_orientation=None)
        assert command[first : first + 3] == pytest.approx(RPY_CMD)

    @pytest.mark.parametrize(("label", "factory", "first"), SURFACES, ids=_SURFACE_IDS)
    def test_exactly_three_components_are_written_verbatim(self, label: str, factory: Any, first: int) -> None:
        command, _ = _resolve(factory(), target_orientation=[0.1, 0.2, 0.3])
        assert command[first : first + 3] == pytest.approx([0.1, 0.2, 0.3])

    @pytest.mark.parametrize("value", HONORED_ORIENTATIONS, ids=[repr(v) for v in HONORED_ORIENTATIONS])
    @pytest.mark.parametrize(("label", "factory", "first"), SURFACES, ids=_SURFACE_IDS)
    def test_a_long_orientation_is_still_truncated_not_refused(
        self, label: str, factory: Any, first: int, value: Any
    ) -> None:
        """Surplus components are dropped, as they are for a packed velocity.

        Every component the block has room for is honored, so unlike a short
        sequence nothing the caller supplied is replaced by a value they did not
        choose. Two existing rollout tests supply 4- and 6-component
        orientations; refusing them would be a behaviour change, not a fix.
        """
        command, _ = _resolve(factory(), target_orientation=value)
        expected = np.asarray(value, dtype=np.float64).ravel()[:3]
        assert command[first : first + 3] == pytest.approx(expected)


class TestTheTwoVectorComponentsAgree:
    """Neither vector component of the block may accept what the other refuses.

    ``target_velocity`` and ``target_orientation`` are both triples written into
    the same zero-initialised block by the same method. A caller supplying too
    few components is one class of mistake, so it must get one answer -- which is
    what stops the two rules drifting apart again.
    """

    @pytest.mark.parametrize("value", SHORT_ORIENTATIONS, ids=[repr(v) for v in SHORT_ORIENTATIONS])
    def test_a_short_vector_is_refused_for_both(self, value: Any) -> None:
        policy = _main_policy()
        with pytest.raises(ValueError, match=_ARITY_MESSAGE):
            _resolve(policy, target_velocity=value)
        with pytest.raises(ValueError, match=_ARITY_MESSAGE):
            _resolve(policy, target_orientation=value)

    def test_the_two_refusals_share_one_message_shape(self) -> None:
        """Same sentence, differing only in the parameter and the axis names."""
        policy = _main_policy()
        with pytest.raises(ValueError) as velocity:
            _resolve(policy, target_velocity=[0.5])
        with pytest.raises(ValueError) as orientation:
            _resolve(policy, target_orientation=[0.5])
        assert str(velocity.value) == "target_velocity must have at least 3 elements [vx, vy, omega], got 1"
        assert str(orientation.value) == "target_orientation must have at least 3 elements [roll, pitch, yaw], got 1"

    def test_both_still_accept_a_full_triple(self) -> None:
        """Non-vacuity: the parity above is not two blanket refusals."""
        policy = _main_policy()
        assert _resolve(policy, target_velocity=[0.1, 0.2, 0.3])[1] == pytest.approx([0.1, 0.2, 0.3])
        assert _resolve(policy, target_orientation=[0.1, 0.2, 0.3])[0][4:7] == pytest.approx([0.1, 0.2, 0.3])


class TestArityIsCheckedBeforeValues:
    """A short sequence reports its count, not a component complaint.

    :meth:`_validate_velocity` orders its checks coercion -> arity -> per
    component, so a caller who supplies too few components learns that first
    rather than being told about the one component they did supply.
    """

    def test_a_short_sequence_reports_the_count_not_the_component(self) -> None:
        with pytest.raises(ValueError, match=r"at least 3 elements"):
            _resolve(_main_policy(), target_orientation=[float("nan")])

    def test_a_full_triple_still_reports_the_bad_component(self) -> None:
        with pytest.raises(ValueError, match=r"target_orientation\[1\]"):
            _resolve(_main_policy(), target_orientation=[0.0, float("nan"), 0.0])

    def test_a_non_numeric_sequence_still_reports_the_type(self) -> None:
        """The coercion failure precedes the arity check, as it does for velocity."""
        with pytest.raises(ValueError, match="target_orientation must be a numeric sequence"):
            _resolve(_main_policy(), target_orientation="ab")
