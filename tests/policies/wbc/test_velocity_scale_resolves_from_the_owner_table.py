"""The velocity ``cmd_scale`` resolves from its owner table, not from a bare 1.0.

:meth:`WBCConfig.__post_init__` requires ``cmd_scale`` to carry exactly three
entries ``[vx, vy, omega]`` *when provided*, and admits an EMPTY sequence as "not
provided" (``if cmd_scale_length and cmd_scale_length != 3``). But this field's
default is not empty - it is the full upstream triple
``_DEFAULT_CMD_SCALE = (2.0, 2.0, 0.5)``. So omitting the argument and passing an
empty sequence are two spellings of ONE request, and they resolved to different
scales: both command-block builders fell back to a bare unit scale for a vector
too short to slice::

    scale = cmd_scale[:n_vel] if cmd_scale.shape[0] >= n_vel else np.ones(n_vel)

With ``target_velocity=[0.5, -0.25, 2.0]``, omitting ``cmd_scale`` commanded
``[1.0, -0.5, 1.0]`` while ``cmd_scale=[]`` commanded ``[0.5, -0.25, 2.0]`` - so
``vx``/``vy`` arrived HALVED and ``omega`` arrived DOUBLED, under a ``success``
result. A length-TWO ``cmd_scale`` was refused by name the whole time
(``must have exactly 3 entries``), which is what makes the empty one a hole
rather than a policy: the one wrong length that is not reported is the one that
silently substitutes a scale no table states.

That second fallback number is exactly the failure the sibling scale field in the
same module already fixed and documented at length. ``obs_scales`` is completed
from ``_DEFAULT_OBS_SCALES`` at construction, and
:func:`~strands_robots.policies.wbc.observation.build_single_frame` resolves an
omitted key from that same table rather than a bare ``1.0``, because "a second
fallback number ... would silently multiply the 29 joint-velocity entries of the
frame by 20 for exactly the configs that name a sibling key, which the network
reads as a malformed observation". The velocity scale is the other half of the
same idea, and the command block is the observation's FIRST ``command_dim``
entries - through a dense network it reaches all ``num_actions`` joint targets.

Pinned here: an empty ``cmd_scale`` means the documented triple on both command
blocks (:class:`WBCPolicy` and :class:`WBCGaitPolicy`, whose ``freq_cmd`` slot
does not move the velocity slots), the config states the vector the block is
built with, and a component the config DOES state still wins - so the completion
cannot creep into ignoring a scale a caller legitimately chose, including a
deliberate unit scale.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from strands_robots.policies.wbc import config as wbc_config
from strands_robots.policies.wbc.policy import WBCPolicy

from .test_command_override_domains import _config, _gait, _resolve

# The upstream velocity scale, spelled here as the contract rather than imported
# as the implementation: a config that states nothing must command THIS, and the
# agreement cell below pins that the module's own table still says so.
DOCUMENTED_CMD_SCALE: tuple[float, float, float] = (2.0, 2.0, 0.5)

# A velocity whose three components scale by three DIFFERENT factors, so a
# substituted scale cannot coincide with the documented one on any axis: omega's
# documented 0.5 against a unit fallback is the widest gap and the wrong
# direction (a doubled yaw rate).
VELOCITY: list[float] = [0.5, -0.25, 2.0]
DOCUMENTED_BLOCK: list[float] = [v * s for v, s in zip(VELOCITY, DOCUMENTED_CMD_SCALE, strict=True)]

# Every spelling of an empty cmd_scale the config accepts as "not stated".
EMPTY_SPELLINGS: list[Any] = [[], (), np.array([])]
_EMPTY_IDS = ["list", "tuple", "ndarray"]


def _main(**cfg: Any) -> Any:
    """A non-gait policy whose velocity slots are ``command[0:3]``."""
    return WBCPolicy(config=_config(**cfg), walk=False, allow_missing_models=True)


def _gait_surface(**cfg: Any) -> Any:
    """A gait policy: its ``freq_cmd`` slot widens the block but not the velocity."""
    return _gait(config=_config(command_dim=8, single_obs_dim=95, **cfg))


# (label, factory) for the two independent command-block builders.
SURFACES: list[tuple[str, Any]] = [("WBCPolicy", _main), ("WBCGaitPolicy", _gait_surface)]
_SURFACE_IDS = [label for label, _ in SURFACES]


def _block(policy: Any) -> list[float]:
    """The velocity slots of one command block, built from :data:`VELOCITY`."""
    command, _raw = _resolve(policy, target_velocity=VELOCITY)
    return [float(x) for x in np.asarray(command)[:3]]


class TestAnEmptyCmdScaleMeansTheDocumentedScale:
    """ "Not stated" resolves from the owner table on both command blocks."""

    @pytest.mark.parametrize("label,factory", SURFACES, ids=_SURFACE_IDS)
    @pytest.mark.parametrize("empty", EMPTY_SPELLINGS, ids=_EMPTY_IDS)
    def test_an_empty_cmd_scale_commands_the_documented_scale(self, label: str, factory: Any, empty: Any) -> None:
        assert _block(factory(cmd_scale=empty)) == pytest.approx(DOCUMENTED_BLOCK)

    @pytest.mark.parametrize("label,factory", SURFACES, ids=_SURFACE_IDS)
    @pytest.mark.parametrize("empty", EMPTY_SPELLINGS, ids=_EMPTY_IDS)
    def test_an_empty_cmd_scale_agrees_with_omitting_it(self, label: str, factory: Any, empty: Any) -> None:
        """The two spellings of one request give one command block."""
        assert _block(factory(cmd_scale=empty)) == pytest.approx(_block(factory()))


class TestTheConfigStatesTheVectorTheBlockIsBuiltWith:
    """``cmd_scale`` reads back as the scale actually applied, like ``obs_scales``."""

    @pytest.mark.parametrize("empty", EMPTY_SPELLINGS, ids=_EMPTY_IDS)
    def test_an_empty_cmd_scale_is_completed_at_construction(self, empty: Any) -> None:
        assert list(_config(cmd_scale=empty).cmd_scale) == pytest.approx(list(DOCUMENTED_CMD_SCALE))

    def test_the_module_table_is_the_documented_scale(self) -> None:
        """The contract above is the module's own single owner of the scale."""
        assert tuple(wbc_config._DEFAULT_CMD_SCALE) == DOCUMENTED_CMD_SCALE


class TestAStatedScaleStillWins:
    """The completion fills only what the config leaves out."""

    @pytest.mark.parametrize("label,factory", SURFACES, ids=_SURFACE_IDS)
    def test_a_stated_unit_scale_is_honored(self, label: str, factory: Any) -> None:
        """A caller who ASKS for unit scale gets it - it is not read as "not stated"."""
        assert _block(factory(cmd_scale=[1.0, 1.0, 1.0])) == pytest.approx(VELOCITY)

    @pytest.mark.parametrize("label,factory", SURFACES, ids=_SURFACE_IDS)
    def test_a_stated_scale_is_honored(self, label: str, factory: Any) -> None:
        assert _block(factory(cmd_scale=[3.0, 4.0, 5.0])) == pytest.approx([1.5, -1.0, 10.0])

    @pytest.mark.parametrize("label,factory", SURFACES, ids=_SURFACE_IDS)
    def test_a_short_vector_reaching_the_block_fills_only_the_missing_components(
        self, label: str, factory: Any
    ) -> None:
        """A config assembled AROUND the constructor is the fallback's input class.

        ``__post_init__`` completes an empty vector, so a partial one reaches the
        builder only on a config mutated past the constructor - which is the case
        ``build_single_frame``'s ``obs_scales`` fallback answers for too. The
        stated component wins; the rest resolve from the table, not from 1.0.
        """
        policy = factory()
        object.__setattr__(policy._config, "cmd_scale", [3.0])
        assert _block(policy) == pytest.approx([0.5 * 3.0, -0.25 * 2.0, 2.0 * 0.5])


class TestTheSurroundingContractIsUnchanged:
    """The refusals and the unscaled velocity the completion must not disturb."""

    @pytest.mark.parametrize("bad", [[2.0, 2.0], [2.0], [2.0, 2.0, 0.5, 1.0]], ids=["two", "one", "four"])
    def test_a_wrong_nonempty_length_is_still_refused_by_name(self, bad: list[float]) -> None:
        with pytest.raises(ValueError, match=r"WBCConfig\.cmd_scale must have exactly 3 entries"):
            _config(cmd_scale=bad)

    def test_a_scalar_cmd_scale_is_still_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match=r"WBCConfig\.cmd_scale must be a sequence of 3 numbers"):
            _config(cmd_scale=2.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "2.0"], ids=repr)
    def test_a_non_finite_component_is_still_refused_by_name(self, bad: Any) -> None:
        with pytest.raises(ValueError, match=r"cmd_scale\[0\]"):
            _config(cmd_scale=[bad, 2.0, 0.5])

    @pytest.mark.parametrize("label,factory", SURFACES, ids=_SURFACE_IDS)
    @pytest.mark.parametrize("spelling", [*EMPTY_SPELLINGS, [2.0, 2.0, 0.5]], ids=[*_EMPTY_IDS, "stated"])
    def test_the_raw_velocity_is_never_scaled(self, label: str, factory: Any, spelling: Any) -> None:
        """Walk-vs-main selection reads the UNSCALED triple, whatever the scale."""
        _command, raw = _resolve(factory(cmd_scale=spelling), target_velocity=VELOCITY)
        assert [float(x) for x in np.asarray(raw)] == pytest.approx(VELOCITY)
