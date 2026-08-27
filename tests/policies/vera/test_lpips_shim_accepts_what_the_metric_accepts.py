"""The LPIPS compat shim accepts exactly what the check it stands in for accepts.

:mod:`strands_robots.policies.vera.docker.sitecustomize_vera` exists because
VERA's DFoT metrics import ``_valid_img`` and ``NoTrainLpips`` from
``torchmetrics.image.lpip``, and modern torchmetrics keeps neither name there.
The image installs the module as an auto-imported ``sitecustomize``, so it runs
before anything else in the container and re-exposes both names on that module.
VERA then calls the shim's ``_valid_img`` rather than the real one.

That makes the shim's verdict the metric's verdict, and the two branches of the
real check are deliberately asymmetric. ``torchmetrics.functional.image.lpips``
bounds both ends of ``[0, 1]`` when ``normalize=True`` and only the LOWER end of
``[-1, 1]`` when ``normalize=False``: a frame already in the network's own range
is allowed to overshoot ``1.0``, which is what a decoder emitting a tanh-shaped
frame routinely does. ``_lpips_update`` turns a ``False`` from that check into a
``ValueError``, so an upper bound added on the ``normalize=False`` side does not
make the metric stricter - it makes an otherwise healthy eval raise.

The cases below are graded against the range each frame actually carries rather
than against a hand-written expectation, and the table is checked for the
asymmetry itself, so a shim that collapses the two branches into one rule is
caught here instead of in a container.

The frames are plain :class:`numpy.ndarray` objects. The real check is duck-typed
over ``min``/``max``/``ndim``/``shape``, and torch is not a dependency of this
package, so an array is both a faithful stand-in and one that runs everywhere.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import numpy as np
import pytest

# The container helpers are excluded from type checking (``tool.mypy.exclude``):
# they are written against the dependency set of the image, where ``torchmetrics``
# is installed and is not stubbed here. Reaching the module by name is how
# tests/policies/vera/test_offline_ckpt_index_skips_an_unusable_provenance_record.py
# reaches its own subject on that same path: it keeps an excluded module out of
# this file's static import graph and types it as ``Any``, which is all an
# unchecked module is worth to a caller.
shim: Any = importlib.import_module("strands_robots.policies.vera.docker.sitecustomize_vera")

# The smallest float32 above 1.0. A frame that overshoots by exactly this much is
# the sharpest form of the case the shim used to refuse.
FLOAT32_STEP_ABOVE_ONE = float(np.nextafter(np.float32(1.0), np.float32(2.0)))


def _module(name: str, **attrs: Any) -> Any:
    """A stand-in module carrying the attributes an importer will read off it."""
    module: Any = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _PrivateNoTrainLpips:
    """Stands in for the class modern torchmetrics keeps under a leading underscore."""


def _install_torchmetrics(monkeypatch: pytest.MonkeyPatch, **lpip_attrs: Any) -> Any:
    """Make ``torchmetrics.image.lpip`` importable, as it is inside the image.

    Args:
        monkeypatch: Fixture used to restore every ``sys.modules`` entry.
        lpip_attrs: Attributes to place on the ``lpip`` stand-in. Omitting
            ``_valid_img`` / ``NoTrainLpips`` reproduces the modern layout the
            shim exists for.

    Returns:
        The ``torchmetrics.image.lpip`` stand-in the shim will patch.
    """
    lpip = _module("torchmetrics.image.lpip", **lpip_attrs)
    image = _module("torchmetrics.image", __path__=[], lpip=lpip)
    torchmetrics = _module("torchmetrics", __path__=[], image=image)
    for name, module in (
        ("torchmetrics", torchmetrics),
        ("torchmetrics.image", image),
        ("torchmetrics.image.lpip", lpip),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return lpip


def _applied(monkeypatch: pytest.MonkeyPatch, **lpip_attrs: Any) -> Any:
    """Install the stand-in, run the shim over it, and hand back the module."""
    lpip = _install_torchmetrics(monkeypatch, _NoTrainLpips=_PrivateNoTrainLpips, **lpip_attrs)
    shim._apply()
    return lpip


def _frame(low: float, high: float, *, batch: int = 1, channels: int = 3, dims: int = 4) -> np.ndarray:
    """A frame batch spanning exactly ``[low, high]``.

    Args:
        low: Value planted in the first element, so ``min()`` is exactly this.
        high: Value planted in the last element, so ``max()`` is exactly this.
        batch: Leading batch dimension.
        channels: Channel count - the check demands 3.
        dims: Rank of the result - the check demands 4.

    Returns:
        A ``float32`` array whose extremes are the two arguments.
    """
    shape = (batch, channels) + (2,) * (dims - 2) if dims >= 2 else (batch,) * dims
    array = np.full(shape, (low + high) / 2.0, dtype=np.float32)
    flat = array.reshape(-1)
    flat[0] = low
    flat[-1] = high
    return array


# Each entry is a frame plus the verdict the real check returns for it, on both
# branches. The verdicts are not free choices: they follow from the frame's own
# range under the rule ``normalize`` bounds ``[0, 1]`` at both ends and
# ``not normalize`` bounds ``[-1, ...]`` from below only.
CASES: tuple[tuple[str, np.ndarray, bool, bool], ...] = (
    ("the-unit-range", _frame(0.0, 1.0), True, True),
    ("a-narrow-band-inside-the-unit-range", _frame(0.2, 0.8), True, True),
    ("the-network-range", _frame(-1.0, 1.0), False, True),
    ("one-float-step-above-the-network-range", _frame(-1.0, FLOAT32_STEP_ABOVE_ONE), False, True),
    ("a-visible-overshoot-above-one", _frame(-1.0, 1.05), False, True),
    ("far-above-one", _frame(-1.0, 3.0), False, True),
    ("below-minus-one", _frame(-1.05, 1.0), False, False),
    ("far-below-minus-one", _frame(-4.0, 0.5), False, False),
)

_IDS = tuple(case[0] for case in CASES)


class TestTheCaseTableIsWhatItClaims:
    """Each expected verdict follows from the frame's range, and both branches differ."""

    @pytest.mark.parametrize(("label", "frame", "normalized", "unnormalized"), CASES, ids=_IDS)
    def test_each_verdict_follows_from_the_range_the_frame_carries(
        self, label: str, frame: np.ndarray, normalized: bool, unnormalized: bool
    ) -> None:
        low, high = float(frame.min()), float(frame.max())
        assert normalized == (low >= 0.0 and high <= 1.0), f"{label}: normalized verdict contradicts [{low}, {high}]"
        assert unnormalized == (low >= -1.0), f"{label}: unnormalized verdict contradicts a low of {low}"

    def test_both_verdicts_occur_on_both_branches(self) -> None:
        assert {case[2] for case in CASES} == {True, False}, "the normalized column never refuses or never accepts"
        assert {case[3] for case in CASES} == {True, False}, "the unnormalized column never refuses or never accepts"

    def test_the_asymmetry_between_the_branches_is_represented(self) -> None:
        disagreeing = [case[0] for case in CASES if case[2] != case[3]]
        assert disagreeing, "no case distinguishes the two branches, so a single shared rule would pass"

    def test_an_overshoot_above_one_is_represented(self) -> None:
        overshooting = [case[0] for case in CASES if float(case[1].max()) > 1.0]
        assert overshooting, "no case overshoots 1.0, so the upper bound under test is never reached"


class TestTheShimReproducesTheCheckItStandsIn:
    """The installed ``_valid_img`` answers what the real one answers."""

    @pytest.mark.parametrize(("label", "frame", "normalized", "unnormalized"), CASES, ids=_IDS)
    def test_the_verdict_matches_on_both_branches(
        self, monkeypatch: pytest.MonkeyPatch, label: str, frame: np.ndarray, normalized: bool, unnormalized: bool
    ) -> None:
        lpip = _applied(monkeypatch)
        assert lpip._valid_img(frame, True) is normalized, f"{label}: wrong verdict with normalize=True"
        assert lpip._valid_img(frame, False) is unnormalized, f"{label}: wrong verdict with normalize=False"

    def test_the_verdict_is_a_plain_bool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A caller reading the result with ``is True`` needs a bool, not an array scalar."""
        lpip = _applied(monkeypatch)
        assert type(lpip._valid_img(_frame(0.0, 1.0), True)) is bool


class TestTheUnnormalizedBranchIsBoundedOnlyFromBelow:
    """A frame already in the network's range may overshoot 1.0."""

    def test_one_float_step_above_one_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        lpip = _applied(monkeypatch)
        frame = _frame(-1.0, FLOAT32_STEP_ABOVE_ONE)
        assert float(frame.max()) > 1.0, "the frame does not overshoot, so it grades nothing"
        assert lpip._valid_img(frame, False) is True

    @pytest.mark.parametrize("high", [1.0, FLOAT32_STEP_ABOVE_ONE, 1.05, 2.0, 10.0, 1.0e6])
    def test_raising_the_upper_extreme_never_turns_an_accept_into_a_refusal(
        self, monkeypatch: pytest.MonkeyPatch, high: float
    ) -> None:
        """The derived form of "no upper bound": the branch is monotone in ``max``."""
        lpip = _applied(monkeypatch)
        assert lpip._valid_img(_frame(-1.0, high), False) is True

    @pytest.mark.parametrize("low", [-1.0 - 1.0e-3, -1.05, -2.0, -100.0])
    def test_the_lower_bound_is_still_enforced(self, monkeypatch: pytest.MonkeyPatch, low: float) -> None:
        lpip = _applied(monkeypatch)
        assert lpip._valid_img(_frame(low, 0.5), False) is False

    def test_minus_one_itself_is_inside_the_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        lpip = _applied(monkeypatch)
        assert lpip._valid_img(_frame(-1.0, 0.5), False) is True


class TestTheNormalizedBranchIsBoundedAtBothEnds:
    """Widening the other branch must not widen this one with it."""

    @pytest.mark.parametrize("high", [FLOAT32_STEP_ABOVE_ONE, 1.05, 2.0])
    def test_an_overshoot_above_one_is_refused(self, monkeypatch: pytest.MonkeyPatch, high: float) -> None:
        lpip = _applied(monkeypatch)
        assert lpip._valid_img(_frame(0.0, high), True) is False

    @pytest.mark.parametrize("low", [-1.0e-3, -1.0, -2.0])
    def test_anything_below_zero_is_refused(self, monkeypatch: pytest.MonkeyPatch, low: float) -> None:
        lpip = _applied(monkeypatch)
        assert lpip._valid_img(_frame(low, 1.0), True) is False

    def test_the_closed_unit_range_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        lpip = _applied(monkeypatch)
        assert lpip._valid_img(_frame(0.0, 1.0), True) is True


class TestTheShapeRulesAreUnchanged:
    """A range check that accepts more must not accept a frame of the wrong shape."""

    @pytest.mark.parametrize("normalize", [True, False])
    @pytest.mark.parametrize("dims", [2, 3, 5])
    def test_only_a_rank_four_batch_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch, normalize: bool, dims: int
    ) -> None:
        lpip = _applied(monkeypatch)
        frame = _frame(0.0, 1.0, dims=dims)
        assert frame.ndim == dims
        assert lpip._valid_img(frame, normalize) is False

    @pytest.mark.parametrize("normalize", [True, False])
    @pytest.mark.parametrize("channels", [1, 2, 4])
    def test_only_three_channels_are_accepted(
        self, monkeypatch: pytest.MonkeyPatch, normalize: bool, channels: int
    ) -> None:
        lpip = _applied(monkeypatch)
        assert lpip._valid_img(_frame(0.0, 1.0, channels=channels), normalize) is False


class TestNoTrainLpipsIsReExposed:
    """The other name VERA imports from that module."""

    def test_the_public_name_becomes_the_private_class(self, monkeypatch: pytest.MonkeyPatch) -> None:
        lpip = _applied(monkeypatch)
        assert lpip.NoTrainLpips is _PrivateNoTrainLpips

    def test_a_module_carrying_neither_spelling_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Nothing to alias means nothing is invented under the public name."""
        lpip = _install_torchmetrics(monkeypatch)
        shim._apply()
        assert not hasattr(lpip, "NoTrainLpips")


class TestTheShimOnlyFillsInWhatIsMissing:
    """A name the installed torchmetrics still provides is the one that is used."""

    def test_an_existing_valid_img_is_not_replaced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sentinel = object()
        lpip = _applied(monkeypatch, _valid_img=sentinel)
        assert lpip._valid_img is sentinel

    def test_an_existing_public_no_train_lpips_is_not_replaced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _PublicNoTrainLpips:
            pass

        lpip = _applied(monkeypatch, NoTrainLpips=_PublicNoTrainLpips)
        assert lpip.NoTrainLpips is _PublicNoTrainLpips

    def test_torchmetrics_being_absent_is_not_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The shim rides on ``sitecustomize``, so a raise here would break the interpreter."""
        monkeypatch.setitem(sys.modules, "torchmetrics", None)
        monkeypatch.delitem(sys.modules, "torchmetrics.image", raising=False)
        monkeypatch.delitem(sys.modules, "torchmetrics.image.lpip", raising=False)
        shim._apply()

    def test_applying_the_shim_twice_keeps_the_first_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        lpip = _applied(monkeypatch)
        first = lpip._valid_img
        shim._apply()
        assert lpip._valid_img is first
