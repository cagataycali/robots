# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The period a reference motion is resampled onto has to be a period.

``control_dt`` is the one field of
:class:`~strands_robots.policies.protomotions.config.ProtoMotionsConfig`'s timing
block the control path spends. It is the period
:class:`~strands_robots.policies.protomotions.motion_utils.MotionPlayer`
RESAMPLES a raw motion onto, so it decides how many frames one clip becomes, and
``ProtoMotionsPolicy.get_actions`` advances the playhead exactly one of those
frames per tick - so the frame count it produces is how long the reference motion
lasts, in ticks.

Its two gated siblings in the same ``__post_init__`` each carry a written reason
(``anchor_body_index``/``root_body_index`` must address a row of ``body_names``;
``action_ema_alpha`` must be a weight in ``(0, 1]``), and the sidecar loader
carries two comments saying a value is "handed through raw, not through
``int()``" because coercing first is what laundered a yaml ``true`` past the
domain. Three lines below the second of those comments the timing block did
exactly that: ``control_dt = float(timing.get("control_dt", ...))``, and no
domain stood behind it.

Measured on the pre-fix tree, resampling a 3-second 30 fps clip and asking the
default lookahead offsets ``(1, 2, 4, 8)`` for frames ahead of the playhead:

===========================  ==========  =======  ==================  ==============================
sidecar ``timing.control_dt``  resolved   frames   widest joint travel  lookahead offsets past the end
===========================  ==========  =======  ==================  ==============================
``0.02`` (shipped)             ``0.02``      151          1.997 rad                        0 of 4
``true``                       ``1.0``         4          0.999 rad                        2 of 4
``-0.02``                      ``-0.02``       1          0.0 rad                          4 of 4
``inf``                        ``inf``         1          0.0 rad                          4 of 4
===========================  ==========  =======  ==================  ==============================

A negative period and ``inf`` collapse the clip to a SINGLE frame, because the
conversion is ``max(1, round(motion_length / control_dt) + 1)`` and both make
that term non-positive; the index clamp in
:meth:`MotionPlayer.get_state_at_frame` then serves that one frame for every tick
of the episode - a tracker that reports a motion and holds one pose. ``0`` was
accepted by the config and surfaced as ``ZeroDivisionError`` out of the loader's
own ``logger.info`` argument (``1.0 / cfg.control_dt``), so the only thing in the
tree that objected to it was a log format string - and a directly constructed
``ProtoMotionsConfig(control_dt=0)`` was accepted with nothing raised at all.
``nan`` reached ``int(round(...))`` inside the resampler and raised "cannot
convert float NaN to integer" there, naming neither the field nor the sidecar.

Both places a period enters are held to the domain, because a cache dict's own
``control_dt`` OUTRANKS the ``control_dt=`` argument: gating only the config
would leave a caller who passed a good period playing a clip at an unusable one.

``physics_dt`` and ``decimation`` are deliberately NOT gated, and the two cells
at the bottom pin that boundary: no reader in this package consumes either, so
refusing a value there would change which sidecars load with no behaviour to
protect.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.protomotions.config import (
    GTP_G1_CONTROL_DT,
    GTP_G1_DEFAULT_LOOKAHEAD_STEPS,
    ProtoMotionsConfig,
    load_config_from_yaml,
)
from strands_robots.policies.protomotions.motion_utils import MotionPlayer

# Periods no reader can honor. ``True``/``"0.02"`` are the spellings the removed
# ``float()`` used to launder; the rest are the values that reached a consumer.
_UNUSABLE: list[Any] = [
    pytest.param(True, id="bool_true_reads_as_a_one_second_period"),
    pytest.param(False, id="bool_false_reads_as_zero"),
    pytest.param(0, id="zero"),
    pytest.param(0.0, id="zero_float"),
    pytest.param(-0.02, id="negative_period_collapses_the_clip"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(float("-inf"), id="minus_inf"),
    pytest.param("0.02", id="string_spelling_of_the_shipped_period"),
    pytest.param(None, id="none"),
    pytest.param([0.02], id="list"),
]

# Periods a caller may legitimately ask for: the shipped rate, a deliberate
# 25 Hz, a fractional rate, and a NumPy scalar read out of a config array.
_USABLE: list[Any] = [
    pytest.param(GTP_G1_CONTROL_DT, 0.02, id="the_shipped_50hz_rate"),
    pytest.param(0.04, 0.04, id="a_deliberate_25hz"),
    pytest.param(1.0 / 30.0, 1.0 / 30.0, id="a_fractional_30hz"),
    pytest.param(np.float32(0.05), 0.05, id="a_numpy_scalar_from_a_config_array"),
]

_SRC_FPS = 30.0
_SRC_SECONDS = 3.0
_SRC_FRAMES = int(_SRC_FPS * _SRC_SECONDS) + 1
_NUM_BODIES = 4
_NUM_DOFS = 6


def _built(**overrides: Any) -> ProtoMotionsConfig:
    """Build a config with ``overrides`` applied.

    Deliberately typed loose, the way the sibling body-index guard is: most of
    the cases below pass values the signature refuses, which is the point, and
    splatting a ``dict[str, Any]`` through the real constructor is what a type
    checker reports rather than what the guard reports.
    """
    return ProtoMotionsConfig(**overrides)


def _sidecar(tmp_path: Path, **timing: Any) -> Path:
    """Write a sidecar whose only non-default block is ``timing``."""
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "unified_pipeline.yaml"
    path.write_text(yaml.safe_dump({"timing": timing}), encoding="utf-8")
    return path


def _cache(num_rows: int = 8, **extra: Any) -> dict[str, Any]:
    """A minimal consistent cache dict, plus whatever ``extra`` states."""
    cache: dict[str, Any] = {
        "dof_pos": np.zeros((num_rows, _NUM_DOFS), dtype=np.float32),
        "dof_vel": np.zeros((num_rows, _NUM_DOFS), dtype=np.float32),
        "body_rot": np.zeros((num_rows, _NUM_BODIES, 4), dtype=np.float32),
        "body_pos": np.zeros((num_rows, _NUM_BODIES, 3), dtype=np.float32),
        "body_vel": np.zeros((num_rows, _NUM_BODIES, 3), dtype=np.float32),
        "body_ang_vel": np.zeros((num_rows, _NUM_BODIES, 3), dtype=np.float32),
    }
    cache.update(extra)
    return cache


def _raw_motion(tmp_path: Path) -> str:
    """A single-motion ``.pt`` sweeping every joint once over ``_SRC_SECONDS``.

    A half-cycle sine rather than a whole one: a full cycle sampled at exactly
    its own period reads as no travel at all, which would credit a wrong period
    with a defect it does not have.
    """
    torch = pytest.importorskip("torch")
    t = np.linspace(0.0, _SRC_SECONDS, _SRC_FRAMES, dtype=np.float32)
    dof = np.stack([np.sin(np.pi * t / _SRC_SECONDS + j / 100.0) for j in range(_NUM_DOFS)], axis=1)
    rot = np.zeros((_SRC_FRAMES, _NUM_BODIES, 4), dtype=np.float32)
    rot[..., 3] = 1.0
    zeros = np.zeros((_SRC_FRAMES, _NUM_BODIES, 3), dtype=np.float32)
    path = tmp_path / "motion.pt"
    torch.save(
        {
            "fps": _SRC_FPS,
            "rigid_body_pos": torch.from_numpy(zeros.copy()),
            "rigid_body_rot": torch.from_numpy(rot),
            "rigid_body_vel": torch.from_numpy(zeros.copy()),
            "rigid_body_ang_vel": torch.from_numpy(zeros.copy()),
            "dof_pos": torch.from_numpy(dof.astype(np.float32)),
            "dof_vel": torch.from_numpy(np.zeros_like(dof, dtype=np.float32)),
        },
        path,
    )
    return str(path)


class TestTheConfigRefusesAnUnplayablePeriod:
    """``ProtoMotionsConfig`` settles ``control_dt`` when it is built."""

    @pytest.mark.parametrize("value", _UNUSABLE)
    def test_an_unusable_period_is_refused_at_construction(self, value: Any) -> None:
        """Pre-fix every one of these was stored and carried to the resampler."""
        with pytest.raises(ValueError, match="control_dt"):
            _built(control_dt=value)

    @pytest.mark.parametrize(("value", "expected"), _USABLE)
    def test_a_usable_period_is_kept_as_a_plain_float(self, value: Any, expected: float) -> None:
        """Nothing a caller could legitimately ask for becomes an error.

        The normalisation matters for the NumPy case: the period divides a motion
        length and multiplies a frame number, so a ``float32`` read out of a
        config array would otherwise set the dtype of that arithmetic.
        """
        config = _built(control_dt=value)
        assert config.control_dt == pytest.approx(expected)
        assert type(config.control_dt) is float

    def test_the_default_config_is_the_shipped_rate(self) -> None:
        """The domain does not move the checkpoint's own value."""
        assert ProtoMotionsConfig().control_dt == GTP_G1_CONTROL_DT


class TestTheSidecarRefusesAnUnplayablePeriod:
    """A checkpoint's ``unified_pipeline.yaml`` reaches the same domain."""

    @pytest.mark.parametrize("value", _UNUSABLE)
    def test_a_sidecar_period_is_reported_as_a_value_error(self, value: Any, tmp_path: Path) -> None:
        """The loader hands the value through raw, so the domain sees it.

        Pre-fix ``true`` built a config declaring a one-second control period,
        and ``0``/``false`` raised ``ZeroDivisionError`` out of the loader's own
        ``logger.info`` argument - neither naming the field.
        """
        with pytest.raises(ValueError, match="control_dt"):
            load_config_from_yaml(_sidecar(tmp_path, control_dt=value))

    def test_a_sidecar_stating_a_deliberate_rate_still_loads(self, tmp_path: Path) -> None:
        """The control: a real sidecar timing block is unaffected."""
        config = load_config_from_yaml(_sidecar(tmp_path, control_dt=0.04, decimation=10))
        assert (config.control_dt, config.decimation) == (0.04, 10)

    def test_a_sidecar_with_no_timing_block_takes_the_shipped_rate(self, tmp_path: Path) -> None:
        """An absent field is still the default, not a refusal."""
        assert load_config_from_yaml(_sidecar(tmp_path)).control_dt == GTP_G1_CONTROL_DT


class TestThePlayerRefusesAnUnplayablePeriod:
    """Both places a period enters ``MotionPlayer`` are held to the domain."""

    @pytest.mark.parametrize("value", _UNUSABLE)
    def test_the_argument_is_refused_and_names_the_player(self, value: Any) -> None:
        with pytest.raises(ValueError, match=r"MotionPlayer: control_dt"):
            MotionPlayer(_cache(), control_dt=value)

    @pytest.mark.parametrize("value", _UNUSABLE)
    def test_a_cache_stating_an_unusable_period_is_refused_and_names_the_cache(self, value: Any) -> None:
        """The cache's own key outranks the argument, so it needs its own gate.

        A good ``control_dt=`` does not rescue it: pre-fix this is exactly the
        path that played a clip at a period its caller never asked for.
        """
        with pytest.raises(ValueError, match=r"MotionPlayer cache: control_dt"):
            MotionPlayer(_cache(control_dt=value), control_dt=GTP_G1_CONTROL_DT)

    def test_a_cache_without_the_key_keeps_the_argument(self) -> None:
        assert MotionPlayer(_cache(), control_dt=0.05).control_dt == pytest.approx(0.05)

    def test_a_cache_with_the_key_outranks_the_argument(self) -> None:
        player = MotionPlayer(_cache(control_dt=1.0 / 30.0), control_dt=0.05)
        assert player.control_dt == pytest.approx(1.0 / 30.0)

    def test_an_npz_round_trip_still_reloads_its_own_period(self, tmp_path: Path) -> None:
        """Reading a NumPy scalar back is a dtype unwrap, not a laundered value."""
        path = str(tmp_path / "cache.npz")
        MotionPlayer(_cache(control_dt=0.04)).save_cache_npz(path)
        assert MotionPlayer(path).control_dt == pytest.approx(0.04)

    def test_a_period_the_caller_wrapped_in_an_array_is_still_a_period(self) -> None:
        """A zero-dim array is how this module's own loaders hand a scalar over.

        ``np.load`` returns one for every key of an ``.npz``, and the
        cache-shaped-``.pt`` branch of ``_load_file`` wraps the scalars with
        ``np.asarray`` alongside the channels. Unwrapping it is a dtype step.
        """
        player = MotionPlayer(_cache(control_dt=np.asarray(0.04)))
        assert player.control_dt == pytest.approx(0.04)
        assert type(player.control_dt) is float

    @pytest.mark.parametrize("value", [float("nan"), -0.02, 0.0])
    def test_wrapping_an_unusable_period_in_an_array_does_not_hide_it(self, value: float) -> None:
        """The unwrap carries no value, so it cannot launder one past the domain."""
        with pytest.raises(ValueError, match=r"MotionPlayer cache: control_dt"):
            MotionPlayer(_cache(control_dt=np.asarray(value)))


class TestAUsablePeriodPlaysTheWholeClip:
    """What the domain is protecting, measured on a real resample."""

    def test_the_shipped_period_spans_the_source_motion(self, tmp_path: Path) -> None:
        """The resampled clip lasts as long as the motion it came from.

        This is the property a wrong period destroys: the playhead advances one
        resampled frame per control tick, so the frame count IS the reference
        motion's duration in ticks.
        """
        player = MotionPlayer(_raw_motion(tmp_path), control_dt=GTP_G1_CONTROL_DT)
        span_s = (player.total_frames - 1) * player.control_dt
        assert player.total_frames == 151
        assert span_s == pytest.approx(_SRC_SECONDS, abs=GTP_G1_CONTROL_DT)

    def test_every_lookahead_offset_lands_inside_the_clip(self, tmp_path: Path) -> None:
        """From the first tick, none of the tracker's future frames is clamped.

        Pre-fix a ``control_dt: true`` sidecar left 2 of these 4 offsets past the
        end of a 4-frame clip, and a negative period left all 4 past the end of a
        1-frame one - the state the clamp then served for the whole episode.
        """
        player = MotionPlayer(_raw_motion(tmp_path), control_dt=GTP_G1_CONTROL_DT)
        assert max(GTP_G1_DEFAULT_LOOKAHEAD_STEPS) < player.total_frames
        future = player.get_future_references(0, list(GTP_G1_DEFAULT_LOOKAHEAD_STEPS))
        # Distinct frames, so the window is a window rather than one repeated pose.
        rows = {tuple(row.tolist()) for row in future["dof_pos"]}
        assert len(rows) == len(GTP_G1_DEFAULT_LOOKAHEAD_STEPS)

    def test_a_coarser_period_still_carries_the_whole_motion(self, tmp_path: Path) -> None:
        """A deliberately slower rate is a real request, not a degraded one."""
        player = MotionPlayer(_raw_motion(tmp_path), control_dt=0.04)
        travel = float(np.ptp(player.as_cache()["dof_pos"], axis=0).max())
        reference = MotionPlayer(_raw_motion(tmp_path), control_dt=GTP_G1_CONTROL_DT)
        assert travel == pytest.approx(float(np.ptp(reference.as_cache()["dof_pos"], axis=0).max()), abs=0.01)


class TestTheUncheckedTimingFieldsAreADeclaredBoundary:
    """``physics_dt`` and ``decimation`` carry no reader, so they carry no domain."""

    @pytest.mark.parametrize("field_name", ["physics_dt", "decimation"])
    def test_the_field_is_admitted_because_nothing_reads_it(self, field_name: str) -> None:
        """Refusing here would change which sidecars load and protect nothing.

        Pinned so the boundary is visible: if either field gains a reader, this
        cell is what says the decision was never made rather than merely missed.
        """
        config = _built(**{field_name: float("nan")})
        assert math.isnan(float(getattr(config, field_name)))

    def test_the_stated_relation_between_the_three_is_not_checked(self) -> None:
        """``decimation`` is documented as ``control_dt / physics_dt``, unenforced.

        The shipped defaults do satisfy it (``0.02 / 0.001 == 20``); a sidecar
        that contradicts it is still loaded, for the same reason.
        """
        assert ProtoMotionsConfig().decimation == pytest.approx(
            ProtoMotionsConfig().control_dt / ProtoMotionsConfig().physics_dt
        )
        contradictory = _built(control_dt=0.02, physics_dt=0.001, decimation=5)
        assert contradictory.decimation == 5
