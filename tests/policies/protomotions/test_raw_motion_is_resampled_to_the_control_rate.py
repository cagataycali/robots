# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A raw ProtoMotions motion is resampled onto the tracker's own control rate.

:class:`~strands_robots.policies.protomotions.motion_utils.MotionPlayer` accepts
two kinds of source: a cache dict that is *already* at the control rate, and a
raw ProtoMotions ``.pt``, which is at whatever rate it was captured at (30 Hz,
120 Hz, per-entry inside a library) and has to be resampled. The resample is the
only thing that decides which source instants the ONNX tracker ever sees, so
getting it wrong is not slower or coarser playback - it is a different motion:

* a mis-derived frame count truncates the clip or invents frames past its end,
  and :meth:`MotionPlayer.get_state_at_frame` clamps to that count, so the tail
  is silently unreachable rather than reported;
* a packed library slices one entry out of a concatenated buffer using
  ``length_starts``, so an offset that is off by one entry plays *another clip*
  under the requested index, at full confidence;
* the six state channels are sliced with that one offset, so a channel that
  drifted from the others would pair one clip's joints with another's bodies.

Both raw layouts and every line of the resampler were unexercised. The existing
``.pt`` coverage (``test_motion_file_loading``) writes a *cache-shaped* ``.pt``,
which returns from the loader before the resampler is reached, so the rate
conversion, the library slice, the rotation interpolation and the
unrecognised-layout refusal had no test at all.

The fixture makes every source row identifiable: each channel is keyed to the
**global** source frame index (``body_pos[i]`` carries ``x = i``,
``body_ang_vel[i]`` carries ``z = i``, and so on), so an assertion can name the
source row a resampled frame came from instead of merely checking that it looks
plausible. The source rate is 10 Hz and the control rates are exact multiples of
it, so a resampled frame either lands on a source instant (and must equal that
row exactly) or lands on a known fraction between two (and must equal their
blend exactly) - no tolerance is needed for the five linear channels.

Rotations are the one channel interpolated on the sphere rather than the line,
and two plausible degradations of that are pinned by measurement rather than by
reading the source:

* interpolating rotations with the same ``lerp`` the other five channels use
  leaves a blended rotation off the unit sphere - measured ``0.9239`` for a
  half-blend of the fixture's 90-degrees-apart neighbours, against the
  ``1.0`` asserted below;
* normalising that blend instead (``nlerp``) restores the norm but not the
  *rate*: measured per-step rotation over a 4x upsample is ``21.598 deg`` /
  ``23.402 deg`` alternating, against a uniform ``22.5 deg``, i.e. a reference
  motion that speeds up and slows down inside every source interval. The
  constant-rate assertion below holds slerp to ``0.05 deg`` (the observed slerp
  spread is ``1e-4 deg``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.protomotions import MotionPlayer
from tests.mocks.torch_mock import real_torch_installed

#: Reading a raw motion goes through ``torch.load``; the numpy torch stand-in has
#: no serializer, so the classes that write a ``.pt`` need the real package. The
#: two source-level refusals at the bottom of this module do not, and are not
#: marked, so they still run on an install without torch.
needs_real_torch = pytest.mark.skipif(
    not real_torch_installed(),
    reason="writes and reads a real .pt through torch.load; the torch mock has no serializer",
)

_NUM_BODIES = 3
_NUM_DOFS = 4

#: Source period of the fixture motions, seconds (10 Hz).
_SRC_DT = 0.1
#: Frame counts of the two entries packed into the fixture library. Different on
#: purpose: an entry's length has to come from ``motion_num_frames``, not from
#: dividing the buffer by the entry count.
_ENTRY_FRAMES = (3, 5)
#: Start row of each entry inside the concatenated buffer.
_ENTRY_STARTS = (0, _ENTRY_FRAMES[0])
#: Rotation advance per source frame, degrees. Large enough that a rotation
#: interpolated on the line instead of the sphere is measurably off the unit
#: sphere (see the module docstring).
_DEG_PER_SOURCE_FRAME = 90.0


def _quat_about_z(degrees: float) -> np.ndarray:
    """Unit ``xyzw`` quaternion for a rotation of ``degrees`` about world +Z."""
    half = np.deg2rad(degrees) / 2.0
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float32)


def _source_rows(total_frames: int) -> dict[str, np.ndarray]:
    """Six state channels whose every row is keyed to its own frame index."""
    body_pos = np.zeros((total_frames, _NUM_BODIES, 3), dtype=np.float32)
    body_rot = np.zeros((total_frames, _NUM_BODIES, 4), dtype=np.float32)
    body_vel = np.zeros((total_frames, _NUM_BODIES, 3), dtype=np.float32)
    body_ang_vel = np.zeros((total_frames, _NUM_BODIES, 3), dtype=np.float32)
    dof_pos = np.zeros((total_frames, _NUM_DOFS), dtype=np.float32)
    dof_vel = np.zeros((total_frames, _NUM_DOFS), dtype=np.float32)
    for frame in range(total_frames):
        for body in range(_NUM_BODIES):
            body_pos[frame, body] = (frame, body, 0.0)
            body_vel[frame, body] = (0.0, frame, 0.0)
            body_ang_vel[frame, body] = (0.0, 0.0, frame)
            body_rot[frame, body] = _quat_about_z(frame * _DEG_PER_SOURCE_FRAME)
        for dof in range(_NUM_DOFS):
            dof_pos[frame, dof] = frame + dof / 100.0
            dof_vel[frame, dof] = -(frame + dof / 100.0)
    return {
        "body_pos": body_pos,
        "body_rot": body_rot,
        "body_vel": body_vel,
        "body_ang_vel": body_ang_vel,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
    }


def _write_pt(payload: dict[str, Any], path: Path) -> str:
    """Save ``payload`` as a real ``.pt`` of tensors and plain scalars."""
    import torch

    tensors = {
        key: (torch.from_numpy(value) if isinstance(value, np.ndarray) else value) for key, value in payload.items()
    }
    torch.save(tensors, path)
    return str(path)


def _packed_library(path: Path) -> tuple[str, dict[str, np.ndarray]]:
    """A two-entry ProtoMotions motion library, plus its raw source channels."""
    total = sum(_ENTRY_FRAMES)
    rows = _source_rows(total)
    payload: dict[str, Any] = {
        "length_starts": np.asarray(_ENTRY_STARTS, dtype=np.int64),
        "motion_num_frames": np.asarray(_ENTRY_FRAMES, dtype=np.int64),
        "motion_dt": np.asarray([_SRC_DT] * len(_ENTRY_FRAMES), dtype=np.float32),
        "gts": rows["body_pos"],
        "grs": rows["body_rot"],
        "gvs": rows["body_vel"],
        "gavs": rows["body_ang_vel"],
        "dps": rows["dof_pos"],
        "dvs": rows["dof_vel"],
    }
    return _write_pt(payload, path), rows


def _single_motion(path: Path, fps: float, num_frames: int) -> tuple[str, dict[str, np.ndarray]]:
    """A single-motion ProtoMotions ``.pt`` declaring its own ``fps``."""
    rows = _source_rows(num_frames)
    payload: dict[str, Any] = {
        "fps": fps,
        "rigid_body_pos": rows["body_pos"],
        "rigid_body_rot": rows["body_rot"],
        "rigid_body_vel": rows["body_vel"],
        "rigid_body_ang_vel": rows["body_ang_vel"],
        "dof_pos": rows["dof_pos"],
        "dof_vel": rows["dof_vel"],
    }
    return _write_pt(payload, path), rows


def _expected_frames(num_source_frames: int, src_dt: float, control_dt: float) -> int:
    """Frame count the documented conversion gives, derived independently here."""
    motion_length = src_dt * (num_source_frames - 1)
    return max(1, int(round(motion_length / control_dt)) + 1)


def _rotation_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Angle between two ``xyzw`` quaternions, degrees, sign-insensitive."""
    dot = abs(float(np.dot(np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64))))
    return float(np.rad2deg(2.0 * np.arccos(min(dot, 1.0))))


#: The five channels interpolated linearly, so an exact blend can be asserted.
_LINEAR_CHANNELS = ("body_pos", "body_vel", "body_ang_vel", "dof_pos", "dof_vel")


@needs_real_torch
class TestAPackedLibraryPlaysTheEntryItWasAskedFor:
    """``motion_index`` selects an entry, and every channel is cut at its rows."""

    @pytest.mark.parametrize("motion_index", [0, 1])
    def test_the_entry_is_resampled_from_its_own_rows_and_length(self, tmp_path: Path, motion_index: int) -> None:
        """The clip spans exactly the requested entry's first and last source row."""
        path, rows = _packed_library(tmp_path / "library.pt")
        control_dt = 0.05
        player = MotionPlayer(path, control_dt=control_dt, motion_index=motion_index)

        start = _ENTRY_STARTS[motion_index]
        last = start + _ENTRY_FRAMES[motion_index] - 1
        assert player.total_frames == _expected_frames(_ENTRY_FRAMES[motion_index], _SRC_DT, control_dt)
        assert player.control_dt == control_dt
        assert player.num_bodies == _NUM_BODIES
        assert player.num_dofs == _NUM_DOFS

        first_frame = player.get_state_at_frame(0)
        final_frame = player.get_state_at_frame(player.total_frames - 1)
        for channel in _LINEAR_CHANNELS:
            assert np.array_equal(first_frame[channel], rows[channel][start]), channel
            assert np.array_equal(final_frame[channel], rows[channel][last]), channel
        assert np.allclose(first_frame["body_rot"], rows["body_rot"][start], atol=1e-6)
        assert np.allclose(final_frame["body_rot"], rows["body_rot"][last], atol=1e-6)

    def test_the_two_entries_of_one_library_are_not_the_same_clip(self, tmp_path: Path) -> None:
        """Non-vacuity: the index would be inert if the entries were identical."""
        path, _ = _packed_library(tmp_path / "library.pt")
        first = MotionPlayer(path, control_dt=0.05, motion_index=0)
        second = MotionPlayer(path, control_dt=0.05, motion_index=1)
        assert first.total_frames != second.total_frames
        assert not np.array_equal(
            first.get_state_at_frame(0)["body_pos"],
            second.get_state_at_frame(0)["body_pos"],
        )


@needs_real_torch
class TestTheResampleIsARateConversion:
    """Resampled frames sit at control instants, not at renumbered source rows."""

    def test_a_frame_on_a_source_instant_is_that_source_row_exactly(self, tmp_path: Path) -> None:
        """A 2x upsample puts every even control frame back on a source row."""
        path, rows = _packed_library(tmp_path / "library.pt")
        start = _ENTRY_STARTS[1]
        player = MotionPlayer(path, control_dt=_SRC_DT / 2, motion_index=1)

        for offset in range(_ENTRY_FRAMES[1]):
            state = player.get_state_at_frame(2 * offset)
            for channel in _LINEAR_CHANNELS:
                assert np.array_equal(state[channel], rows[channel][start + offset]), (channel, offset)

    def test_a_frame_between_two_source_instants_is_their_blend(self, tmp_path: Path) -> None:
        """A 2x upsample puts every odd control frame at the exact half-blend."""
        path, rows = _packed_library(tmp_path / "library.pt")
        start = _ENTRY_STARTS[1]
        player = MotionPlayer(path, control_dt=_SRC_DT / 2, motion_index=1)

        for offset in range(_ENTRY_FRAMES[1] - 1):
            state = player.get_state_at_frame(2 * offset + 1)
            for channel in _LINEAR_CHANNELS:
                expected = 0.5 * (rows[channel][start + offset] + rows[channel][start + offset + 1])
                assert np.allclose(state[channel], expected, atol=1e-6), (channel, offset)

    def test_the_control_period_sets_the_frame_count_and_keeps_the_endpoints(self, tmp_path: Path) -> None:
        """Halving the control period doubles the span over the same clip."""
        path, rows = _packed_library(tmp_path / "library.pt")
        start = _ENTRY_STARTS[1]
        last = start + _ENTRY_FRAMES[1] - 1

        coarse = MotionPlayer(path, control_dt=_SRC_DT / 2, motion_index=1)
        fine = MotionPlayer(path, control_dt=_SRC_DT / 4, motion_index=1)
        assert coarse.total_frames == _expected_frames(_ENTRY_FRAMES[1], _SRC_DT, _SRC_DT / 2)
        assert fine.total_frames == _expected_frames(_ENTRY_FRAMES[1], _SRC_DT, _SRC_DT / 4)
        assert fine.total_frames == 2 * coarse.total_frames - 1

        for player in (coarse, fine):
            assert np.array_equal(player.get_state_at_frame(0)["body_pos"], rows["body_pos"][start])
            assert np.array_equal(
                player.get_state_at_frame(player.total_frames - 1)["body_pos"], rows["body_pos"][last]
            )


@needs_real_torch
class TestRotationsAreInterpolatedOnTheSphere:
    """The rotation channel keeps unit length and a constant angular rate."""

    def test_every_resampled_rotation_is_a_unit_quaternion(self, tmp_path: Path) -> None:
        """A blend taken on the line instead of the sphere measures 0.9239 here."""
        path, _ = _packed_library(tmp_path / "library.pt")
        player = MotionPlayer(path, control_dt=_SRC_DT / 2, motion_index=1)
        for frame in range(player.total_frames):
            norms = np.linalg.norm(player.get_state_at_frame(frame)["body_rot"], axis=-1)
            assert np.allclose(norms, 1.0, atol=1e-6), (frame, norms)

    def test_the_reference_rotation_advances_at_a_constant_rate(self, tmp_path: Path) -> None:
        """Uniform source spacing plus uniform sampling is a uniform rotation."""
        path, _ = _packed_library(tmp_path / "library.pt")
        upsample = 4
        player = MotionPlayer(path, control_dt=_SRC_DT / upsample, motion_index=1)

        steps = [
            _rotation_angle_deg(
                player.get_state_at_frame(frame)["body_rot"][0],
                player.get_state_at_frame(frame + 1)["body_rot"][0],
            )
            for frame in range(player.total_frames - 1)
        ]
        expected = _DEG_PER_SOURCE_FRAME / upsample
        assert steps, "the clip must span more than one control frame for a rate to exist"
        assert max(abs(step - expected) for step in steps) < 0.05, steps


@needs_real_torch
class TestASingleMotionIsResampledFromItsOwnFps:
    """A single-motion payload states its rate as ``fps`` rather than per entry."""

    @pytest.mark.parametrize("fps", [10.0, 20.0])
    def test_the_declared_fps_sets_the_source_rate(self, tmp_path: Path, fps: float) -> None:
        """Two files with the same rows and different ``fps`` are different clips."""
        num_frames = 5
        path, rows = _single_motion(tmp_path / f"motion_{fps:.0f}.pt", fps=fps, num_frames=num_frames)
        control_dt = 0.05
        player = MotionPlayer(path, control_dt=control_dt)

        assert player.total_frames == _expected_frames(num_frames, 1.0 / fps, control_dt)
        assert np.array_equal(player.get_state_at_frame(0)["body_pos"], rows["body_pos"][0])
        assert np.array_equal(
            player.get_state_at_frame(player.total_frames - 1)["body_pos"], rows["body_pos"][num_frames - 1]
        )


class TestASourceThatCannotBePlayedIsRefused:
    """The three refusals a caller reaches before any frame is served.

    Grouped because they answer one question - can this source be played at all
    - and none of them was exercised. Only the first needs a real ``.pt``.
    """

    @needs_real_torch
    def test_a_raw_payload_in_neither_layout_names_both_layouts(self, tmp_path: Path) -> None:
        """A ``.pt`` that is neither a cache nor either raw layout is refused."""
        path = _write_pt({"unrelated": np.zeros((2, 2), dtype=np.float32)}, tmp_path / "mystery.pt")
        with pytest.raises(ValueError, match=r"Unrecognised raw motion format") as refusal:
            MotionPlayer(path)
        message = str(refusal.value)
        assert "length_starts" in message, message
        assert "rigid_body_pos" in message, message

    def test_a_source_that_is_neither_a_cache_nor_a_path_is_refused(self) -> None:
        """The refusal names the type it got, so a misplaced argument is visible."""
        with pytest.raises(TypeError, match=r"cache dict or a \.pt path") as refusal:
            MotionPlayer(42)  # type: ignore[arg-type]
        assert "int" in str(refusal.value)

    def test_a_cache_missing_channels_names_every_missing_one(self) -> None:
        """A hand-built cache is refused by naming the channels it lacks."""
        with pytest.raises(KeyError, match=r"missing required keys") as refusal:
            MotionPlayer({"dof_pos": np.zeros((2, _NUM_DOFS), dtype=np.float32)})
        message = str(refusal.value)
        for channel in ("dof_vel", "body_rot", "body_pos", "body_vel", "body_ang_vel"):
            assert re.search(rf"\b{channel}\b", message), (channel, message)
