# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every route into the MotionPlayer cache format reads the same format.

:class:`~strands_robots.policies.protomotions.motion_utils.MotionPlayer` takes a
reference-motion cache three ways - as a ``dict`` a caller builds, as an ``.npz``
written by :meth:`MotionPlayer.save_cache_npz`, and as a ``.pt`` that is already
a cache - and the module documents ONE format for all three. Two of its keys are
documented optional: ``control_dt`` (omit it and the ``control_dt=`` argument
stands) and ``num_frames`` (omit it and the channels' own row count is used).

The two file routes used to state a narrower format than the one they read. The
``.npz`` reader indexed all eight keys itself, so a file omitting either scalar
was refused with NumPy's bare ``KeyError: 'control_dt'``, and a file short of a
channel was refused with ``KeyError: 'dof_vel'`` - the first missing channel
only, no list, nothing naming the player or the file. The ``.pt`` reader decided
"this is a cache" on the presence of ``control_dt``, so a cache-shaped ``.pt``
that omitted it was sent down the raw-motion path and refused as
"Unrecognised raw motion format" - a report about a layout the caller was not
using. The ``.npz`` is the route the ``.pt`` refusal itself points callers to,
and the one this module writes, so the format it accepts is the one that has to
match the documentation.

Each route is now a row in one table, so a fourth route or a new optional key is
covered by adding a row rather than by hoping the file readers were kept in step.
The raw layouts are the control: ``body_rot`` is the key no raw layout carries
(a packed library spells its rotations ``grs``, a single motion
``rigid_body_rot``), so a raw motion must still be resampled rather than read as
a cache whose rows happen to be at the wrong rate.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.protomotions import MotionPlayer
from tests.mocks.torch_mock import real_torch_installed

#: The ``.pt`` route goes through ``torch.load``; the numpy torch stand-in has no
#: serializer. The ``dict`` and ``.npz`` rows need no torch and are not marked.
needs_real_torch = pytest.mark.skipif(
    not real_torch_installed(),
    reason="writes and reads a real .pt through torch.load; the torch mock has no serializer",
)

_NUM_FRAMES = 5
_NUM_DOFS = 4
_NUM_BODIES = 3

#: The ``control_dt=`` argument and the period a cache states, deliberately
#: different so an assertion can say WHICH of the two a load honored.
_ARG_DT = 0.02
_CACHE_DT = 0.04

#: Source rate of the raw-motion controls, and the frame count the resampler owes
#: at ``_ARG_DT``: a 4-frame clip at 10 Hz spans 0.3 s, so 0.3 / 0.02 + 1 frames.
_RAW_FPS = 10.0
_RAW_FRAMES = 4
_RESAMPLED_FRAMES = 16


def _cache(**scalars: Any) -> dict[str, Any]:
    """A well-formed cache, plus whichever optional scalars the caller names.

    Each channel is keyed to its frame index, so an assertion can say the three
    routes returned the same ROWS and not merely the same shapes.
    """
    frames = np.arange(_NUM_FRAMES, dtype=np.float32)
    payload: dict[str, Any] = {
        "dof_pos": np.tile(frames[:, None], (1, _NUM_DOFS)),
        "dof_vel": np.tile(frames[:, None], (1, _NUM_DOFS)) * 0.5,
        "body_rot": np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (_NUM_FRAMES, _NUM_BODIES, 1)),
        "body_pos": np.tile(frames[:, None, None], (1, _NUM_BODIES, 3)),
        "body_vel": np.tile(frames[:, None, None], (1, _NUM_BODIES, 3)) * 2.0,
        "body_ang_vel": np.tile(frames[:, None, None], (1, _NUM_BODIES, 3)) * 3.0,
    }
    payload.update(scalars)
    return payload


def _as_dict(payload: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    return dict(payload)


def _as_npz(payload: dict[str, Any], tmp_path: Path) -> str:
    path = tmp_path / "cache.npz"
    np.savez(str(path), **payload)
    return str(path)


def _as_pt(payload: dict[str, Any], tmp_path: Path) -> str:
    import torch

    path = tmp_path / "cache.pt"
    torch.save(
        {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in payload.items()},
        path,
    )
    return str(path)


#: The routes a cache reaches the player by. ``Route`` turns a cache payload into
#: whatever the constructor's ``source`` argument is for that route.
Route = Callable[[dict[str, Any], Path], Any]

every_route = pytest.mark.parametrize(
    "build",
    [
        pytest.param(_as_dict, id="dict"),
        pytest.param(_as_npz, id="npz"),
        pytest.param(_as_pt, id="pt", marks=needs_real_torch),
    ],
)


def _refusal(source: Any, expected: type[Exception]) -> str:
    """Load ``source`` expecting a refusal, and return its message."""
    with pytest.raises(expected) as excinfo:
        MotionPlayer(source, control_dt=_ARG_DT)
    message = str(excinfo.value)
    if isinstance(source, str):
        assert source in message, (
            f"a refused file must name the file, else a caller holding several caches cannot tell "
            f"which one was refused. Got: {message}"
        )
    return message


class TestTheOptionalScalarsAreOptionalOnEveryRoute:
    """``control_dt`` and ``num_frames`` may be omitted however the cache arrives."""

    @every_route
    def test_a_cache_stating_both_scalars_is_read_as_stated(self, build: Route, tmp_path: Path) -> None:
        """The control: nothing about a complete cache changes."""
        player = MotionPlayer(build(_cache(control_dt=_CACHE_DT, num_frames=_NUM_FRAMES), tmp_path), control_dt=_ARG_DT)

        assert player.total_frames == _NUM_FRAMES
        assert player.control_dt == pytest.approx(_CACHE_DT)

    @every_route
    def test_a_cache_omitting_control_dt_keeps_the_argument(self, build: Route, tmp_path: Path) -> None:
        """Pre-fix: ``KeyError: 'control_dt'`` (.npz) / "Unrecognised raw motion format" (.pt)."""
        player = MotionPlayer(build(_cache(num_frames=_NUM_FRAMES), tmp_path), control_dt=_ARG_DT)

        assert player.control_dt == pytest.approx(_ARG_DT)
        assert player.total_frames == _NUM_FRAMES

    @every_route
    def test_a_cache_omitting_num_frames_takes_its_own_row_count(self, build: Route, tmp_path: Path) -> None:
        """Pre-fix: ``KeyError: 'num_frames'`` on the ``.npz`` route."""
        player = MotionPlayer(build(_cache(control_dt=_CACHE_DT), tmp_path), control_dt=_ARG_DT)

        assert player.total_frames == _NUM_FRAMES
        assert player.control_dt == pytest.approx(_CACHE_DT)

    @every_route
    def test_a_cache_omitting_both_scalars_is_still_playable(self, build: Route, tmp_path: Path) -> None:
        """The six channels alone are the format's required content."""
        player = MotionPlayer(build(_cache(), tmp_path), control_dt=_ARG_DT)

        assert (player.total_frames, player.num_dofs, player.num_bodies) == (_NUM_FRAMES, _NUM_DOFS, _NUM_BODIES)
        assert player.control_dt == pytest.approx(_ARG_DT)

    @needs_real_torch
    def test_the_three_routes_return_the_same_rows(self, tmp_path: Path) -> None:
        """One format means one clip: same channels, frame for frame.

        Shapes agreeing is not the claim - a route that dropped or reordered the
        frame axis would keep them. The channels are keyed to the frame index, so
        this compares the rows the tracker would actually be handed.
        """
        payload = _cache(control_dt=_CACHE_DT, num_frames=_NUM_FRAMES)
        players = []
        for route, build in (("dict", _as_dict), ("npz", _as_npz), ("pt", _as_pt)):
            here = tmp_path / route
            here.mkdir()
            players.append(MotionPlayer(build(payload, here), control_dt=_ARG_DT))

        for frame in range(_NUM_FRAMES):
            states = [p.get_state_at_frame(frame) for p in players]
            for channel in states[0]:
                for other in states[1:]:
                    np.testing.assert_array_equal(states[0][channel], other[channel])
            assert states[0]["dof_pos"][0] == pytest.approx(float(frame))


class TestARefusalStatesTheWholeProblemOnEveryRoute:
    """A cache the format cannot accept is refused the same way however it arrived."""

    @every_route
    def test_a_cache_short_of_channels_names_every_missing_one(self, build: Route, tmp_path: Path) -> None:
        """Pre-fix the ``.npz`` route reported NumPy's ``KeyError: 'dof_vel'`` - one name, no context.

        A caller fixing a hand-built cache needs the whole list: repairing one
        channel per load is what the single-name refusal cost.
        """
        short = {k: v for k, v in _cache(control_dt=_CACHE_DT, num_frames=_NUM_FRAMES).items() if k.startswith("body_")}

        message = _refusal(build(short, tmp_path), KeyError)

        for channel in ("dof_pos", "dof_vel"):
            assert channel in message, (channel, message)
        assert "missing required keys" in message, message

    @every_route
    def test_a_cache_declaring_a_count_its_rows_cannot_serve_is_refused(self, build: Route, tmp_path: Path) -> None:
        """The frame index is clamped to the declared count, so it cannot outrun the rows."""
        lying = _cache(control_dt=_CACHE_DT, num_frames=_NUM_FRAMES + 40)

        message = _refusal(build(lying, tmp_path), ValueError)

        assert str(_NUM_FRAMES + 40) in message and str(_NUM_FRAMES) in message, message

    @every_route
    @pytest.mark.parametrize("period", [0.0, -0.02, float("nan"), float("inf")])
    def test_a_cache_stating_an_unplayable_period_is_refused(self, build: Route, period: float, tmp_path: Path) -> None:
        """A cache's own period outranks the argument, so a good argument must not rescue it."""
        message = _refusal(build(_cache(control_dt=period, num_frames=_NUM_FRAMES), tmp_path), ValueError)

        assert "control_dt" in message, message


class TestARawMotionIsStillResampled:
    """The cache discriminator must not claim a raw ProtoMotions motion."""

    @staticmethod
    def _raw_rows() -> dict[str, np.ndarray]:
        frames = np.arange(_RAW_FRAMES, dtype=np.float32)
        return {
            "body_pos": np.tile(frames[:, None, None], (1, _NUM_BODIES, 3)),
            "body_rot": np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (_RAW_FRAMES, _NUM_BODIES, 1)),
            "body_vel": np.zeros((_RAW_FRAMES, _NUM_BODIES, 3), dtype=np.float32),
            "body_ang_vel": np.zeros((_RAW_FRAMES, _NUM_BODIES, 3), dtype=np.float32),
            "dof_pos": np.tile(frames[:, None], (1, _NUM_DOFS)),
            "dof_vel": np.zeros((_RAW_FRAMES, _NUM_DOFS), dtype=np.float32),
        }

    @needs_real_torch
    def test_a_raw_single_motion_is_resampled_onto_the_control_rate(self, tmp_path: Path) -> None:
        """Read as a cache instead, the clip would keep its 4 source frames at 10 Hz."""
        rows = self._raw_rows()
        payload: dict[str, Any] = {
            "fps": _RAW_FPS,
            "rigid_body_pos": rows["body_pos"],
            "rigid_body_rot": rows["body_rot"],
            "rigid_body_vel": rows["body_vel"],
            "rigid_body_ang_vel": rows["body_ang_vel"],
            "dof_pos": rows["dof_pos"],
            "dof_vel": rows["dof_vel"],
        }

        player = MotionPlayer(_as_pt(payload, tmp_path), control_dt=_ARG_DT)

        assert player.total_frames == _RESAMPLED_FRAMES
        assert player.control_dt == pytest.approx(_ARG_DT)

    @needs_real_torch
    def test_a_raw_packed_library_is_resampled_onto_the_control_rate(self, tmp_path: Path) -> None:
        """The packed layout names its rotations ``grs``, so it carries no ``body_rot`` either."""
        rows = self._raw_rows()
        payload: dict[str, Any] = {
            "length_starts": np.array([0], dtype=np.int64),
            "motion_num_frames": np.array([_RAW_FRAMES], dtype=np.int64),
            "motion_dt": np.array([1.0 / _RAW_FPS], dtype=np.float32),
            "gts": rows["body_pos"],
            "grs": rows["body_rot"],
            "gvs": rows["body_vel"],
            "gavs": rows["body_ang_vel"],
            "dps": rows["dof_pos"],
            "dvs": rows["dof_vel"],
        }

        player = MotionPlayer(_as_pt(payload, tmp_path), control_dt=_ARG_DT)

        assert player.total_frames == _RESAMPLED_FRAMES
        assert player.control_dt == pytest.approx(_ARG_DT)
