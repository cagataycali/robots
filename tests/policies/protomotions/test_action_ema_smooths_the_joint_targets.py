# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A declared smoothing factor smooths the targets, and an unusable one is refused.

``ProtoMotionsConfig.action_ema_alpha`` is parsed from the checkpoint's own
``unified_pipeline.yaml`` (``control.action_ema_alpha``), stored on the frozen
config, and documented as "exponential-moving-average smoothing on the joint
target output". Measured against the shipped policy before the fix, the emitted
targets were byte-identical for ``alpha`` of ``1.0``, ``0.5`` and ``0.2`` - mean
per-tick change ``0.24000`` in all three cases - because nothing outside
``config.py`` ever read the field. A caller who dialled smoothing in to tame
per-tick jitter on a 29-DOF humanoid got a config that reported the value back
and a PD loop that received the raw network output.

Because the field had no reader it also had no domain, so it accepted every
value that cannot be a weight in ``y[t] = a*x[t] + (1-a)*y[t-1]``:

* ``0`` weights the current network output at zero, freezing the commanded pose
  at the first tick's target for the whole clip - a tracker that reports every
  frame and moves through none of them.
* A negative weight drives each joint the opposite way from the motion.
* Above ``1`` the previous target carries a negative weight, extrapolating past
  the motion rather than smoothing toward it.
* ``nan`` enters the filter state and never leaves it, so every joint of every
  later tick is ``nan`` however good the network output is.

The controls below are as load-bearing as the refusals: ``1.0`` (the shipped
checkpoint's own value) must stay bit-exact passthrough, the first tick must
seed from the network's output rather than from zeros, and the
historical-actions buffer must keep carrying the RAW output - it feeds the ONNX
graph's ``historical_processed_actions`` input, which is defined over it, so
smoothing what the network reads back would change its input distribution.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from strands_robots.policies.protomotions import (
    MotionPlayer,
    ProtoMotionsConfig,
    ProtoMotionsPolicy,
)

NUM_BODIES, NUM_DOFS = 33, 29
JOINT = "left_hip_pitch_joint"  # index 0 of GTP_G1_JOINT_NAMES


class _JitterSession:
    """A tracker whose joint-0 output is a slow ramp plus alternating jitter.

    The two components are what a smoothing factor has to tell apart: the ramp
    is the motion being tracked, the alternating term is the per-tick noise.
    ``raw`` records every output so a test can assert the emitted trace against
    the EMA recursion over exactly what the network produced.
    """

    def __init__(self) -> None:
        self.tick = 0
        self.raw: list[float] = []
        self.history_inputs: list[np.ndarray] = []

    def run(self, output_names: list[str] | None, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.history_inputs.append(inputs["historical_processed_actions"].copy())
        value = 0.3 * np.sin(2 * np.pi * self.tick / 40.0) + 0.12 * (1 if self.tick % 2 == 0 else -1)
        self.tick += 1
        self.raw.append(float(np.float32(value)))
        vec = np.zeros(NUM_DOFS, dtype=np.float32)
        vec[0] = np.float32(value)
        return [
            vec.reshape(1, NUM_DOFS),
            vec.reshape(1, NUM_DOFS),
            np.full((1, NUM_DOFS), 40.0, dtype=np.float32),
            np.full((1, NUM_DOFS), 2.5, dtype=np.float32),
        ]


def _flat_motion(num_frames: int = 200) -> MotionPlayer:
    return MotionPlayer(
        {
            "dof_pos": np.zeros((num_frames, NUM_DOFS), dtype=np.float32),
            "dof_vel": np.zeros((num_frames, NUM_DOFS), dtype=np.float32),
            "body_rot": np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (num_frames, NUM_BODIES, 1)),
            "body_pos": np.zeros((num_frames, NUM_BODIES, 3), dtype=np.float32),
            "body_vel": np.zeros((num_frames, NUM_BODIES, 3), dtype=np.float32),
            "body_ang_vel": np.zeros((num_frames, NUM_BODIES, 3), dtype=np.float32),
            "control_dt": 0.02,
            "num_frames": num_frames,
        }
    )


def _sidecar(tmp_path: Path, declared: str) -> str:
    """Write a sidecar declaring ``control.action_ema_alpha``, the caller's route."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "unified_pipeline.yaml"
    path.write_text(f"control:\n  action_ema_alpha: {declared}\n", encoding="utf-8")
    return str(path)


def _tick(policy: ProtoMotionsPolicy) -> float:
    """Drive one control tick and return the joint-0 target the PD loop receives."""
    actions = asyncio.run(
        policy.get_actions(
            {"observation.state": [0.0] * NUM_DOFS},
            "",
            anchor_rot_xyzw=[0.0, 0.0, 0.0, 1.0],
            root_ang_vel_local=[0.0, 0.0, 0.0],
            dof_pos=[0.0] * NUM_DOFS,
            dof_vel=[0.0] * NUM_DOFS,
        )
    )
    return actions[0][JOINT]


def _run(alpha: str, tmp_path: Path, ticks: int = 60) -> tuple[list[float], _JitterSession]:
    session = _JitterSession()
    policy = ProtoMotionsPolicy(session=session, yaml_path=_sidecar(tmp_path, alpha), motion=_flat_motion())
    return [_tick(policy) for _ in range(ticks)], session


def _ema(raw: list[float], alpha: float) -> list[float]:
    """The closed-form recursion, seeded from the first raw value."""
    out = [raw[0]]
    for value in raw[1:]:
        out.append(float(np.float32(alpha * np.float32(value) + (1.0 - alpha) * np.float32(out[-1]))))
    return out


# ---------------------------------------------------------------------------
# The declared factor reaches the targets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.2, 0.05])
def test_declared_alpha_smooths_the_emitted_targets(tmp_path: Path, alpha: float) -> None:
    """Every declared factor produces the EMA of what the network emitted.

    ``1.0`` is included as the passthrough control: the recursion collapses to
    the raw trace there, so one assertion covers both the honoured factors and
    the default that must not change.
    """
    emitted, session = _run(str(alpha), tmp_path / str(alpha))
    assert emitted == pytest.approx(_ema(session.raw, alpha), abs=1e-6)


def test_a_smaller_alpha_admits_less_per_tick_jitter(tmp_path: Path) -> None:
    """The factor's purpose, measured: less weight on the current output, less jitter.

    Pre-fix all four traces were byte-identical, so the ordering below was
    ``0.24 < 0.24 < 0.24 < 0.24`` and could not hold.
    """
    jitter = {}
    for alpha in (1.0, 0.5, 0.2, 0.05):
        emitted, _ = _run(str(alpha), tmp_path / f"j{alpha}")
        jitter[alpha] = float(np.abs(np.diff(emitted)).mean())
    assert jitter[1.0] > jitter[0.5] > jitter[0.2] > jitter[0.05]


def test_the_first_tick_seeds_from_the_network_not_from_zero(tmp_path: Path) -> None:
    """A zero-seeded filter would command a pose between the origin and the target.

    On a humanoid holding a stance that first command is a lurch toward the zero
    pose, and the smaller the alpha the further it sits from the motion. Control:
    green before the fix too, since there was no filter to seed.
    """
    emitted, session = _run("0.05", tmp_path)
    assert emitted[0] == pytest.approx(session.raw[0], abs=1e-6)


# ---------------------------------------------------------------------------
# Filter state is episode-scoped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("boundary", ["reset", "load_motion"])
def test_the_filter_state_does_not_cross_an_episode_boundary(tmp_path: Path, boundary: str) -> None:
    """Two episodes of one clip emit the same trace, however the boundary is spelled.

    Carrying the previous tick's smoothed target across would blend the last pose
    of one episode into the first tick of the next, which is the failure
    :meth:`ProtoMotionsPolicy.reset` already documents for the playhead.
    """
    session = _JitterSession()
    policy = ProtoMotionsPolicy(session=session, yaml_path=_sidecar(tmp_path, "0.2"), motion=_flat_motion())
    first = [_tick(policy) for _ in range(20)]
    session.tick = 0  # the clip replays, so the network sees the same frames again
    if boundary == "reset":
        policy.reset(seed=0)
    else:
        policy.load_motion(_flat_motion())
    second = [_tick(policy) for _ in range(20)]
    assert second == pytest.approx(first, abs=1e-6)


def test_the_history_buffer_still_carries_the_raw_network_output(tmp_path: Path) -> None:
    """Smoothing the emitted target must not change what the network reads back.

    ``historical_processed_actions`` is defined over the graph's own ``actions``
    output, so the buffer keeps the raw value while the PD loop gets the smoothed
    one. Control: this held before the fix and must still hold.
    """
    emitted, session = _run("0.05", tmp_path, ticks=3)
    fed_back = float(session.history_inputs[1].reshape(-1)[0])
    assert fed_back == pytest.approx(session.raw[0], abs=1e-6)
    assert emitted[1] != pytest.approx(session.raw[1], abs=1e-3)


# ---------------------------------------------------------------------------
# Domain - one weight, two spellings, one verdict
# ---------------------------------------------------------------------------

#: Values that cannot be the weight of the current output in the EMA blend.
UNUSABLE = [
    0,
    0.0,
    -0.5,
    -3.0,
    1.5,
    2,
    float("nan"),
    float("inf"),
    float("-inf"),
    True,
    "0.5",
    None,
    [0.5],
    10**400,
]

#: Values that are, including both ends of the closed upper bound.
USABLE = [1.0, 1, 0.5, 0.05, 1e-9, np.float32(0.25)]


@pytest.mark.parametrize("value", UNUSABLE)
def test_an_unusable_smoothing_factor_is_refused_by_name(value: object) -> None:
    with pytest.raises(ValueError, match="action_ema_alpha"):
        ProtoMotionsConfig(action_ema_alpha=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", USABLE)
def test_a_usable_smoothing_factor_is_stored_as_a_plain_float(value: object) -> None:
    """Accepted, and normalised - the filter multiplies with it every tick, so a
    NumPy scalar would otherwise set the output dtype from the weight."""
    config = ProtoMotionsConfig(action_ema_alpha=value)  # type: ignore[arg-type]
    assert type(config.action_ema_alpha) is float
    assert config.action_ema_alpha == pytest.approx(float(value))  # type: ignore[arg-type]


@pytest.mark.parametrize("declared", ["0.0", "-0.5", "1.5", ".nan", ".inf", "true"])
def test_a_sidecar_reports_the_same_verdict_as_a_config_built_by_hand(tmp_path: Path, declared: str) -> None:
    """The yaml route is the only way a caller sets this field, so it is the route
    that has to refuse. The loader hands the value through raw for the reason the
    body indices are handed through raw: a ``float()`` there would launder a
    ``true`` into an unsmoothed ``1.0`` before the domain could see it."""
    with pytest.raises(ValueError, match="action_ema_alpha"):
        ProtoMotionsPolicy(session=_JitterSession(), yaml_path=_sidecar(tmp_path, declared))
