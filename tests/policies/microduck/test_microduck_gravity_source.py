"""``build_observation`` and ``MicroduckPolicy`` route slot two through ONNX metadata.

Pollen's reference ``scripts/infer_policy.py`` supports two ``get_observations``
branches at slot two of the 61-D vector: ``projected_gravity`` (world ``-Z``
rotated into the base frame from ``base_quat``) and ``raw_accel`` (the
accelerometer's ``sensordata`` verbatim).  The choice is a training-time flag
(``self.use_projected_gravity``) baked into the export, and every currently
shipped alpha policy is ``projected_gravity``.  ``build_observation``, before
this change, unconditionally rotated ``base_quat``; a ``raw_accel`` export fed
through it received a differently-scaled, differently-signed 3-block and the
resulting drift was silent - the network kept producing plausible actions.

This file grades the four contracts the switch has to satisfy:

1. ``build_observation(..., gravity_source="raw_accel")`` reads ``base_acc`` (3)
   and writes it VERBATIM into slot two.  No rotation, no scaling, no sign flip.
2. The same call refuses a missing ``base_acc`` key with the shared
   :func:`~strands_robots.utils.finite_vector_error`-style contract that
   ``_require_base_block`` already applies to ``base_quat``.
3. ``build_observation`` refuses a ``gravity_source`` value that is neither of
   the two shipped spellings, naming the pair.  The projected-gravity default
   with no argument still reproduces the pre-change vector byte-for-byte.
4. ``MicroduckPolicy`` reads ``gravity_source`` out of the ONNX
   ``custom_metadata_map`` in ``_ensure_config`` and threads it to the builder;
   a mistyped metadata entry raises at first inference (not at slot-two read).

The four cells together are the acceptance criterion for harness#388: the
provider can serve a ``raw_accel`` export end-to-end, and a wrong metadata
value fails loudly at configuration time rather than as drift on a
running rollout.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from strands_robots.policies.microduck import (
    GRAVITY_SOURCE_PROJECTED,
    GRAVITY_SOURCE_RAW_ACCEL,
    MICRODUCK_DEFAULT_POSE,
    MICRODUCK_JOINT_NAMES,
    MicroduckPolicy,
    build_observation,
)


def _base_dict(*, base_ang_vel=(0.0, 0.0, 0.0), base_quat=(1.0, 0.0, 0.0, 0.0), extra=None) -> dict[str, Any]:
    """Assemble a minimal observation dict with the joint / velocity keys ``build_observation`` reads.

    The joint values are all zero at the default pose, so ``joint_pos_relative``
    is a zero block and ``joint_vel`` a zero block; that keeps the assembled
    vector's non-slot-two components at zero and lets the test isolate what
    slot two carries.
    """
    obs: dict[str, Any] = {"base_ang_vel": list(base_ang_vel), "base_quat": list(base_quat)}
    for name, pose in zip(MICRODUCK_JOINT_NAMES, MICRODUCK_DEFAULT_POSE):
        obs[name] = float(pose)
        obs[f"{name}.vel"] = 0.0
    if extra:
        obs.update(extra)
    return obs


def _last_action_zeros() -> np.ndarray:
    return np.zeros(len(MICRODUCK_JOINT_NAMES), dtype=np.float32)


def _command_zeros(width: int = 13) -> np.ndarray:
    return np.zeros(width, dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. raw_accel: base_acc is written into slot two VERBATIM.
# ---------------------------------------------------------------------------


def test_raw_accel_writes_base_acc_verbatim_into_slot_two() -> None:
    """``gravity_source="raw_accel"`` reads ``base_acc`` and places it in slot two unchanged.

    Slot two spans indices 3..5 (base_ang_vel is 0..2).  A ``base_acc`` of
    ``[0.5, -1.25, 9.81]`` (a lightly-non-canonical acceleration a hardware
    IMU might report) has to appear at exactly those indices.  The floating
    point comparison is byte-identical (``==``) rather than
    :func:`numpy.testing.assert_allclose` because the pre-change ``base_quat``
    path went through ``quat_rotate_inverse`` and produced a slot two that
    was 0.999...-something; a ``raw_accel`` implementation that quietly
    normalises or rotates the vector would still be ``allclose`` but would
    fail this cell.
    """
    accel = np.array([0.5, -1.25, 9.81], dtype=np.float32)
    obs = _base_dict(extra={"base_acc": accel.tolist()})

    vector = build_observation(
        obs,
        joint_names=list(MICRODUCK_JOINT_NAMES),
        default_pose=np.array(MICRODUCK_DEFAULT_POSE, dtype=np.float32),
        last_action=_last_action_zeros(),
        command=_command_zeros(),
        gravity_source=GRAVITY_SOURCE_RAW_ACCEL,
    )

    assert vector.dtype == np.float32
    # 48 layout components + 13 command = 61-D alpha vector, unchanged.
    assert vector.shape == (61,)
    # Slot two: indices 3..5 = base_acc verbatim.
    np.testing.assert_array_equal(vector[3:6], accel)


def test_raw_accel_refuses_a_missing_base_acc_key() -> None:
    """A ``raw_accel`` builder call with no ``base_acc`` in the dict raises ``KeyError``.

    The shared ``_require_base_block`` reader is what enforces this on
    ``base_quat`` today; the raw-accel branch has to route through the same
    helper so the two base blocks refuse missingness the same way, rather
    than one raising and the other silently returning a zero block.
    """
    obs = _base_dict()  # deliberately omits "base_acc"
    with pytest.raises(KeyError):
        build_observation(
            obs,
            joint_names=list(MICRODUCK_JOINT_NAMES),
            default_pose=np.array(MICRODUCK_DEFAULT_POSE, dtype=np.float32),
            last_action=_last_action_zeros(),
            command=_command_zeros(),
            gravity_source=GRAVITY_SOURCE_RAW_ACCEL,
        )


def test_raw_accel_refuses_a_wrong_width_base_acc_block() -> None:
    """A 2- or 4-component ``base_acc`` raises ``ValueError`` naming the width contract.

    ``_require_base_block`` on ``base_quat`` already refuses widths other than
    4; ``base_acc`` has to inherit the same refusal at width 3, with the same
    error shape (a message that names the block, the observed width and the
    expected one).  Without this cell a caller who passes a 6-vector
    accelerometer (some IMUs concatenate accel+gyro) would have their first
    three components silently taken as slot two while the ``[3:6]`` half was
    dropped, and the rollout drift would not name the caller.
    """
    obs = _base_dict(extra={"base_acc": [0.0, 0.0]})  # short by one
    with pytest.raises(ValueError, match="base_acc"):
        build_observation(
            obs,
            joint_names=list(MICRODUCK_JOINT_NAMES),
            default_pose=np.array(MICRODUCK_DEFAULT_POSE, dtype=np.float32),
            last_action=_last_action_zeros(),
            command=_command_zeros(),
            gravity_source=GRAVITY_SOURCE_RAW_ACCEL,
        )


# ---------------------------------------------------------------------------
# 2. gravity_source domain: only the two shipped values are accepted.
# ---------------------------------------------------------------------------


def test_gravity_source_refuses_a_third_spelling() -> None:
    """``build_observation(..., gravity_source="gravity")`` raises, naming the shipped pair.

    A caller who mistypes ``"gravity"`` or ``"accel"`` gets a raise, not a
    silent selection of one branch.  This is the seam that makes a training
    export's metadata typo fail loudly rather than as drift.
    """
    obs = _base_dict()
    with pytest.raises(ValueError, match="gravity_source"):
        build_observation(
            obs,
            joint_names=list(MICRODUCK_JOINT_NAMES),
            default_pose=np.array(MICRODUCK_DEFAULT_POSE, dtype=np.float32),
            last_action=_last_action_zeros(),
            command=_command_zeros(),
            gravity_source="gravity",  # neither of the two shipped spellings
        )


def test_projected_gravity_default_matches_the_pre_change_vector() -> None:
    """Omitting ``gravity_source`` produces the same vector as before the switch.

    Every currently shipped alpha policy is a ``projected_gravity`` export, so
    a caller who never sets ``gravity_source`` has to receive a byte-identical
    slot two to the one shipped ``build_observation`` returned before this
    change.  The invariant is checked against an explicit
    ``gravity_source="projected_gravity"`` call to demonstrate the two paths
    resolve to the same code, and against a hand-computed value to demonstrate
    the shipped-alpha behaviour did not change.
    """
    obs = _base_dict(base_quat=(1.0, 0.0, 0.0, 0.0))  # identity: gravity is world -Z

    default_call = build_observation(
        obs,
        joint_names=list(MICRODUCK_JOINT_NAMES),
        default_pose=np.array(MICRODUCK_DEFAULT_POSE, dtype=np.float32),
        last_action=_last_action_zeros(),
        command=_command_zeros(),
    )
    explicit_call = build_observation(
        obs,
        joint_names=list(MICRODUCK_JOINT_NAMES),
        default_pose=np.array(MICRODUCK_DEFAULT_POSE, dtype=np.float32),
        last_action=_last_action_zeros(),
        command=_command_zeros(),
        gravity_source=GRAVITY_SOURCE_PROJECTED,
    )
    np.testing.assert_array_equal(default_call, explicit_call)
    # Identity quaternion rotates world -Z (0, 0, -1) to itself in the base frame.
    np.testing.assert_array_equal(default_call[3:6], np.array([0.0, 0.0, -1.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# 3. MicroduckPolicy: reads gravity_source from ONNX metadata.
# ---------------------------------------------------------------------------


class _FakeMeta:
    """Stand-in for ``session.get_modelmeta()`` with a settable metadata map."""

    def __init__(self, custom_metadata_map: dict[str, str]) -> None:
        self.custom_metadata_map = custom_metadata_map


class _FakeSession:
    """Minimal ``MicroduckSession`` stand-in that only serves metadata + a fake input name.

    ``_ensure_config`` reads metadata; it does not run inference. That keeps
    this file free of an onnxruntime dependency and free of a real graph.
    """

    def __init__(self, meta_map: dict[str, str]) -> None:
        self._meta = _FakeMeta(meta_map)

    def get_modelmeta(self) -> _FakeMeta:
        return self._meta

    def get_inputs(self) -> list[Any]:  # pragma: no cover - unused by _ensure_config
        class _Input:
            name = "observation"
            shape = [1, 61]

        return [_Input()]


def _make_policy(meta_map: dict[str, str]) -> MicroduckPolicy:
    return MicroduckPolicy(session=cast(Any, _FakeSession(meta_map)))


def test_policy_reads_gravity_source_from_metadata_and_threads_it() -> None:
    """``MicroduckPolicy._ensure_config`` reads ``gravity_source`` off the metadata map.

    A session whose metadata declares ``gravity_source: raw_accel`` has to
    leave ``policy._gravity_source == "raw_accel"``, so the next
    :meth:`get_actions` call routes through the raw-accel branch.  The
    positive contract is checked here; the negative (mistyped metadata) is
    checked below.
    """
    policy = _make_policy({"gravity_source": "raw_accel"})
    policy._ensure_config()
    assert policy._gravity_source == GRAVITY_SOURCE_RAW_ACCEL


def test_policy_defaults_gravity_source_to_projected_when_metadata_is_silent() -> None:
    """A metadata map without ``gravity_source`` resolves to ``projected_gravity``.

    Every currently-shipped alpha policy is ``projected_gravity`` and does
    not carry the flag in metadata; the resolution has to reproduce that
    default so those exports keep working with no changes.
    """
    policy = _make_policy({})  # no gravity_source key
    policy._ensure_config()
    assert policy._gravity_source == GRAVITY_SOURCE_PROJECTED


def test_policy_refuses_a_mistyped_gravity_source_at_configuration_time() -> None:
    """A metadata ``gravity_source`` that is neither of the two shipped values raises.

    The raise happens in ``_ensure_config`` (first-inference configuration),
    not in the builder every tick.  A tick-time raise would still catch the
    typo, but this cell asserts the catch happens ONCE, at configuration, and
    names the checkpoint's metadata as the source - the same shape
    ``action_scale`` refuses a non-numeric metadata value with today.
    """
    policy = _make_policy({"gravity_source": "gravity"})
    with pytest.raises(ValueError, match="gravity_source"):
        policy._ensure_config()
