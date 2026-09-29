"""The non-finite refusal is the only thing a caller hears about an ``inf`` orientation.

:func:`strands_robots.policies.microduck.observation.quat_rotate_inverse`
normalises the orientation it reads and, by its own contract, lets a ``nan`` or
``inf`` component propagate so that the assembled-vector pass in
:func:`build_observation` refuses it by the block it becomes (``ValueError``,
"non-finite"). Measured on the pre-fix tree with ``base_quat = [1, inf, 0, 0]``:
the norm is ``inf``, ``inf / inf`` is ``nan``, and NumPy reported the division
as ``RuntimeWarning: invalid value encountered in divide`` before the
``ValueError`` was raised. Under ``-W error`` - the setting that turns every
warning into a finding - the caller then saw a ``RuntimeWarning`` naming a
divide in an internal helper instead of the documented refusal naming their
observation. The propagation is deliberate; the warning was not.

The ``nan`` spelling never warned (``nan / nan`` is a quiet ``nan``), which is
why the pre-fix suite, run without ``-W error``, stayed green.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.microduck import build_observation

_N_JOINTS = 14
_ALPHA_COMMAND_WIDTH = 13


def _build(base_quat: list[float]) -> np.ndarray:
    joints = [f"j{i}" for i in range(_N_JOINTS)]
    observation: dict[str, Any] = {}
    for name in joints:
        observation[name] = 0.1
        observation[f"{name}.vel"] = 0.0
    observation["base_ang_vel"] = [0.0, 0.0, 0.0]
    observation["base_quat"] = base_quat
    return build_observation(
        observation,
        joint_names=joints,
        default_pose=np.zeros(_N_JOINTS, np.float32),
        last_action=np.zeros(_N_JOINTS, np.float32),
        command=np.zeros(_ALPHA_COMMAND_WIDTH, np.float32),
    )


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")], ids=["inf", "-inf", "nan"])
def test_the_refusal_arrives_without_a_runtime_warning(bad: float) -> None:
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="non-finite"):
            _build([1.0, bad, 0.0, 0.0])
    runtime = [r for r in records if issubclass(r.category, RuntimeWarning)]
    assert not runtime, "the non-finite refusal was preceded by: " + "; ".join(
        f"{r.filename}:{r.lineno}: {r.message}" for r in runtime
    )


def test_a_finite_orientation_still_normalises() -> None:
    # The silenced state is scoped to the one division; a scaled unit quaternion
    # answers as the unit one it encodes, with the gravity block a unit vector.
    vector = _build([2.0, 0.0, 0.0, 0.0])
    gravity = vector[3:6]
    assert np.allclose(np.linalg.norm(gravity), 1.0, atol=1e-6)
    assert np.allclose(gravity, [0.0, 0.0, -1.0], atol=1e-6)
