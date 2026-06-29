"""Live integration test for the MotionBricks generator - gated, needs checkpoints + GPU/CPU.

Skipped unless ``MOTIONBRICKS_CKPT`` points at the upstream ``out/`` checkpoint
tree (``motionbricks_pose`` / ``motionbricks_root`` / ``motionbricks_vqvae`` +
``G1-clip.ckpt``, fetched with git-LFS) AND the ``motionbricks`` package is
importable. Run with::

    MOTIONBRICKS_CKPT=/path/to/GR00T-WholeBodyControl/motionbricks/out \
        pytest -m motionbricks tests_integ/policies/motionbricks/

This exercises the real generator end to end (the path the unit tests stub out):
build from checkpoints, roll out >= 100 control steps through a style sequence,
and assert the synthesised joint motion is non-trivial and finite.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

_CKPT = os.environ.get("MOTIONBRICKS_CKPT", "")
_HAS_MB = importlib.util.find_spec("motionbricks") is not None

pytestmark = [
    pytest.mark.motionbricks,
    pytest.mark.skipif(not _CKPT, reason="set MOTIONBRICKS_CKPT to the MotionBricks out/ checkpoint tree"),
    pytest.mark.skipif(not _HAS_MB, reason="motionbricks package not installed ([motionbricks] extra + git install)"),
]

# A CPU run is correct (just slower); default to it so the test does not contend
# for GPU memory. Override with MOTIONBRICKS_DEVICE=cuda.
_DEVICE = os.environ.get("MOTIONBRICKS_DEVICE", "cpu")


@pytest.fixture(scope="module")
def policy():  # type: ignore[no-untyped-def]
    from strands_robots.policies.motionbricks import MotionBricksConfig, MotionBricksPolicy

    config = MotionBricksConfig(result_dir=_CKPT, device=_DEVICE)
    return MotionBricksPolicy(config=config, style="walk")


def test_live_rollout_produces_finite_nonzero_motion(policy) -> None:  # type: ignore[no-untyped-def]
    """>= 100 steps across styles -> finite joint targets with real motion."""
    policy.reset()
    styles = ["walk", "stealth_walk", "walk_boxing"]
    n_per = 40  # 120 steps total (>= 100)
    joint_log: list[list[float]] = []
    root_log: list[np.ndarray] = []
    for style in styles:
        for _ in range(n_per):
            actions = policy.get_actions_sync({}, "", style=style)
            assert len(actions) == 1
            act = actions[0]
            assert len(act) == 29
            values = list(act.values())
            assert all(np.isfinite(v) for v in values), "non-finite joint target emitted"
            joint_log.append(values)
            qpos = policy.last_qpos
            assert qpos is not None and qpos.shape[0] == 7 + 29
            root_log.append(qpos[:3].copy())

    arr = np.asarray(joint_log)
    assert arr.shape[0] >= 100
    assert np.isfinite(arr).all()
    # Joints actually move over the rollout (not a frozen / zero output).
    per_joint_std = arr.std(axis=0)
    assert float(per_joint_std.mean()) > 1e-3, f"joint motion too small: {per_joint_std.mean()}"
    # The root translates across the ground (the character walks somewhere).
    roots = np.asarray(root_log)
    assert float(np.linalg.norm(roots[-1, :2] - roots[0, :2])) > 0.1, "root did not translate"


def test_live_unknown_style_raises(policy) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="not a known mode"):
        policy.get_actions_sync({}, "", style="does_not_exist")
