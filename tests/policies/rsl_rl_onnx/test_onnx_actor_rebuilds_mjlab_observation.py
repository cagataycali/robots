"""The rsl_rl_onnx provider rebuilds an mjlab actor's observation from engine keys.

A tiny ONNX actor is written here with the same metadata mjlab stamps
(``mjlab/rl/exporter_utils.py``), so the test owns its fixture: it checks the
term order, the frame conventions and the JointPositionAction decode against
hand-computed values, with no training run involved.
"""

from __future__ import annotations

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from tests.policies.rsl_rl_onnx.actor_fixture import JOINTS  # noqa: E402
from tests.policies.rsl_rl_onnx.actor_fixture import write_actor as _write_actor


def test_locomotion_terms_and_frames(tmp_path):
    from strands_robots.policies.rsl_rl_onnx import RslRlOnnxPolicy

    terms = ["base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command"]
    p = _write_actor(tmp_path / "a.onnx", terms, 3 + 3 + 3 + 2 + 2 + 2 + 3)
    pol = RslRlOnnxPolicy(onnx_path=p, command=[0.3, 0.0, 0.1])
    # Base yawed 90 deg about z: world +x velocity reads as body -y.
    q = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
    obs = {
        "a": 0.6,
        "a.vel": 1.0,
        "b": -0.2,
        "b.vel": -2.0,
        "base_quat": q,
        "base_lin_vel": [1.0, 0, 0],
        "base_ang_vel": [0, 0, 0.5],
    }
    vec = pol.build_observation(obs)
    assert vec.shape == (18,)
    np.testing.assert_allclose(vec[0:3], [0.0, -1.0, 0.0], atol=1e-6)  # lin vel rotated into base
    np.testing.assert_allclose(vec[3:6], [0.0, 0.0, 0.5], atol=1e-6)  # ang vel already body frame
    np.testing.assert_allclose(vec[6:9], [0.0, 0.0, -1.0], atol=1e-6)  # gravity unchanged by yaw
    np.testing.assert_allclose(vec[9:11], [0.5, 0.0], atol=1e-6)  # joint_pos minus default
    np.testing.assert_allclose(vec[11:13], [1.0, -2.0], atol=1e-6)
    np.testing.assert_allclose(vec[13:15], [0.0, 0.0])  # no previous action after construction
    np.testing.assert_allclose(vec[15:18], [0.3, 0.0, 0.1])
    # Decode: target = default + scale * raw; raw = (vec[0], vec[1]) = (0, -1).
    act = pol.get_actions_sync(obs, "")[0]
    assert list(act) == JOINTS
    np.testing.assert_allclose([act["a"], act["b"]], [0.1 + 0.5 * 0.0, -0.2 + 0.25 * -1.0], atol=1e-6)
    # The raw action is remembered for the next tick's ``actions`` term, cleared by reset.
    np.testing.assert_allclose(pol.build_observation(obs)[13:15], [0.0, -1.0], atol=1e-6)
    pol.reset()
    np.testing.assert_allclose(pol.build_observation(obs)[13:15], [0.0, 0.0])
    # target_velocity kwarg overrides the configured command.
    np.testing.assert_allclose(pol.build_observation(obs, target_velocity=[1.0, 0.0, 0.0])[15:18], [1.0, 0.0, 0.0])


def test_unknown_term_and_missing_joint_are_refused_by_name(tmp_path):
    from strands_robots.policies.rsl_rl_onnx import RslRlOnnxPolicy

    p = _write_actor(tmp_path / "bad.onnx", ["joint_pos", "height_scan"], 2 + 4)
    with pytest.raises(ValueError, match="height_scan"):
        RslRlOnnxPolicy(onnx_path=p)
    p = _write_actor(tmp_path / "ok.onnx", ["joint_pos", "actions"], 4)
    pol = RslRlOnnxPolicy(onnx_path=p)
    with pytest.raises(ValueError, match="omits joints.*'b'"):
        pol.build_observation({"a": 0.0})
    with pytest.raises(ValueError, match="onnx_path is required"):
        RslRlOnnxPolicy()


def test_registered_with_the_policy_factory():
    from strands_robots.policies.factory import list_providers

    assert "rsl_rl_onnx" in list_providers()
