"""Tests for the Pollen Microduck locomotion policy provider.

Two layers:

* **Unit** - drive the whole obs-build / infer / decode / last-action pipeline
  through an INJECTED stub session, so no ``onnxruntime`` / ``mujoco`` is needed.
* **Real weights** (``TestRealWeights``) - byte-compatibility against a raw
  onnxruntime session and a real MuJoCo rollout, skipped unless Pollen's shipped
  ``alpha_walking.onnx`` and the optional deps are present.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from strands_robots.policies.microduck import (
    MICRODUCK_DEFAULT_POSE,
    MICRODUCK_JOINT_NAMES,
    MicroduckPolicy,
    MicroduckPolicyBundle,
)
from strands_robots.policies.microduck.observation import build_observation, decode_action

# Path to Pollen's shipped walking weights, relative to the repo root.
_ONNX = Path(__file__).resolve().parents[3].parent / "microduck" / "policies" / "alpha_walking.onnx"


class _Meta:
    def __init__(self, m: dict[str, str]) -> None:
        self.custom_metadata_map = m


class _StubSession:
    """A deterministic ONNX stand-in: echoes the obs tail, self-describes."""

    def __init__(self, *, meta: dict[str, str] | None = None, n_joints: int = 14) -> None:
        self._meta = (
            meta
            if meta is not None
            else {
                "joint_names": ",".join(MICRODUCK_JOINT_NAMES),
                "default_joint_pos": ",".join(f"{v}" for v in MICRODUCK_DEFAULT_POSE),
                "action_scale": "1.0",
                "command_names": "twist,head_pose,body_pose",
            }
        )
        self._n = n_joints
        self.last_input: np.ndarray | None = None

    def run(self, output_names, input_feed):
        self.last_input = next(iter(input_feed.values())).copy()
        # Return a fixed non-trivial action so decode/last_action are observable.
        act = np.arange(self._n, dtype=np.float32) * 0.01
        return [act.reshape(1, -1)]

    def get_modelmeta(self):
        return _Meta(self._meta)


def _obs_dict(pos=0.0, vel=0.0):
    d = {name: float(pos) for name in MICRODUCK_JOINT_NAMES}
    d.update({f"{name}.vel": float(vel) for name in MICRODUCK_JOINT_NAMES})
    d["base_ang_vel"] = [0.0, 0.0, 0.0]
    d["base_quat"] = [1.0, 0.0, 0.0, 0.0]  # identity -> gravity = [0,0,-1]
    return d


class TestObservation:
    def test_vector_width_and_layout(self):
        default = np.array(MICRODUCK_DEFAULT_POSE, dtype=np.float32)
        v = build_observation(
            _obs_dict(pos=0.0, vel=0.0),
            joint_names=list(MICRODUCK_JOINT_NAMES),
            default_pose=default,
            last_action=np.zeros(14, dtype=np.float32),
            command=np.zeros(13, dtype=np.float32),
        )
        assert v.shape == (61,)
        assert v.dtype == np.float32
        # base_ang_vel(3) zero, projected_gravity(3) = [0,0,-1] for identity quat
        np.testing.assert_allclose(v[:3], [0, 0, 0])
        np.testing.assert_allclose(v[3:6], [0, 0, -1], atol=1e-6)
        # joint_pos block is (pos - default) = -default here
        np.testing.assert_allclose(v[6:20], -default, atol=1e-6)

    def test_command_width_parameterized(self):
        v = build_observation(
            _obs_dict(),
            joint_names=list(MICRODUCK_JOINT_NAMES),
            default_pose=np.array(MICRODUCK_DEFAULT_POSE, dtype=np.float32),
            last_action=np.zeros(14, dtype=np.float32),
            command=np.zeros(3, dtype=np.float32),  # legacy twist-only
        )
        assert v.shape == (51,)

    def test_decode_is_default_plus_scaled_action(self):
        default = np.array(MICRODUCK_DEFAULT_POSE, dtype=np.float32)
        raw = np.ones(14, dtype=np.float32)
        out = decode_action(raw, default_pose=default, action_scale=2.0)
        np.testing.assert_allclose(out, default + 2.0, atol=1e-6)


class TestPolicyUnit:
    def test_requires_session_or_path(self):
        with pytest.raises(ValueError):
            MicroduckPolicy()

    def test_provider_name(self):
        assert MicroduckPolicy(session=_StubSession()).provider_name == "microduck"

    def test_requires_images_false(self):
        assert MicroduckPolicy(session=_StubSession()).requires_images is False

    def test_autoconfig_from_metadata(self):
        p = MicroduckPolicy(session=_StubSession())
        p._ensure_config()
        assert p._joint_names == list(MICRODUCK_JOINT_NAMES)
        assert p._action_scale == 1.0
        assert p._command_width() == 13
        np.testing.assert_allclose(p._default_pose, MICRODUCK_DEFAULT_POSE, atol=1e-6)

    def test_get_actions_decodes_and_tracks_last_action(self):
        stub = _StubSession()
        p = MicroduckPolicy(session=stub)
        out = asyncio.run(p.get_actions(_obs_dict(), ""))
        assert isinstance(out, list) and len(out) == 1
        act = out[0]
        assert set(act) == set(MICRODUCK_JOINT_NAMES)
        # motor_target = default + rawaction*scale ; raw = i*0.01
        default = dict(zip(MICRODUCK_JOINT_NAMES, MICRODUCK_DEFAULT_POSE))
        for i, name in enumerate(MICRODUCK_JOINT_NAMES):
            assert act[name] == pytest.approx(default[name] + i * 0.01, abs=1e-5)
        # last_action recorded as the RAW action (not the motor target)
        np.testing.assert_allclose(p._last_action, np.arange(14) * 0.01, atol=1e-6)

    def test_last_action_feeds_next_obs(self):
        stub = _StubSession()
        p = MicroduckPolicy(session=stub)
        asyncio.run(p.get_actions(_obs_dict(), ""))
        asyncio.run(p.get_actions(_obs_dict(), ""))
        # obs layout: [0:3] ang_vel, [3:6] grav, [6:20] jpos, [20:34] jvel,
        # [34:48] last_action -> the previous tick's raw action.
        fed_last_action = stub.last_input.reshape(-1)[34:48]
        np.testing.assert_allclose(fed_last_action, np.arange(14) * 0.01, atol=1e-6)

    def test_feeds_raw_observation_no_renormalisation(self):
        stub = _StubSession()
        p = MicroduckPolicy(session=stub)
        obs = _obs_dict(pos=0.1, vel=0.2)
        asyncio.run(p.get_actions(obs, ""))
        # joint_pos block must equal (0.1 - default) exactly - no scaling/centering.
        default = np.array(MICRODUCK_DEFAULT_POSE, dtype=np.float32)
        fed_jpos = stub.last_input.reshape(-1)[6:20]
        np.testing.assert_allclose(fed_jpos, 0.1 - default, atol=1e-6)

    def test_target_velocity_writes_twist_slots(self):
        stub = _StubSession()
        p = MicroduckPolicy(session=stub)
        asyncio.run(p.get_actions(_obs_dict(), "", target_velocity=[0.5, -0.2, 0.1]))
        cmd = stub.last_input.reshape(-1)[48:61]
        np.testing.assert_allclose(cmd[:3], [0.5, -0.2, 0.1], atol=1e-6)
        np.testing.assert_allclose(cmd[3:], 0.0, atol=1e-6)  # dead weight stays zero

    def test_set_robot_state_keys_rejects_bare_string(self):
        p = MicroduckPolicy(session=_StubSession())
        with pytest.raises(ValueError):
            p.set_robot_state_keys("left_hip_yaw")  # type: ignore[arg-type]

    def test_set_robot_state_keys_accepts_full_set_any_order(self):
        p = MicroduckPolicy(session=_StubSession())
        p.set_robot_state_keys(list(reversed(MICRODUCK_JOINT_NAMES)))  # no raise


class TestBundle:
    def _bundle(self):
        return MicroduckPolicyBundle(
            {"walk": MicroduckPolicy(session=_StubSession()), "stand": MicroduckPolicy(session=_StubSession())},
            active="stand",
        )

    def test_provider_name_and_children(self):
        b = self._bundle()
        assert b.provider_name == "microduck_bundle"
        assert len(b.children) == 2

    def test_switch_selects_active(self):
        b = self._bundle()
        assert b.active == "stand"
        b.switch("walk")
        assert b.active == "walk"
        with pytest.raises(ValueError):
            b.switch("nope")

    def test_get_actions_delegates_and_select_switches(self):
        b = self._bundle()
        out = asyncio.run(b.get_actions(_obs_dict(), "", select="walk"))
        assert b.active == "walk"
        assert set(out[0]) == set(MICRODUCK_JOINT_NAMES)

    def test_velocity_gate_auto_switches(self):
        b = MicroduckPolicyBundle(
            {"walk": MicroduckPolicy(session=_StubSession()), "stand": MicroduckPolicy(session=_StubSession())},
            active="stand",
            switch_on_velocity=0.1,
        )
        asyncio.run(b.get_actions(_obs_dict(), "", target_velocity=[0.5, 0, 0]))
        assert b.active == "walk"
        asyncio.run(b.get_actions(_obs_dict(), "", target_velocity=[0.0, 0, 0]))
        assert b.active == "stand"

    def test_rejects_non_microduck_policy(self):
        with pytest.raises(TypeError):
            MicroduckPolicyBundle({"x": object()})  # type: ignore[dict-item]


class TestRegistry:
    def test_resolves_from_registry(self):
        from strands_robots.registry.policies import import_policy_class

        assert import_policy_class("microduck") is MicroduckPolicy


@pytest.mark.skipif(not _ONNX.exists(), reason="Pollen alpha_walking.onnx not present")
class TestRealWeights:
    def test_byte_compat_with_raw_session(self):
        ort = pytest.importorskip("onnxruntime")
        sess = ort.InferenceSession(str(_ONNX))
        inn = sess.get_inputs()[0].name
        v = np.random.default_rng(0).standard_normal(61).astype(np.float32)
        ref = sess.run(None, {inn: v.reshape(1, -1)})[0].squeeze(0).astype(np.float32)

        mine = MicroduckPolicy(onnx_path=str(_ONNX)).infer_raw(v)
        assert float(np.max(np.abs(ref - mine))) < 1e-6

    def test_real_mujoco_rollout_moves_joints(self):
        pytest.importorskip("onnxruntime")
        pytest.importorskip("mujoco")
        from strands_robots import Robot

        robot = Robot("microduck")
        robot.reset()
        watch = ["left_knee", "right_knee", "head_pitch"]
        before = np.array([robot.get_observation()[k] for k in watch], dtype=float)
        robot.run_policy(
            policy_object=MicroduckPolicy(onnx_path=str(_ONNX)),
            control_frequency=50,
            duration=3.0,
            policy_kwargs={"target_velocity": [0.3, 0.0, 0.0]},
        )
        after = np.array([robot.get_observation()[k] for k in watch], dtype=float)
        assert float(np.max(np.abs(after - before))) > 1e-3
