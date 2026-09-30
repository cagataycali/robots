"""SonicDecoder assembles the 994-D observation and applies the action law the C++ deploy loop does.

Reference: NVlabs/GR00T-WholeBodyControl ``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref``
(``policy_parameters.hpp``, ``g1_deploy_onnx_ref.cpp`` GatherHis*/CreatePolicyCommand,
``state_logger.cpp`` GetLatest). Every test runs against a stub session; no ONNX file,
no onnxruntime and no network are touched.
"""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots.policies.wbc_latent import (
    SONIC_ACTION_SCALE,
    SONIC_DEFAULT_ANGLES,
    SONIC_JOINT_NAMES,
    SONIC_KDS,
    SONIC_KPS,
    STANDING_TOKEN,
    SonicDecoder,
    sonic_variant_error,
)
from strands_robots.policies.wbc_latent.constants import (
    HARDWARE_TO_ISAACLAB,
    HISTORY_LEN,
    ISAACLAB_TO_HARDWARE,
    NUM_JOINTS,
    OBS_DIM,
    OBS_LAYOUT,
    TOKEN_DIM,
)


class RecordingSession:
    """A stand-in decoder: records every observation, returns a scripted action."""

    def __init__(self, action: np.ndarray | None = None) -> None:
        self.observations: list[np.ndarray] = []
        self.action = np.zeros(NUM_JOINTS, dtype=np.float32) if action is None else np.asarray(action, np.float32)

    def run(self, output_names, input_feed):
        assert output_names is None
        assert list(input_feed) == ["obs_dict"]
        obs = input_feed["obs_dict"]
        assert obs.dtype == np.float32
        assert obs.shape == (1, OBS_DIM)
        self.observations.append(obs[0].copy())
        return [self.action[None, :].copy()]


def _offsets() -> dict[str, int]:
    out, off = {}, 0
    for name, frames, width in OBS_LAYOUT:
        out[name] = off
        off += frames * width
    return out


OFF = _offsets()
UPRIGHT = np.array([1.0, 0.0, 0.0, 0.0])


def test_permutations_are_inverses_and_names_are_hardware_order():
    a = np.asarray(ISAACLAB_TO_HARDWARE)
    b = np.asarray(HARDWARE_TO_ISAACLAB)
    assert (a[b] == np.arange(NUM_JOINTS)).all()
    assert (b[a] == np.arange(NUM_JOINTS)).all()
    assert SONIC_JOINT_NAMES[0] == "left_hip_pitch_joint"
    assert SONIC_JOINT_NAMES[12] == "waist_yaw_joint"
    assert SONIC_JOINT_NAMES[-1] == "right_wrist_yaw_joint"
    assert OBS_DIM == 64 + 10 * (3 + 29 + 29 + 29 + 3)


def test_gains_match_the_armature_table():
    # STIFFNESS_7520_22 = 0.025101925 * (20 pi)^2 ; hip pitch uses it once, ankle 2 x 5020.
    w2 = (20.0 * np.pi) ** 2
    assert SONIC_KPS[0] == pytest.approx(0.025101925 * w2)
    assert SONIC_KPS[4] == pytest.approx(2.0 * 0.003609725 * w2)
    assert SONIC_KDS[0] == pytest.approx(2.0 * 2.0 * 0.025101925 * 20.0 * np.pi)
    # action scale ignores the 2x gain multiplier: 0.25 * effort / stiffness of the motor type.
    assert SONIC_ACTION_SCALE[4] == pytest.approx(0.25 * 25.0 / (0.003609725 * w2))
    assert SONIC_ACTION_SCALE[20] == pytest.approx(0.25 * 5.0 / (0.00425 * w2))
    assert SONIC_DEFAULT_ANGLES[3] == 0.669 and SONIC_DEFAULT_ANGLES[23] == -0.2


def test_first_tick_observation_layout_and_zero_padding():
    sess = RecordingSession()
    dec = SonicDecoder(session=sess)
    token = np.linspace(-1, 1, TOKEN_DIM)
    q = SONIC_DEFAULT_ANGLES + 0.1 * np.arange(NUM_JOINTS)
    dq = np.full(NUM_JOINTS, 0.5)
    gyro = np.array([0.1, 0.2, 0.3])
    dec.step(token, q, dq, gyro, UPRIGHT)
    obs = sess.observations[0].astype(np.float64)
    # token first
    np.testing.assert_allclose(obs[:TOKEN_DIM], token, atol=1e-6)
    # nine zero frames then the newest frame: oldest first with zero padding
    o = OFF["his_base_angular_velocity_10frame_step1"]
    assert not obs[o : o + 27].any()
    np.testing.assert_allclose(obs[o + 27 : o + 30], gyro, atol=1e-6)
    # joint positions: IsaacLab order, defaults subtracted
    o = OFF["his_body_joint_positions_10frame_step1"]
    newest = obs[o + 9 * NUM_JOINTS : o + 10 * NUM_JOINTS]
    expected = (q - SONIC_DEFAULT_ANGLES)[np.asarray(HARDWARE_TO_ISAACLAB)]
    np.testing.assert_allclose(newest, expected, atol=1e-6)
    assert newest[1] == pytest.approx(0.1 * HARDWARE_TO_ISAACLAB[1])  # net idx 1 <- hw idx 6
    # velocities reordered the same way
    o = OFF["his_body_joint_velocities_10frame_step1"]
    np.testing.assert_allclose(obs[o + 9 * NUM_JOINTS : o + 10 * NUM_JOINTS], dq, atol=1e-6)
    # no actions have been taken yet: the whole last-action history is zero
    o = OFF["his_last_actions_10frame_step1"]
    assert not obs[o : o + 10 * NUM_JOINTS].any()
    # gravity direction of an upright base is straight down in the body frame
    o = OFF["his_gravity_dir_10frame_step1"]
    np.testing.assert_allclose(obs[o + 27 : o + 30], [0.0, 0.0, -1.0], atol=1e-6)


def test_history_is_oldest_first_and_holds_ten_frames():
    sess = RecordingSession()
    dec = SonicDecoder(session=sess)
    q = SONIC_DEFAULT_ANGLES.copy()
    for k in range(13):
        dec.step(np.zeros(TOKEN_DIM), q, np.zeros(NUM_JOINTS), np.array([float(k), 0.0, 0.0]), UPRIGHT)
    obs = sess.observations[-1]
    o = OFF["his_base_angular_velocity_10frame_step1"]
    xs = obs[o : o + 30].reshape(HISTORY_LEN, 3)[:, 0]
    np.testing.assert_allclose(xs, np.arange(3, 13), atol=1e-6)  # ticks 3..12, oldest first


def test_last_action_history_lags_by_one_tick_and_is_in_network_order():
    action = np.arange(NUM_JOINTS, dtype=np.float32) / 100.0
    sess = RecordingSession(action)
    dec = SonicDecoder(session=sess)
    args = (np.zeros(TOKEN_DIM), SONIC_DEFAULT_ANGLES, np.zeros(NUM_JOINTS), np.zeros(3), UPRIGHT)
    dec.step(*args)
    dec.step(*args)
    o = OFF["his_last_actions_10frame_step1"]
    second = sess.observations[1][o : o + 10 * NUM_JOINTS].reshape(HISTORY_LEN, NUM_JOINTS)
    assert not second[:9].any()
    np.testing.assert_allclose(second[9], action, atol=1e-6)  # raw, unpermuted, unscaled


def test_action_law_default_plus_permuted_scaled_action():
    action = np.zeros(NUM_JOINTS, dtype=np.float32)
    # q_target[i] = default[i] + action[isaaclab_to_mujoco[i]] * scale[i] (CreatePolicyCommand), so
    # the network slot that drives hardware joint 4 (left ankle pitch) is ISAACLAB_TO_HARDWARE[4] = 13.
    action[ISAACLAB_TO_HARDWARE[4]] = 1.0
    assert ISAACLAB_TO_HARDWARE[4] == 13
    dec = SonicDecoder(session=RecordingSession(action))
    t = dec.step(np.zeros(TOKEN_DIM), SONIC_DEFAULT_ANGLES, np.zeros(NUM_JOINTS), np.zeros(3), UPRIGHT)
    expected = SONIC_DEFAULT_ANGLES.copy()
    expected[4] += SONIC_ACTION_SCALE[4]
    np.testing.assert_allclose(t, expected, atol=1e-9)
    d = dec.targets_dict()
    assert list(d) == list(SONIC_JOINT_NAMES)
    assert all(isinstance(v, float) for v in d.values())
    assert d["left_ankle_pitch_joint"] == pytest.approx(expected[4])


def test_reset_clears_history_and_counters():
    sess = RecordingSession()
    dec = SonicDecoder(session=sess)
    args = (STANDING_TOKEN, SONIC_DEFAULT_ANGLES, np.ones(NUM_JOINTS), np.ones(3), UPRIGHT)
    for _ in range(3):
        dec.step(*args)
    assert dec.ticks == 3 and dec.last_targets is not None
    dec.reset()
    assert dec.ticks == 0 and dec.last_targets is None
    dec.step(*args)
    obs = sess.observations[-1]
    o = OFF["his_body_joint_velocities_10frame_step1"]
    assert not obs[o : o + 9 * NUM_JOINTS].any()


@pytest.mark.parametrize(
    ("name", "bad"),
    [
        ("token", np.zeros(63)),
        ("q", np.zeros(28)),
        ("dq", np.zeros(30)),
        ("gyro", np.zeros(2)),
        ("quat_wxyz", np.zeros(3)),
    ],
)
def test_wrong_lengths_are_refused_by_name(name, bad):
    dec = SonicDecoder(session=RecordingSession())
    kwargs = {
        "token": np.zeros(TOKEN_DIM),
        "q": SONIC_DEFAULT_ANGLES,
        "dq": np.zeros(NUM_JOINTS),
        "gyro": np.zeros(3),
        "quat_wxyz": UPRIGHT,
    }
    kwargs[name] = bad
    with pytest.raises(ValueError, match=name):
        dec.step(**kwargs)


def test_non_finite_inputs_and_outputs_are_refused():
    dec = SonicDecoder(session=RecordingSession())
    q = SONIC_DEFAULT_ANGLES.copy()
    q[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        dec.step(np.zeros(TOKEN_DIM), q, np.zeros(NUM_JOINTS), np.zeros(3), UPRIGHT)
    bad = np.zeros(NUM_JOINTS, dtype=np.float32)
    bad[7] = np.inf
    dec = SonicDecoder(session=RecordingSession(bad))
    with pytest.raises(RuntimeError, match="non-finite"):
        dec.step(np.zeros(TOKEN_DIM), SONIC_DEFAULT_ANGLES, np.zeros(NUM_JOINTS), np.zeros(3), UPRIGHT)


def test_token_out_of_range_warns_once(caplog):
    dec = SonicDecoder(session=RecordingSession())
    hot = np.zeros(TOKEN_DIM)
    hot[3] = 1.5
    with caplog.at_level("WARNING", logger="strands_robots.policies.wbc_latent.decoder"):
        for _ in range(3):
            dec.step(hot, SONIC_DEFAULT_ANGLES, np.zeros(NUM_JOINTS), np.zeros(3), UPRIGHT)
    hits = [r for r in caplog.records if "exceeds 1.25" in r.getMessage()]
    assert len(hits) == 1


def test_variant_is_a_closed_set():
    assert sonic_variant_error("default") is None
    assert sonic_variant_error("low_latency") is None
    assert "sonic_v1_1" in (sonic_variant_error("v2") or "")
    with pytest.raises(ValueError, match="variant"):
        SonicDecoder(session=RecordingSession(), variant="v2")


def test_directory_without_a_decoder_is_refused_with_the_names(tmp_path):
    from strands_robots.policies.wbc_latent import resolve_decoder_path

    with pytest.raises(FileNotFoundError, match="model_decoder.onnx"):
        resolve_decoder_path(tmp_path)
    f = tmp_path / "model_decoder.onnx"
    f.write_bytes(b"not really onnx")
    assert resolve_decoder_path(tmp_path) == f
    assert resolve_decoder_path(f) == f
    sub = tmp_path / "low_latency"
    sub.mkdir()
    g = sub / "model_decoder.onnx"
    g.write_bytes(b"x")
    assert resolve_decoder_path(tmp_path, variant="low_latency") == g
