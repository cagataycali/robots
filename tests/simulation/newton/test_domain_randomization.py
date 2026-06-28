"""Domain randomization and sensor-noise hooks for the Newton backend.

Two layers:

- Pure-helper unit tests (no Newton/Warp required) cover the deterministic
  sampling, range validation, and observation/frame noise application that the
  feature is built on. These run everywhere, including CPU-only CI.
- Integration tests (gated on Newton + Warp) drive the real engine: physics
  randomization mutates the finalized model's mass/friction, the multiplier
  sequence is reproducible for a fixed seed and varies across episodes, lighting
  and colors flow through to rendered frames, and configured sensor noise shows
  up with the requested standard deviation on observations.

The integration tests use the ``featherstone`` solver so they do not require
``mujoco_warp`` (the default ``mujoco`` solver's extra dependency).
"""

from __future__ import annotations

import importlib.util
import threading

import numpy as np
import pytest

from strands_robots.simulation.newton.randomization import (
    DomainRandomizationMixin,
    _validate_range,
)

_HAS_NEWTON = importlib.util.find_spec("newton") is not None and importlib.util.find_spec("warp") is not None


# --------------------------------------------------------------------------- #
# Pure-helper unit tests (no Newton required)
# --------------------------------------------------------------------------- #


class _NoiseHost(DomainRandomizationMixin):
    """Minimal host exposing only the state the noise/validation helpers touch."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._world = None
        self._dr = None
        self._dr_applied = None
        self._dr_light_dir = None
        self._obs_noise = None
        self._obs_noise_rng = None


class TestValidateRange:
    def test_accepts_ordered_non_negative_pair(self):
        assert _validate_range("mass_range", (0.5, 2.0)) is None

    def test_rejects_inverted_bounds(self):
        msg = _validate_range("mass_range", (2.0, 0.5))
        assert msg is not None and "exceeds upper bound" in msg

    def test_rejects_negative_bound(self):
        msg = _validate_range("friction_range", (-0.1, 1.0))
        assert msg is not None and "non-negative" in msg

    def test_rejects_non_numeric(self):
        assert _validate_range("color_range", "nope") is not None

    def test_rejects_non_finite(self):
        assert _validate_range("mass_range", (0.0, float("inf"))) is not None


class TestSetObsNoiseValidation:
    def test_negative_std_is_error(self):
        host = _NoiseHost()
        result = host.set_obs_noise(joint_pos_std=-0.1)
        assert result["status"] == "error"
        assert host._obs_noise is None

    def test_non_finite_is_error(self):
        host = _NoiseHost()
        assert host.set_obs_noise(camera_jitter_px=float("nan"))["status"] == "error"

    def test_valid_config_is_stored(self):
        host = _NoiseHost()
        result = host.set_obs_noise(joint_pos_std=0.02, joint_vel_std=0.1, camera_jitter_px=3, seed=0)
        assert result["status"] == "success"
        assert host._obs_noise == {"joint_pos_std": 0.02, "joint_vel_std": 0.1, "camera_jitter_px": 3.0}
        assert host._obs_noise_rng is not None


class TestJointPosNoise:
    def test_disabled_returns_input_unchanged(self):
        host = _NoiseHost()
        obs = {"Rotation": 0.5, "Pitch": -0.2}
        assert host._apply_joint_pos_noise(obs) is obs

    def test_adds_noise_with_requested_std(self):
        host = _NoiseHost()
        host.set_obs_noise(joint_pos_std=0.05, seed=0)
        samples = [host._apply_joint_pos_noise({"j": 1.0})["j"] for _ in range(20000)]
        assert abs(float(np.std(samples)) - 0.05) < 0.005
        assert abs(float(np.mean(samples)) - 1.0) < 0.005

    def test_is_reproducible_for_same_seed(self):
        a, b = _NoiseHost(), _NoiseHost()
        a.set_obs_noise(joint_pos_std=0.1, seed=7)
        b.set_obs_noise(joint_pos_std=0.1, seed=7)
        seq_a = [a._apply_joint_pos_noise({"j": 0.0})["j"] for _ in range(50)]
        seq_b = [b._apply_joint_pos_noise({"j": 0.0})["j"] for _ in range(50)]
        assert seq_a == seq_b


class TestStateNoise:
    def test_velocity_noise_independent_of_position(self):
        host = _NoiseHost()
        host.set_obs_noise(joint_pos_std=0.0, joint_vel_std=0.2, seed=1)
        states = [host._apply_state_noise({"j": {"position": 0.3, "velocity": 0.0}})["j"] for _ in range(20000)]
        positions = [s["position"] for s in states]
        velocities = [s["velocity"] for s in states]
        assert positions == [0.3] * len(positions)  # pos std == 0 -> untouched
        assert abs(float(np.std(velocities)) - 0.2) < 0.02

    def test_disabled_returns_input_unchanged(self):
        host = _NoiseHost()
        state = {"j": {"position": 0.1, "velocity": 0.2}}
        assert host._apply_state_noise(state) is state


class TestFrameJitter:
    def test_disabled_returns_input_unchanged(self):
        host = _NoiseHost()
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        assert host._maybe_jitter_frame(frame) is frame

    def test_jitter_shifts_pixels_without_changing_shape(self):
        host = _NoiseHost()
        host.set_obs_noise(camera_jitter_px=3, seed=2)
        frame = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        out = host._maybe_jitter_frame(frame)
        assert out.shape == frame.shape
        assert not np.array_equal(out, frame)
        # A roll is a permutation: the pixel multiset is preserved.
        assert sorted(out.flatten().tolist()) == sorted(frame.flatten().tolist())


class TestRandomizeGuards:
    def test_no_world_is_error(self):
        host = _NoiseHost()  # _world is None
        assert host.randomize(randomize_physics=True)["status"] == "error"

    def test_randomize_positions_rejected(self):
        host = _NoiseHost()
        host._world = object()  # non-None so the world guard passes
        result = host.randomize(randomize_positions=True)
        assert result["status"] == "error"
        assert "randomize_positions is not supported" in result["content"][0]["text"]

    def test_invalid_range_rejected(self):
        host = _NoiseHost()
        host._world = object()
        assert host.randomize(mass_range=(2.0, 0.5))["status"] == "error"


# --------------------------------------------------------------------------- #
# Integration tests (real Newton engine)
# --------------------------------------------------------------------------- #

newton_only = pytest.mark.skipif(not _HAS_NEWTON, reason="newton/warp not installed")


@pytest.fixture
def engine():
    from strands_robots.simulation.newton.simulation import NewtonSimEngine

    sim = NewtonSimEngine(solver="featherstone")
    sim.create_world()
    sim.add_robot("so100")
    yield sim
    sim.destroy()


def _model_mass(sim) -> np.ndarray:
    return np.array(sim._model.body_mass.numpy())


def _model_friction(sim) -> np.ndarray:
    return np.array(sim._model.shape_material_mu.numpy())


@newton_only
class TestPhysicsRandomization:
    def test_physics_scales_mass_and_friction(self, engine):
        base_mass = _model_mass(engine)
        base_friction = _model_friction(engine)
        result = engine.randomize(randomize_physics=True, mass_range=(0.8, 1.2), friction_range=(0.5, 1.5), seed=42)
        assert result["status"] == "success"
        assert not np.allclose(base_mass, _model_mass(engine))
        assert not np.allclose(base_friction, _model_friction(engine))

    def test_mass_scales_stay_within_range(self, engine):
        result = engine.randomize(randomize_physics=True, mass_range=(0.8, 1.2), seed=11)
        scales = result["content"][1]["json"]["mass_scales"]
        assert scales
        assert all(0.8 <= s <= 1.2 for s in scales)

    def test_colors_only_leaves_physics_untouched(self, engine):
        base_mass = _model_mass(engine)
        engine.randomize(randomize_colors=True, randomize_physics=False, seed=1)
        assert np.allclose(base_mass, _model_mass(engine))


@newton_only
class TestReproducibility:
    def test_same_seed_identical_scales(self, engine):
        a = engine.randomize(randomize_physics=True, seed=42)["content"][1]["json"]
        b = engine.randomize(randomize_physics=True, seed=42)["content"][1]["json"]
        assert a["mass_scales"] == b["mass_scales"]
        assert a["friction_scales"] == b["friction_scales"]
        assert a["light_direction"] == b["light_direction"]

    def test_different_seed_differs(self, engine):
        a = engine.randomize(randomize_physics=True, seed=1)["content"][1]["json"]
        b = engine.randomize(randomize_physics=True, seed=2)["content"][1]["json"]
        assert a["mass_scales"] != b["mass_scales"]

    def test_ten_episode_sequence_is_reproducible(self, engine):
        def run() -> list[list[float]]:
            seq = []
            for episode in range(10):
                engine.reset()
                out = engine.randomize(randomize_physics=True, mass_range=(0.8, 1.2), seed=episode)
                seq.append(out["content"][1]["json"]["mass_scales"])
            return seq

        first = run()
        second = run()
        assert first == second
        # Acceptance: across episodes the physics actually varies.
        assert first[0] != first[1]


@newton_only
class TestRenderingAxes:
    def test_lighting_changes_rendered_frame(self, engine):
        engine.randomize(randomize_lighting=True, randomize_colors=False, seed=1)
        img_a = engine.render(width=120, height=90)
        mean_a = img_a["content"][2]["json"]["pixel_mean"]
        engine.randomize(randomize_lighting=True, randomize_colors=False, seed=99)
        img_b = engine.render(width=120, height=90)
        mean_b = img_b["content"][2]["json"]["pixel_mean"]
        assert img_a["status"] == "success" and img_b["status"] == "success"
        assert mean_a != mean_b

    def test_colors_render_succeeds(self, engine):
        engine.randomize(randomize_colors=True, randomize_lighting=False, seed=5)
        assert engine.render(width=120, height=90)["status"] == "success"


@newton_only
class TestSensorNoiseOnEngine:
    def test_observation_position_noise_has_requested_std(self, engine):
        engine.send_action({"Rotation": 0.3}, robot_name="so100", n_substeps=20)
        engine.set_obs_noise(joint_pos_std=0.03, seed=0)
        samples = [engine.get_observation("so100")["Rotation"] for _ in range(3000)]
        assert abs(float(np.std(samples)) - 0.03) < 0.004

    def test_robot_state_velocity_noise(self, engine):
        engine.set_obs_noise(joint_vel_std=0.05, seed=0)
        vels = [
            engine.get_robot_state("so100")["content"][1]["json"]["state"]["Rotation"]["velocity"] for _ in range(3000)
        ]
        assert float(np.std(vels)) > 0.0

    def test_camera_jitter_changes_frame(self, engine):
        baseline = engine.render(width=120, height=90)["content"][1]["image"]["source"]["bytes"]
        engine.set_obs_noise(camera_jitter_px=6, seed=3)
        jittered = engine.render(width=120, height=90)["content"][1]["image"]["source"]["bytes"]
        assert baseline != jittered
