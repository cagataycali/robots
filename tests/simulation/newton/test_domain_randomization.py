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

    def test_non_numeric_is_error(self):
        host = _NoiseHost()
        result = host.set_obs_noise(joint_pos_std="fast")
        assert result["status"] == "error"
        assert host._obs_noise is None

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

    def test_subpixel_jitter_is_noop(self):
        host = _NoiseHost()
        host.set_obs_noise(camera_jitter_px=0.5, seed=0)  # int(0.5) == 0 -> no shift
        frame = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
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


class _FakeWarp:
    """Minimal Warp stand-in: ``mat33f`` passes the inertia matrix through.

    The real Newton backend wraps the scaled inertia in a ``warp.mat33f`` so the
    finalized model can ingest it. The randomization logic only needs the call
    to succeed and round-trip the value, so identity passthrough is faithful.
    """

    @staticmethod
    def mat33f(value):
        return value


class _FakeBuilder:
    """Mimics the slice of a Newton ``ModelBuilder`` that randomization mutates.

    Body 0 is fixed (mass 0) so tests can assert it is skipped by mass scaling,
    matching the real engine where the world/base body has zero mass.
    """

    def __init__(self) -> None:
        self.shape_color = [(0.5, 0.5, 0.5) for _ in range(4)]
        self.body_mass = [0.0, 1.0, 2.0, 3.0]
        self.body_inertia = [np.eye(3, dtype=np.float32) for _ in range(4)]
        self.shape_material_mu = [0.8, 0.8, 0.8, 0.8]


class _RebuildHost(DomainRandomizationMixin):
    """Host that drives the full ``randomize`` -> ``_rebuild`` path off-GPU.

    Mirrors how ``NewtonSimEngine._rebuild`` invokes
    ``_apply_domain_randomization`` on a freshly-assembled builder under the
    instance lock, so the deterministic sampling and array mutation run without
    Newton or Warp installed.
    """

    def __init__(self, builder: _FakeBuilder | None = None) -> None:
        self._lock = threading.RLock()
        self._world = None
        self._dr = None
        self._dr_applied = None
        self._dr_light_dir = None
        self._obs_noise = None
        self._obs_noise_rng = None
        self._wp = _FakeWarp()
        self.builder = builder or _FakeBuilder()

    def _rebuild(self) -> None:
        self._apply_domain_randomization(self.builder)


def _active_host(builder: _FakeBuilder | None = None) -> _RebuildHost:
    """A _RebuildHost with a non-None world so randomize()'s guard passes."""
    host = _RebuildHost(builder)
    # A non-None stand-in is enough; randomize() only checks `_world is None`,
    # so the concrete type is irrelevant to the code under test.
    host._world = object()  # type: ignore[assignment]
    return host


class TestApplyDomainRandomizationNoSpec:
    def test_no_active_spec_leaves_builder_untouched(self):
        host = _RebuildHost()  # _dr is None until randomize() runs
        before = list(host.builder.body_mass)
        host._apply_domain_randomization(host.builder)
        assert host.builder.body_mass == before
        assert host._dr_applied is None


class TestRandomizePhysicsApplied:
    def test_mass_scales_within_range_and_skip_zero_mass_body(self):
        host = _active_host()
        base_mass = list(host.builder.body_mass)
        result = host.randomize(
            randomize_physics=True,
            randomize_colors=False,
            randomize_lighting=False,
            mass_range=(0.8, 1.2),
            friction_range=(0.5, 1.5),
            seed=42,
        )
        assert result["status"] == "success"
        applied = result["content"][1]["json"]
        # The fixed (mass-0) body is skipped; the three positive bodies scale.
        assert len(applied["mass_scales"]) == 3
        assert all(0.8 <= s <= 1.2 for s in applied["mass_scales"])
        assert host.builder.body_mass[0] == 0.0
        for i in range(1, 4):
            assert host.builder.body_mass[i] != base_mass[i]
        # Every shape's friction is scaled within range.
        assert len(applied["friction_scales"]) == 4
        assert all(0.5 <= f <= 1.5 for f in applied["friction_scales"])
        assert "bodies mass-scaled" in result["content"][0]["text"]

    def test_inertia_tracks_mass_scale(self):
        host = _active_host()
        host.randomize(
            randomize_physics=True,
            randomize_colors=False,
            randomize_lighting=False,
            mass_range=(0.8, 1.2),
            seed=7,
        )
        scales = host._dr_applied["mass_scales"]
        # body_inertia[1] started as identity; it must now equal identity * scale.
        np.testing.assert_allclose(np.asarray(host.builder.body_inertia[1]), np.eye(3) * scales[0], rtol=1e-5)


class TestRandomizeColorsApplied:
    def test_every_shape_color_resampled_within_range(self):
        host = _active_host()
        result = host.randomize(
            randomize_colors=True,
            randomize_lighting=False,
            randomize_physics=False,
            color_range=(0.1, 0.9),
            seed=3,
        )
        assert result["status"] == "success"
        assert result["content"][1]["json"]["n_colors"] == 4
        for color in host.builder.shape_color:
            assert len(color) == 3
            assert all(0.1 <= c <= 0.9 for c in color)
        assert "shapes randomized" in result["content"][0]["text"]


class TestRandomizeLightingApplied:
    def test_light_direction_is_unit_vector_and_recorded(self):
        host = _active_host()
        result = host.randomize(randomize_lighting=True, randomize_colors=False, randomize_physics=False, seed=2)
        light = result["content"][1]["json"]["light_direction"]
        assert len(light) == 3
        assert abs(float(np.linalg.norm(light)) - 1.0) < 1e-6
        assert host._dr_light_dir == light
        assert "light direction" in result["content"][0]["text"]

    def test_lighting_disabled_clears_light_direction(self):
        host = _active_host()
        result = host.randomize(randomize_lighting=False, randomize_colors=False, randomize_physics=False)
        assert host._dr_light_dir is None
        assert "No axes enabled" in result["content"][0]["text"]


class TestRandomizeReproducibility:
    def test_same_seed_identical_multipliers(self):
        a = _active_host().randomize(randomize_physics=True, mass_range=(0.8, 1.2), seed=99)["content"][1]["json"]
        b = _active_host().randomize(randomize_physics=True, mass_range=(0.8, 1.2), seed=99)["content"][1]["json"]
        assert a["mass_scales"] == b["mass_scales"]
        assert a["friction_scales"] == b["friction_scales"]
        assert a["light_direction"] == b["light_direction"]

    def test_different_seed_differs(self):
        a = _active_host().randomize(randomize_physics=True, seed=1)["content"][1]["json"]
        b = _active_host().randomize(randomize_physics=True, seed=2)["content"][1]["json"]
        assert a["mass_scales"] != b["mass_scales"]


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
