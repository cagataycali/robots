"""Domain randomization refuses ranges, noise amplitudes and seeds it cannot apply.

``randomize`` writes its numeric arguments straight into the live ``mjModel``
(``body_mass``, ``body_inertia``, ``geom_friction``, ``geom_rgba``) and into
``data.qpos``, and ``set_obs_noise`` stores a seed that is only drawn from later.
Neither validated any of those values, so a value with no valid sampling
interval had two ways to go wrong:

* it raised deep inside the mutation loop - ``TypeError``/``IndexError``/
  ``OverflowError`` past the tool envelope these methods are dispatched behind
  (a scalar or 3-element ``mass_range``, a 1-element ``color_range``, a NaN
  ``position_noise``, a float ``seed``); or
* it succeeded and left a world that models nothing: ``mass_range=(-1, -1)``
  installed a NEGATIVE body mass, so the object fell *upward* (0.30 m -> 2.07 m
  in 300 steps) while the call reported "Physics: N bodies mass-scaled";
  ``mass_range=(0, 0)`` left a massless body frozen in mid-air; and
  ``friction_range=(-2, -2)`` installed a negative Coulomb coefficient.

The Newton backend already refused the three shared ranges through its own
private copy of the rule. These tests pin the promoted, shared contract
(:func:`~strands_robots.simulation.base.randomization_range_error`,
:func:`~strands_robots.simulation.base.finite_non_negative_error`,
:func:`~strands_robots.simulation.base.randomization_seed_error`) on the MuJoCo
backend, and pin that both backends call it so the accepted domains cannot
diverge again.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

import strands_robots.simulation as simulation_pkg  # noqa: E402
from strands_robots.simulation.base import (  # noqa: E402
    finite_non_negative_error,
    randomization_range_error,
    randomization_seed_error,
)
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

CUBE_Z = 0.30


@pytest.fixture
def sim():
    """A world holding one dynamic 1 kg cube hovering at 0.30 m."""
    s = Simulation(tool_name="test_randomization_guards", mesh=False)
    assert s.create_world(gravity=[0, 0, -9.81])["status"] == "success"
    assert (
        s.add_object(name="cube", shape="box", size=[0.06, 0.06, 0.06], position=[0.0, 0.0, CUBE_Z], mass=1.0)["status"]
        == "success"
    )
    yield s
    s.cleanup()


def _err_text(result: dict) -> str:
    return " ".join(block["text"] for block in result["content"] if "text" in block)


def _cube_z(sim: Simulation) -> float:
    body = sim.get_body_state("cube")
    payload = next(block["json"] for block in body["content"] if "json" in block)
    return float(payload["position"][2])


def _cube_mass(sim: Simulation) -> float:
    assert sim._world is not None and sim._world._model is not None
    model = sim._world._model
    return float(model.body_mass[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "cube")])


class TestRangeGuardHelper:
    """The promoted range rule: a finite, ordered, physically usable interval."""

    @pytest.mark.parametrize(
        "value",
        [0.5, (0.5,), (0.5, 2.0, 3.0), "ab", None, {"lo": 0.5}],
        ids=["scalar", "one-element", "three-element", "two-char-string", "none", "mapping"],
    )
    def test_rejects_anything_that_is_not_a_numeric_pair(self, value):
        msg = randomization_range_error(value, "mass_range")
        assert msg is not None and "must be a (lo, hi) pair of numbers" in msg

    @pytest.mark.parametrize("value", [(float("nan"), 1.0), (0.5, float("inf")), (-float("inf"), 1.0)])
    def test_rejects_non_finite_bounds(self, value):
        assert "bounds must be finite" in (randomization_range_error(value, "mass_range") or "")

    def test_rejects_inverted_bounds(self):
        assert "exceeds upper bound" in (randomization_range_error((2.0, 0.5), "friction_range") or "")

    def test_zero_is_a_real_setting_for_friction_and_colour(self):
        assert randomization_range_error((0.0, 1.5), "friction_range") is None
        assert randomization_range_error((0.0, 1.0), "color_range") is None

    def test_zero_and_negative_mass_scales_are_both_refused_with_distinct_reasons(self):
        zero = randomization_range_error((0.0, 0.0), "mass_range", allow_zero=False)
        negative = randomization_range_error((-1.0, -0.5), "mass_range", allow_zero=False)
        assert zero is not None and "must be positive" in zero and "erases" in zero
        assert negative is not None and "must be positive" in negative and "flips the sign" in negative

    def test_accepts_an_ordered_positive_pair(self):
        assert randomization_range_error((0.5, 2.0), "mass_range", allow_zero=False) is None


class TestMagnitudeAndSeedGuardHelpers:
    """Noise amplitudes describe a distribution; seeds must reach ``default_rng``."""

    @pytest.mark.parametrize("value", ["x", None, [0.1]])
    def test_non_numeric_amplitude_is_rejected(self, value):
        msg = finite_non_negative_error(value, "position_noise", "randomize")
        assert msg is not None and msg.startswith("randomize: position_noise must be a number")

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.5])
    def test_non_finite_or_negative_amplitude_is_rejected(self, value):
        msg = finite_non_negative_error(value, "position_noise", "randomize")
        assert msg is not None and "finite non-negative number" in msg

    def test_zero_amplitude_is_a_valid_no_op(self):
        assert finite_non_negative_error(0.0, "position_noise", "randomize") is None

    @pytest.mark.parametrize("value", [2.5, "abc", True, -1, [1, 2]], ids=["float", "str", "bool", "negative", "list"])
    def test_seed_outside_the_annotated_domain_is_rejected(self, value):
        msg = randomization_seed_error(value, "randomize")
        assert msg is not None and msg.startswith("randomize: seed must be a non-negative integer or None")

    @pytest.mark.parametrize("value", [None, 0, 7, np.int64(3)])
    def test_seed_inside_the_annotated_domain_is_accepted(self, value):
        assert randomization_seed_error(value, "set_obs_noise") is None
        assert np.random.default_rng(value) is not None


class TestRandomizeRefusesUnusablePhysics:
    """A scale a body cannot have is refused, not installed."""

    def test_negative_mass_scale_is_refused_and_the_cube_still_falls(self, sim):
        """Pre-fix: status=success, mass -1 kg, cube RISES 0.30 -> 2.07 m."""
        result = sim.randomize(
            randomize_colors=False, randomize_lighting=False, randomize_physics=True, mass_range=(-1.0, -1.0)
        )
        assert result["status"] == "error"
        assert "mass_range" in _err_text(result)
        assert _cube_mass(sim) == pytest.approx(1.0)
        sim.step(300)
        assert _cube_z(sim) < 0.1, "a refused randomization must leave gravity working"

    def test_zero_mass_scale_is_refused_and_the_cube_is_not_frozen(self, sim):
        """Pre-fix: status=success, mass 0, cube hovered at 0.30 m forever."""
        result = sim.randomize(
            randomize_colors=False, randomize_lighting=False, randomize_physics=True, mass_range=(0.0, 0.0)
        )
        assert result["status"] == "error"
        assert _cube_mass(sim) == pytest.approx(1.0)
        sim.step(300)
        assert _cube_z(sim) < 0.1

    def test_negative_friction_scale_is_refused_and_friction_stays_coulomb(self, sim):
        """Pre-fix: status=success with geom_friction[:, 0] == -1.54."""
        assert sim._world is not None and sim._world._model is not None
        before = np.array(sim._world._model.geom_friction[:, 0], copy=True)
        result = sim.randomize(
            randomize_colors=False, randomize_lighting=False, randomize_physics=True, friction_range=(-2.0, -1.0)
        )
        assert result["status"] == "error"
        assert "friction_range" in _err_text(result)
        assert np.array_equal(sim._world._model.geom_friction[:, 0], before)
        assert (sim._world._model.geom_friction[:, 0] >= 0).all()

    @pytest.mark.parametrize(
        ("param", "value"),
        [
            ("mass_range", 0.5),
            ("mass_range", (0.5, 2.0, 3.0)),
            ("mass_range", (float("nan"), float("nan"))),
            ("friction_range", (0.5,)),
            ("color_range", (0.1,)),
            ("color_range", "ab"),
        ],
        ids=["mass-scalar", "mass-triple", "mass-nan", "friction-single", "color-single", "color-string"],
    )
    def test_malformed_range_returns_the_tool_envelope_instead_of_raising(self, sim, param, value):
        """Pre-fix these escaped as TypeError / IndexError / OverflowError."""
        result = sim.randomize(
            randomize_colors=True, randomize_lighting=False, randomize_physics=True, **{param: value}
        )
        assert result["status"] == "error"
        assert param in _err_text(result)

    @pytest.mark.parametrize("value", [float("nan"), -0.5, "x"], ids=["nan", "negative", "string"])
    def test_unusable_position_noise_is_refused_and_qpos_stays_finite(self, sim, value):
        """Pre-fix a NaN half-width raised, and would have written NaN into qpos."""
        result = sim.randomize(
            randomize_colors=False, randomize_lighting=False, randomize_positions=True, position_noise=value
        )
        assert result["status"] == "error"
        assert "position_noise" in _err_text(result)
        assert sim._world is not None and sim._world._data is not None
        assert np.isfinite(sim._world._data.qpos).all()

    @pytest.mark.parametrize("value", [2.5, "abc", -1], ids=["float", "string", "negative"])
    def test_unusable_seed_is_refused_by_both_randomization_entry_points(self, sim, value):
        """Pre-fix ``default_rng`` raised TypeError/ValueError past the envelope."""
        randomize_result = sim.randomize(seed=value)
        noise_result = sim.set_obs_noise(joint_pos_std=0.01, seed=value)
        assert randomize_result["status"] == "error"
        assert noise_result["status"] == "error"
        assert _err_text(randomize_result).startswith("randomize: seed")
        assert _err_text(noise_result).startswith("set_obs_noise: seed")

    def test_usable_values_are_still_applied(self, sim):
        """The happy path: every axis on, zero friction/colour lower bounds, seeded."""
        result = sim.randomize(
            randomize_colors=True,
            randomize_lighting=True,
            randomize_physics=True,
            randomize_positions=True,
            position_noise=0.01,
            color_range=(0.0, 1.0),
            friction_range=(0.0, 1.5),
            mass_range=(0.5, 2.0),
            seed=7,
        )
        assert result["status"] == "success"
        assert 0.5 <= _cube_mass(sim) <= 2.0
        assert sim.set_obs_noise(joint_pos_std=0.01, camera_jitter_px=1, seed=0)["status"] == "success"


class TestBackendGuardParity:
    """Every backend with a randomization mixin routes through the shared guards.

    AST-based so it runs with Newton/Isaac uninstalled: a backend that stops
    calling a guard (or a new backend that never starts) fails here rather than
    silently re-acquiring its own accepted domain.
    """

    _GUARD_NAMES = {"randomization_range_error", "finite_non_negative_error", "randomization_seed_error"}

    @staticmethod
    def _called_guards(module_path: Path, method: str) -> set[str]:
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == method:
                return {
                    call.func.id
                    for call in ast.walk(node)
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                }
        raise AssertionError(f"{module_path.name} defines no {method}()")

    @staticmethod
    def _modules() -> list[Path]:
        package_dir = Path(simulation_pkg.__file__).parent
        found = sorted(package_dir.glob("*/randomization.py"))
        assert {p.parent.name for p in found} >= {"mujoco", "newton"}, found
        return found

    def test_every_backend_randomize_validates_its_ranges_and_seed(self):
        for module_path in self._modules():
            called = self._called_guards(module_path, "randomize")
            assert "randomization_range_error" in called, module_path
            assert "randomization_seed_error" in called, module_path

    def test_every_backend_set_obs_noise_validates_amplitudes_and_seed(self):
        for module_path in self._modules():
            called = self._called_guards(module_path, "set_obs_noise")
            assert "finite_non_negative_error" in called, module_path
            assert "randomization_seed_error" in called, module_path

    def test_no_backend_keeps_a_private_copy_of_the_range_rule(self):
        for module_path in self._modules():
            source = module_path.read_text(encoding="utf-8")
            assert "def _validate_range(" not in source, module_path
