"""Type-coercion guards for MuJoCo sim config mutators.

These pin the contract that ``set_gravity``, ``set_timestep`` and
``add_camera`` reject malformed *types* (a non-numeric entry inside an
otherwise correctly-shaped vector, a non-numeric scalar, or a non-sized
argument) with a structured ``{"status": "error"}`` dict rather than
propagating a ``TypeError`` / ``ValueError`` out of the call.

Existing suites already cover numeric-but-invalid input (wrong length,
NaN, Inf, non-positive). The branches exercised here are the ``float(...)``
and ``len(...)`` coercion-failure paths, which fire only for genuinely
non-numeric / non-sized arguments an agent can still supply.
"""

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim_with_world():
    """A minimal simulation with an empty compiled world."""
    sim = Simulation()
    sim.create_world()
    yield sim
    sim.destroy()


class TestSetterInputTypeValidation:
    def test_set_gravity_non_numeric_entry_errors(self, sim_with_world):
        """A correctly-shaped vector with a non-numeric entry is rejected via
        the float() coercion path, not by raising ValueError to the caller."""
        res = sim_with_world.set_gravity(["x", 0.0, 0.0])
        assert res["status"] == "error"
        assert "numbers" in res["content"][0]["text"]

    def test_set_timestep_non_numeric_string_errors(self, sim_with_world):
        """A non-numeric string timestep is rejected via float() coercion."""
        res = sim_with_world.set_timestep("fast")
        assert res["status"] == "error"
        assert "positive number" in res["content"][0]["text"]

    def test_set_timestep_none_errors(self, sim_with_world):
        """``None`` (TypeError under float()) is rejected, not raised."""
        res = sim_with_world.set_timestep(None)
        assert res["status"] == "error"
        assert "positive number" in res["content"][0]["text"]

    def test_add_camera_non_sized_position_errors(self, sim_with_world):
        """A non-sized position (no ``len()``) is rejected via the TypeError
        branch of the shape check rather than raising to the caller."""
        res = sim_with_world.add_camera(name="cam", position=5)
        assert res["status"] == "error"
        assert "list of 3 numbers" in res["content"][0]["text"]


class TestSetGravityNumpyScalar:
    """A scalar z-only gravity may arrive as a NumPy real scalar.

    ``set_gravity`` accepts a bare number as z-only gravity. A value produced by
    NumPy math (``np.float32``, ``np.int64``, ``np.degrees(...)`` etc.) is a real
    scalar but is not an instance of Python ``int`` / ``float`` (only
    ``np.float64`` subclasses ``float``), so the old ``(int, float)`` guard
    skipped the scalar branch and the value fell through to ``len(gravity)`` --
    raising ``TypeError`` internally and surfacing a misleading
    "must be a 3-element list of numbers (... has no len())" error. These pin
    that any ``numbers.Real`` scalar is treated like a plain float.
    """

    def test_numpy_float32_scalar_accepted(self, sim_with_world):
        import numpy as np

        res = sim_with_world.set_gravity(np.float32(-9.81))
        assert res["status"] == "success"
        gravity = sim_with_world._world._model.opt.gravity
        assert list(gravity[:2]) == [0.0, 0.0]
        assert gravity[2] == pytest.approx(-9.81, abs=1e-4)

    def test_numpy_int64_scalar_accepted(self, sim_with_world):
        import numpy as np

        res = sim_with_world.set_gravity(np.int64(-3))
        assert res["status"] == "success"
        assert list(sim_with_world._world._model.opt.gravity) == [0.0, 0.0, -3.0]

    def test_numpy_array_still_takes_vector_path(self, sim_with_world):
        """A 3-element NumPy array is not a scalar and must set x/y/z directly."""
        import numpy as np

        res = sim_with_world.set_gravity(np.array([0.0, 0.0, -3.7]))
        assert res["status"] == "success"
        assert list(sim_with_world._world._model.opt.gravity) == [0.0, 0.0, -3.7]

    def test_numpy_bool_scalar_still_rejected(self, sim_with_world):
        """``np.bool_`` is not ``numbers.Real`` and has no ``len()`` -- it stays
        refused with a structured error rather than becoming a 1.0 z-gravity."""
        import numpy as np

        res = sim_with_world.set_gravity(np.bool_(True))
        assert res["status"] == "error"
        assert "numbers" in res["content"][0]["text"]
