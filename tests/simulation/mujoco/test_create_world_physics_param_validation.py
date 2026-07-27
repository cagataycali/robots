"""``create_world`` must reject a timestep / gravity the engine cannot honor.

``set_timestep`` and ``set_gravity`` have always validated their input (finite
positive dt, 3-element finite vector or real scalar) and returned a structured
error. ``create_world`` -- which sets those same two values, before any setter
can be called -- validated neither, so a world could be created on terms the
setters would refuse:

* ``create_world(timestep=-1)`` reported ``status="success"`` and compiled a
  negative ``dt`` into ``model.opt``; stepping then ran the integrator
  backwards (``t=-1.0000s``) and produced non-finite accelerations while every
  call still reported success.
* ``create_world(timestep=0)`` was coalesced to the engine default by a
  ``timestep or self.default_timestep`` fallback, so the caller's value was
  discarded silently and the result advertised a dt it had never been asked for.
* ``create_world(timestep=float("nan"))`` produced a ``nan`` world advertised as
  ``"nans (nanHz physics)"``.
* ``create_world(gravity=[0, -9.81])`` raised a bare binding-level
  ``TypeError`` naming MuJoCo's internals out of a method that contracts a
  status dict.
* ``create_world(gravity=["0", "0", "-9.81"])`` echoed the caller's raw input as
  the applied gravity instead of the coerced floats the model received.

These pin the rejection, the parity with the setters, and that a rejected call
leaves no world behind.
"""

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# ``None`` is absent on purpose: it is the "use the engine default" sentinel,
# covered by test_unusable_engine_default_rejected.
_UNUSABLE_TIMESTEPS = [-1, 0, 0.0, float("nan"), float("inf"), "fast", True]


@pytest.fixture
def sim():
    """A simulation with no world yet (create_world is under test)."""
    engine = Simulation(tool_name="test_create_world_physics_params", mesh=False)
    yield engine
    engine.destroy()


def _text(result):
    return result["content"][0]["text"]


class TestTimestepValidation:
    @pytest.mark.parametrize("timestep", _UNUSABLE_TIMESTEPS)
    def test_unusable_timestep_rejected(self, sim, timestep):
        """Every dt the integrator cannot advance by is a structured error."""
        result = sim.create_world(timestep=timestep)
        assert result["status"] == "error"
        assert "timestep" in _text(result)
        # A rejected create_world must not leave a half-built world behind.
        assert sim.step(1)["status"] == "error"

    def test_zero_timestep_is_not_coalesced_to_the_default(self, sim):
        """``0`` must be reported, not silently replaced by the default dt."""
        result = sim.create_world(timestep=0)
        assert result["status"] == "error"
        assert "0.002" not in _text(result)
        assert "timestep must be a finite positive number" in _text(result)

    def test_unusable_engine_default_rejected(self):
        """A bad engine default cannot slip in through ``timestep=None``."""
        engine = Simulation(
            tool_name="test_create_world_bad_default",
            mesh=False,
            default_timestep=-0.002,
        )
        try:
            result = engine.create_world()
            assert result["status"] == "error"
            assert "default_timestep" in _text(result)
            assert engine.step(1)["status"] == "error"
        finally:
            engine.destroy()

    def test_valid_timestep_is_applied(self, sim):
        """A usable dt still reaches the model and is reported accurately."""
        result = sim.create_world(timestep=0.004)
        assert result["status"] == "success"
        assert "0.004s (250Hz physics)" in _text(result)
        assert sim.physics_timestep() == pytest.approx(0.004)

    @pytest.mark.parametrize("timestep", [-1, 0, float("nan"), "fast"])
    def test_create_world_matches_set_timestep(self, sim, timestep):
        """A dt rejected by ``set_timestep`` is rejected by ``create_world``.

        The two entry points write the same ``model.opt.timestep``, so their
        accepted domains must not diverge.
        """
        assert sim.create_world(timestep=timestep)["status"] == "error"
        assert sim.create_world()["status"] == "success"
        assert sim.set_timestep(timestep)["status"] == "error"
        # The world kept the default dt: the refused value was never applied.
        assert sim.physics_timestep() == pytest.approx(0.002)


class TestGravityValidation:
    @pytest.mark.parametrize(
        "gravity",
        [[0.0, -9.81], [0.0, 0.0, -9.81, 0.0], "0,0,-9.81", [0.0, 0.0, float("nan")], ["x", 0.0, 0.0]],
    )
    def test_unusable_gravity_rejected(self, sim, gravity):
        """A mis-shaped gravity is a status dict, never a raised TypeError."""
        result = sim.create_world(gravity=gravity)
        assert result["status"] == "error"
        assert "gravity" in _text(result) or "components" in _text(result)
        assert sim.step(1)["status"] == "error"

    def test_numeric_string_vector_reports_the_applied_floats(self, sim):
        """The result must echo what the model received, not the raw input."""
        result = sim.create_world(gravity=["0", "0", "-3.0"])
        assert result["status"] == "success"
        assert "Gravity: [0.0, 0.0, -3.0]" in _text(result)

    def test_scalar_gravity_is_the_z_component(self, sim):
        """Scalar form keeps parity with ``set_gravity``."""
        result = sim.create_world(gravity=-3.0)
        assert result["status"] == "success"
        assert "Gravity: [0.0, 0.0, -3.0]" in _text(result)

    def test_zero_gravity_is_honored_not_defaulted(self, sim):
        """An all-zero vector is a legitimate request, not a falsy no-value."""
        result = sim.create_world(gravity=[0.0, 0.0, 0.0])
        assert result["status"] == "success"
        assert "Gravity: [0.0, 0.0, 0.0]" in _text(result)


class TestAgentToolDispatch:
    """The agent-tool router validates vector params by name but no numeric
    ranges, so the timestep hole was reachable from the tool as well."""

    def test_dispatch_rejects_negative_timestep(self, sim):
        result = sim._dispatch_action("create_world", {"timestep": -1})
        assert result["status"] == "error"
        assert "timestep" in _text(result)

    def test_dispatch_accepts_valid_world(self, sim):
        result = sim._dispatch_action("create_world", {"timestep": 0.002, "gravity": [0.0, 0.0, -9.81]})
        assert result["status"] == "success"
