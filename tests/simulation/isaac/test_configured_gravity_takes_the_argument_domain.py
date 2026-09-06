"""A configured gravity takes the same domain as the ``create_world`` argument.

``create_world`` has two owners for one value: its own ``gravity=`` argument and
:attr:`IsaacConfig.gravity`, the field it falls back to. The argument was
normalized through the shared gravity domain and then checked against this
backend's own Z-alignment constraint; the field was read straight into
``PhysicsContext.set_gravity``. Every verdict therefore depended on whether the
caller happened to spell the value at the call site rather than in the config:

* ``(0.0, -9.81, 0.0)`` - the exact vector the argument path documents refusing,
  because ``set_gravity`` takes a signed scalar and cannot aim off-axis - reached
  ``set_gravity(0.0)``, so the world ran in **zero gravity** while the result
  echoed the full vector as if applied.
* ``(0.0, 0.0, nan)`` and ``(0.0, 0.0, inf)`` reached ``set_gravity`` unexamined.
* ``("a", "b", "c")`` reached it as the string ``"c"``.
* ``(0.0, 0.0, 0.0, -9.81)`` - four components - was read at index 2, so a
  mis-shaped vector was also silently zero gravity.
* ``(0.0, 0.0)`` raised ``IndexError`` out of a method whose contract is a
  ``{"status": "error"}`` dict.

The fix resolves config-or-argument into one effective value before validating,
the way ``effective_timestep`` immediately above it already does, and passes the
source name to the domain so the message names the owner to fix. These tests
pin the parity, the refusals, and the values that must keep working.

Nothing here needs Isaac Sim: the refusals happen before any Kit import, and the
applied cases run against a fake ``isaacsim`` tree that records what
``set_gravity`` received.
"""

from __future__ import annotations

import sys
import types

import pytest

from strands_robots.simulation.isaac import simulation as isaac_simulation
from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import IsaacSimulation

# ---------------------------------------------------------------------------
# Fake Isaac tree: enough of it to reach set_gravity and record the argument.
# ---------------------------------------------------------------------------


class _RecordingPhysicsContext:
    """Records the single value ``set_gravity`` was called with."""

    def __init__(self) -> None:
        self.gravity_calls: list[object] = []

    def set_gravity(self, magnitude: object) -> None:
        self.gravity_calls.append(magnitude)


class _FakeScene:
    def add_default_ground_plane(self) -> None:
        return None


class _FakeWorld:
    """Stands in for ``isaacsim.core.api.World``, exposing the physics context."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.physics_context = _RecordingPhysicsContext()
        self.scene = _FakeScene()

    def get_physics_context(self) -> _RecordingPhysicsContext:
        return self.physics_context

    def reset(self) -> None:
        return None


@pytest.fixture()
def fake_isaacsim(monkeypatch):
    """Fake ``isaacsim`` tree covering every import ``create_world`` performs."""
    monkeypatch.setattr(isaac_simulation, "_SIMULATION_APP", None)
    monkeypatch.setattr(isaac_simulation, "_SIMULATION_APP_LAUNCH", None)
    mods = {}
    for name in ("isaacsim", "isaacsim.core", "isaacsim.core.api"):
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        mods[name] = module
    mods["isaacsim"].SimulationApp = lambda launch=None: types.SimpleNamespace(launch=launch)
    mods["isaacsim"].core = mods["isaacsim.core"]
    mods["isaacsim.core"].api = mods["isaacsim.core.api"]
    mods["isaacsim.core.api"].World = _FakeWorld
    return mods


def _world_from_config(gravity: object) -> tuple[dict, list[object]]:
    """Create a world with ``gravity`` set on the config, not passed as an argument.

    Returns:
        The ``create_world`` result and the list of values ``set_gravity``
        received (empty when the value was refused before the world was built).
    """
    sim = IsaacSimulation(config=IsaacConfig(gravity=gravity))  # type: ignore[arg-type]
    result = sim.create_world()
    world = sim._world
    return result, list(world.get_physics_context().gravity_calls) if world is not None else []


def _world_from_argument(gravity: object) -> tuple[dict, list[object]]:
    """Create a world with ``gravity`` passed as the ``create_world`` argument."""
    sim = IsaacSimulation()
    result = sim.create_world(gravity=gravity)  # type: ignore[arg-type]
    world = sim._world
    return result, list(world.get_physics_context().gravity_calls) if world is not None else []


# Values whose verdict must not depend on which owner carried them. Each is a
# spelling a caller can reach either way; the applied ones are the values the
# backend can honour, the refused ones the values ``set_gravity`` cannot.
SHARED_DOMAIN_VALUES = (
    (0.0, 0.0, -9.81),
    [0.0, 0.0, -9.81],
    (0.0, 0.0, 0.0),
    -9.81,
    (0.0, -9.81, 0.0),
    (-9.81, 0.0, 0.0),
    (0.0, 0.0, float("nan")),
    (0.0, 0.0, float("inf")),
    True,
    (0.0, 0.0),
    (0.0, 0.0, 0.0, -9.81),
    ("a", "b", "c"),
    (0.0, True, -9.81),
    {"z": -9.81},
)

# The one value whose verdict legitimately differs between the two owners, with
# the reason. ``gravity=None`` as an argument is the spelling of "not stated" -
# it is the parameter's default and means "use the field" - while ``None`` set
# on the field is a stated value that is not a gravity vector.
OWNER_SPECIFIC_VALUES: dict[object, str] = {
    None: (
        "as an argument None is the default and means 'unstated, read the field'; "
        "as a field value it is a stated non-vector and is refused"
    ),
}


class TestOneDomainForOneValue:
    """Both owners of the gravity value take the same domain."""

    @pytest.mark.parametrize("value", SHARED_DOMAIN_VALUES, ids=repr)
    def test_the_two_owners_reach_the_same_verdict(self, value, fake_isaacsim):
        """The config field and the argument agree on accept-or-refuse.

        Pre-fix seven of these values were refused as an argument and applied
        from the field, so a caller could get a world the backend cannot honour
        purely by moving the value from the call site into the config.
        """
        from_config, _ = _world_from_config(value)
        from_argument, _ = _world_from_argument(value)
        assert from_config["status"] == from_argument["status"], (
            f"gravity={value!r} is {from_argument['status']} as an argument but "
            f"{from_config['status']} from IsaacConfig.gravity: "
            f"config said {from_config['content'][0]['text']!r}"
        )

    @pytest.mark.parametrize("value", SHARED_DOMAIN_VALUES, ids=repr)
    def test_an_applied_value_reaches_the_physics_context_identically(self, value, fake_isaacsim):
        """When accepted, both owners hand ``set_gravity`` the same scalar.

        Agreeing on the verdict is not enough: the value the physics context
        receives - and the vector the result reports - must also be the same,
        or the two owners honour the same request differently.
        """
        from_config, config_calls = _world_from_config(value)
        from_argument, argument_calls = _world_from_argument(value)
        if from_argument["status"] != "success":
            pytest.skip(f"{value!r} is refused by the shared domain")
        assert config_calls == argument_calls
        assert from_config["content"][0]["json"]["gravity"] == from_argument["content"][0]["json"]["gravity"]

    def test_every_owner_specific_value_states_why(self):
        """The asymmetry roster is deliberate, not a leftover.

        A value may only be exempt from the parity above with a written reason,
        so a future exemption is a decision rather than a quietly widened list.
        """
        assert OWNER_SPECIFIC_VALUES
        for value, reason in OWNER_SPECIFIC_VALUES.items():
            assert reason.strip(), f"{value!r} is exempt from gravity parity with no reason"

    def test_an_unstated_argument_reads_the_field(self, fake_isaacsim):
        """``None`` as the argument is 'unstated' and takes the configured value."""
        sim = IsaacSimulation(config=IsaacConfig(gravity=(0.0, 0.0, -1.62)))
        result = sim.create_world(gravity=None)
        assert result["status"] == "success", result
        assert sim._world.get_physics_context().gravity_calls == [-1.62]

    def test_a_stated_argument_outranks_the_field(self, fake_isaacsim):
        """An explicit argument still overrides the configured gravity."""
        sim = IsaacSimulation(config=IsaacConfig(gravity=(0.0, 0.0, -9.81)))
        result = sim.create_world(gravity=[0.0, 0.0, -1.62])
        assert result["status"] == "success", result
        assert sim._world.get_physics_context().gravity_calls == [-1.62]


class TestAConfiguredGravityTheBackendCannotHonour:
    """What the field-sourced refusals say, and that nothing was applied."""

    def test_an_off_axis_field_is_refused_not_reduced_to_zero(self, fake_isaacsim):
        """The documented off-axis defect, reached through the config field.

        ``PhysicsContext.set_gravity`` takes a signed scalar, so the y-component
        cannot be applied. Pre-fix the z-component was read instead, making this
        a request for lateral gravity that ran as zero gravity while the result
        reported ``[0.0, -9.81, 0.0]`` as if applied.
        """
        result, calls = _world_from_config((0.0, -9.81, 0.0))
        assert result["status"] == "error"
        text = result["content"][0]["text"].lower()
        assert "z-aligned" in text
        assert calls == [], f"set_gravity was called with {calls!r} for a gravity that was refused"

    @pytest.mark.parametrize("value", [(0.0, 0.0, float("nan")), (0.0, 0.0, float("inf"))], ids=["nan", "inf"])
    def test_a_non_finite_field_is_refused(self, value, fake_isaacsim):
        """A non-finite z reached the physics context unexamined pre-fix."""
        result, calls = _world_from_config(value)
        assert result["status"] == "error"
        assert "finite" in result["content"][0]["text"]
        assert calls == []

    def test_a_non_numeric_component_is_refused_before_the_physics_context(self, fake_isaacsim):
        """``("a", "b", "c")`` reached ``set_gravity("c")`` pre-fix."""
        result, calls = _world_from_config(("a", "b", "c"))
        assert result["status"] == "error"
        assert calls == []

    def test_a_mis_shaped_field_is_reported_not_raised(self, fake_isaacsim):
        """A 2-component field raised ``IndexError`` past the error contract.

        ``create_world`` returns a status dict for every failure it knows about;
        reading index 2 of a 2-vector escaped that contract as an exception the
        surrounding ``except`` clauses deliberately do not catch.
        """
        result, calls = _world_from_config((0.0, 0.0))
        assert result["status"] == "error"
        assert "3-element" in result["content"][0]["text"]
        assert calls == []

    def test_a_four_component_field_is_refused_not_read_at_index_two(self, fake_isaacsim):
        """Four components silently became zero gravity pre-fix, not an error."""
        result, calls = _world_from_config((0.0, 0.0, 0.0, -9.81))
        assert result["status"] == "error"
        assert "3-element" in result["content"][0]["text"]
        assert calls == []

    def test_the_message_names_the_owner_that_carried_the_value(self, fake_isaacsim):
        """A refusal has to say which of the two spellings to go and fix.

        The domain takes the parameter name to quote, so the field-sourced
        message names ``IsaacConfig.gravity`` and the argument-sourced one names
        ``gravity`` - otherwise both read as a complaint about the call site.
        """
        from_config, _ = _world_from_config((0.0, 0.0))
        from_argument, _ = _world_from_argument((0.0, 0.0))
        assert "IsaacConfig.gravity" in from_config["content"][0]["text"]
        assert "IsaacConfig.gravity" not in from_argument["content"][0]["text"]
        assert "'gravity'" in from_argument["content"][0]["text"]


class TestTheValuesThatMustKeepWorking:
    """Controls: the fix refuses what the backend cannot honour, and no more."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ((0.0, 0.0, -9.81), -9.81),
            ([0.0, 0.0, -1.62], -1.62),
            ((0.0, 0.0, 0.0), 0.0),
            ((0.0, 0.0, 5.0), 5.0),
            (-9.81, -9.81),
        ],
        ids=["default", "moon_list", "zero_g", "inverted", "scalar"],
    )
    def test_a_z_aligned_field_still_reaches_the_physics_context(self, value, expected, fake_isaacsim):
        """Z-aligned, finite gravity is applied and reported as three components.

        A real scalar is included on purpose: the shared domain reads it as the
        z-component, which the ``gravity=`` argument has always accepted, so the
        field now accepts the same spelling rather than raising an unrelated
        ``TypeError`` about a float not being iterable.
        """
        result, calls = _world_from_config(value)
        assert result["status"] == "success", result
        assert calls == [expected]
        assert result["content"][0]["json"]["gravity"] == [0.0, 0.0, expected]

    def test_the_default_configuration_is_earth_gravity(self, fake_isaacsim):
        """The out-of-the-box world still falls at 9.81 m/s^2 along -Z."""
        sim = IsaacSimulation()
        result = sim.create_world()
        assert result["status"] == "success", result
        assert sim._world.get_physics_context().gravity_calls == [-9.81]
        assert result["content"][0]["json"]["gravity"] == [0.0, 0.0, -9.81]
