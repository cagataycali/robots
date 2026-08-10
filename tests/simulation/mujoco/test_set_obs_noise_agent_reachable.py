"""``set_obs_noise`` must be reachable from the schema an agent reads.

The MuJoCo dispatch router resolves an action with a bare ``getattr``, so it
accepts any public method by name whether or not ``tool_spec.json`` advertises
it. ``set_obs_noise`` was on the wrong side of that gap: implemented,
documented, validated and dispatchable, but absent from the ``action`` enum,
with none of its three noise magnitudes declared as a schema property. A model
driving the tool reads the schema and nothing else, so observation noise could
not be configured by an agent at all - the capability was reachable only from
Python.

What is pinned here is the *parameter set*, derived from the method signature
rather than written out beside it, so a magnitude added to ``set_obs_noise``
later cannot become undiscoverable by being omitted from the schema. The
narrow scope is deliberate: a general "every documented parameter is a schema
property" rule over-reaches, because several enum actions legitimately take
callables (``on_frame``, ``policy_object``) that a JSON schema cannot express.
See #2093.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

# Skip the whole module if mujoco isn't available (dev env without [sim-mujoco]).
pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# The magnitudes the docstring documents. Named literally so a signature read
# that silently returned nothing cannot make the derived assertions vacuous.
OBS_NOISE_MAGNITUDES = ("joint_pos_std", "joint_vel_std", "camera_jitter_px")


def _shipped_schema() -> dict[str, Any]:
    """The tool_spec.json shipped inside the package, not a rebuilt copy."""
    import strands_robots.simulation.mujoco as mjpkg

    spec_path = Path(mjpkg.__file__).parent / "tool_spec.json"
    with open(spec_path, encoding="utf-8") as f:
        schema: dict[str, Any] = json.load(f)
    return schema


def _declared_parameters() -> list[str]:
    """``set_obs_noise``'s parameters, excluding ``self`` and ``**kwargs``.

    ``**kwargs`` is declared only to match the ``SimEngine.set_obs_noise``
    signature and forwards nothing, so it is not a knob and has no schema
    property to own.
    """
    sig = inspect.signature(Simulation.set_obs_noise)
    return [
        name
        for name, param in sig.parameters.items()
        if name != "self" and param.kind is not inspect.Parameter.VAR_KEYWORD
    ]


@pytest.fixture
def sim() -> Generator[Simulation, None, None]:
    s = Simulation(tool_name="obs_noise_reachability", mesh=False)
    yield s
    s.cleanup()


class TestSetObsNoiseIsAdvertised:
    """The action and its knobs appear in the schema an agent is handed."""

    def test_set_obs_noise_is_in_the_action_enum(self) -> None:
        enum = _shipped_schema()["properties"]["action"]["enum"]
        assert "set_obs_noise" in enum, (
            "set_obs_noise dispatches but is absent from the action enum, so an "
            "agent reading the schema cannot discover it"
        )

    def test_every_declared_parameter_is_a_schema_property(self) -> None:
        properties = _shipped_schema()["properties"]
        declared = _declared_parameters()

        # Non-vacuity: a signature read that returned nothing would otherwise
        # satisfy the loop below trivially.
        assert set(OBS_NOISE_MAGNITUDES).issubset(declared), (
            f"expected {OBS_NOISE_MAGNITUDES} among set_obs_noise's parameters, got {declared}"
        )

        missing = [name for name in declared if name not in properties]
        assert not missing, (
            f"set_obs_noise parameters absent from tool_spec.json properties: {sorted(missing)}; "
            "a parameter no schema property names cannot be supplied by an agent"
        )

    def test_each_magnitude_is_typed_as_a_number(self) -> None:
        """A std is continuous; declaring it ``integer`` would bar 0.01."""
        properties = _shipped_schema()["properties"]
        for name in OBS_NOISE_MAGNITUDES:
            assert properties[name]["type"] == "number", (
                f"{name} is a float std but the schema declares "
                f"{properties[name]['type']!r}, which refuses fractional values"
            )


class TestTheAdvertisedKnobsActuallyConfigureNoise:
    """The advertised names are the ones the dispatch path accepts."""

    def test_dispatch_accepts_every_magnitude_by_its_schema_name(self, sim: Simulation) -> None:
        properties = _shipped_schema()["properties"]
        kwargs = {name: 0.01 for name in OBS_NOISE_MAGNITUDES if name in properties}
        assert len(kwargs) == len(OBS_NOISE_MAGNITUDES)

        result = sim(action="set_obs_noise", **kwargs)

        assert result["status"] == "success", result
        text = result["content"][0]["text"]
        for name in OBS_NOISE_MAGNITUDES:
            assert f"{name}=0.01" in text, f"{name} did not reach the configuration: {text}"

    def test_the_validation_behind_the_advertised_action_is_reachable(self, sim: Simulation) -> None:
        """Advertising the action exposes its refusals too, not just its successes."""
        result = sim(action="set_obs_noise", joint_pos_std=-1.0)

        assert result["status"] == "error", result
        assert "joint_pos_std" in result["content"][0]["text"]
