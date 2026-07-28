"""A ray the caster cannot cast is refused, never reported as a miss.

``raycast`` and ``multi_raycast`` exist to answer clearance and obstacle
questions, so "no intersection" is a load-bearing answer. Two inputs used to
produce that answer without casting anything:

* ``multi_raycast`` validated each direction INSIDE the cast loop and, for a
  malformed one, appended ``{"distance": None, "geom_id": None, "error": ...}``
  and continued. ``distance: None`` is exactly what a genuine miss reports, the
  overall ``status`` stayed ``"success"``, and the summary text folded the
  rejected ray into the hit denominator (``"1/2 hits"``), so a bearing that was
  never cast read as free space. The batch parameter itself was unguarded too: a
  bare string was iterated one ray per character, a non-sequence raised
  ``TypeError`` past the tool contract, and an empty batch reported ``0/0 hits``.
* ``exclude_body`` reached ``mj_ray`` unchecked on both methods. A float / str /
  ``nan`` raised ``TypeError`` out of the pybind11 signature, and an id outside
  ``[0, model.nbody)`` matched no body - so the geoms the caller asked the ray to
  pass through were included and could be reported as the obstacle.
"""

import numpy as np
import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Directions that name no castable ray. Each must be refused identically by the
# single-ray and the batch entry point - a caller must not get two contracts for
# one malformed vector.
UNCASTABLE_DIRECTIONS = [
    pytest.param([0.0, 1.0], id="two-components"),
    pytest.param([0.0, 0.0, -1.0, 0.0], id="four-components"),
    pytest.param([], id="empty"),
    pytest.param([0.0, 0.0, 0.0], id="zero-length"),
    pytest.param([float("nan"), 0.0, 0.0], id="nan"),
    pytest.param([float("inf"), 0.0, 0.0], id="inf"),
    pytest.param(["x", 0.0, 0.0], id="non-numeric"),
    pytest.param(7, id="not-a-sequence"),
]

# Exclusions mj_ray cannot honor: it takes a C int and skips the geoms whose body
# id equals it, so only -1 or an id the compiled model defines means anything.
UNHONORABLE_EXCLUSIONS = [
    pytest.param(-5, id="negative-but-not-minus-one"),
    pytest.param(999, id="beyond-nbody"),
    pytest.param(2.7, id="fractional"),
    pytest.param("1", id="string"),
    pytest.param(True, id="bool"),
    pytest.param(float("nan"), id="nan"),
]


@pytest.fixture
def sim():
    """A world holding exactly two bodies: ``world`` (id 0) and ``cube`` (id 1).

    The cube sits under the origin so a downward ray hits it, and the ground
    plane sits below the cube so excluding the cube's body still yields a hit -
    which is what makes the exclusion observable rather than just accepted.
    """
    engine = Simulation()
    engine.create_world()
    assert (
        engine.add_object(name="cube", shape="box", size=[0.2, 0.2, 0.2], position=[0, 0, 0.1])["status"] == "success"
    )
    yield engine
    engine.destroy()


def _json(result):
    return next(block["json"] for block in result["content"] if "json" in block)


def _text(result):
    return result["content"][0]["text"]


class TestBatchNamesNoCastableRay:
    """The ``directions`` parameter itself must name at least one castable ray."""

    def test_string_is_not_a_ray_per_character(self, sim):
        result = sim.multi_raycast(origin=[0, 0, 1.0], directions="abc")
        assert result["status"] == "error"
        assert "character" in _text(result)

    def test_non_sequence_errors_instead_of_raising(self, sim):
        result = sim.multi_raycast(origin=[0, 0, 1.0], directions=5)
        assert result["status"] == "error"
        assert "sequence of direction vectors" in _text(result)

    def test_empty_batch_is_refused(self, sim):
        result = sim.multi_raycast(origin=[0, 0, 1.0], directions=[])
        assert result["status"] == "error"
        assert "at least one direction" in _text(result)


class TestUncastableDirectionRefusesTheBatch:
    """A malformed direction must not be reported as a bearing with no hit."""

    @pytest.mark.parametrize("direction", UNCASTABLE_DIRECTIONS)
    def test_single_and_batch_agree_on_an_uncastable_direction(self, sim, direction):
        single = sim.raycast(origin=[0, 0, 1.0], direction=direction)
        batch = sim.multi_raycast(origin=[0, 0, 1.0], directions=[direction])
        assert single["status"] == "error"
        assert batch["status"] == "error"

    def test_one_bad_direction_casts_no_ray_at_all(self, sim):
        """The valid rays are not cast, so no partial sweep can be mistaken for a full one."""
        result = sim.multi_raycast(
            origin=[0, 0, 1.0],
            directions=[[0, 0, -1], [0, 0, 0], [1, 0, 0]],
        )
        assert result["status"] == "error"
        payload = _json(result)
        assert "rays" not in payload
        assert [entry["index"] for entry in payload["invalid_directions"]] == [1]

    def test_every_offending_index_is_named(self, sim):
        """One round-trip is enough to fix a whole malformed fan."""
        result = sim.multi_raycast(
            origin=[0, 0, 1.0],
            directions=[[0, 0, -1], [0, 1], [1, 0, 0], [float("nan"), 0, 0]],
        )
        assert result["status"] == "error"
        assert [entry["index"] for entry in _json(result)["invalid_directions"]] == [1, 3]
        assert "2 of 4" in _text(result)


class TestExcludedBodyDomain:
    """``exclude_body`` must name a body the model defines, or nothing (-1)."""

    @pytest.mark.parametrize("method", ["raycast", "multi_raycast"])
    @pytest.mark.parametrize("exclusion", UNHONORABLE_EXCLUSIONS)
    def test_unhonorable_exclusion_is_refused(self, sim, method, exclusion):
        if method == "raycast":
            result = sim.raycast(origin=[0, 0, 1.0], direction=[0, 0, -1], exclude_body=exclusion)
        else:
            result = sim.multi_raycast(origin=[0, 0, 1.0], directions=[[0, 0, -1]], exclude_body=exclusion)
        assert result["status"] == "error"
        text = _text(result)
        assert "exclude_body" in text
        # The world holds world(0) + cube(1), so the honorable ids are -1, 0, 1.
        assert "[0, 2)" in text

    @pytest.mark.parametrize("exclusion", [-1, 0, 1, np.int64(1), 1.0])
    def test_honorable_exclusion_is_accepted(self, sim, exclusion):
        result = sim.raycast(origin=[0, 0, 1.0], direction=[0, 0, -1], exclude_body=exclusion)
        assert result["status"] == "success"

    def test_exclusion_is_actually_applied(self, sim):
        """Excluding the cube's body lets the ray through to the ground below it."""
        included = _json(sim.raycast(origin=[0, 0, 1.0], direction=[0, 0, -1]))
        excluded = _json(sim.raycast(origin=[0, 0, 1.0], direction=[0, 0, -1], exclude_body=1))
        assert included["geom_name"] == "cube_geom"
        assert excluded["geom_name"] == "ground"
        assert excluded["distance"] > included["distance"]


class TestCastableBatchStillResolves:
    def test_hits_and_misses_are_reported_in_order(self, sim):
        result = sim.multi_raycast(origin=[0, 0, 1.0], directions=[[0, 0, -1], [0, 0, 1]])
        assert result["status"] == "success"
        payload = _json(result)
        assert payload["hits"] == 1
        assert payload["rays"][0]["distance"] == pytest.approx(0.8, abs=1e-6)
        assert payload["rays"][1]["distance"] is None
        assert "1/2 hits" in _text(result)

    def test_numpy_direction_array_is_accepted(self, sim):
        result = sim.multi_raycast(origin=[0, 0, 1.0], directions=np.array([[0.0, 0.0, -1.0]]))
        assert result["status"] == "success"
        assert _json(result)["hits"] == 1
