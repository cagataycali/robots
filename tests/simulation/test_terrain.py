"""Unit tests for the procedural terrain heightfield generator.

``strands_robots.simulation.terrain.generate_heightfield`` is the backend- and
MuJoCo-independent ground-generation primitive behind
``create_world(terrain="rough")``. It must be deterministic given
``(kind, resolution, seed)`` (so a benchmark regenerates the identical field on
every reset), produce a genuinely non-flat normalized ``[0, 1]`` field for a
rough kind, and reject an unknown kind with an actionable error. These tests
are pure stdlib (no mujoco / numpy) and exercise the module in isolation.
"""

from __future__ import annotations

import pytest

from strands_robots.simulation import terrain


def test_rough_field_has_correct_length_and_range() -> None:
    n = 16
    h = terrain.generate_heightfield("rough", resolution=n, seed=terrain.TERRAIN_SEED)
    assert len(h) == n * n
    assert all(0.0 <= v <= 1.0 for v in h)


def test_rough_field_is_deterministic_for_same_seed() -> None:
    a = terrain.generate_heightfield("rough", resolution=20, seed=3)
    b = terrain.generate_heightfield("rough", resolution=20, seed=3)
    assert a == b


def test_rough_field_varies_with_seed() -> None:
    a = terrain.generate_heightfield("rough", resolution=20, seed=1)
    b = terrain.generate_heightfield("rough", resolution=20, seed=2)
    assert a != b


def test_rough_field_is_genuinely_non_flat() -> None:
    h = terrain.generate_heightfield("rough", resolution=32, seed=0)
    # A rough field must span most of the [0, 1] range (normalization pins the
    # min to 0 and max to 1), i.e. it is not a near-flat plane.
    assert max(h) - min(h) > 0.5


def test_default_resolution_matches_module_constant() -> None:
    h = terrain.generate_heightfield("rough")
    assert len(h) == terrain.TERRAIN_RESOLUTION * terrain.TERRAIN_RESOLUTION


@pytest.mark.parametrize("bad", ["flat", "ROUGH", "", "spiral", "steps"])
def test_unknown_terrain_kind_is_rejected_actionably(bad: str) -> None:
    with pytest.raises(ValueError) as exc:
        terrain.generate_heightfield(bad)
    msg = str(exc.value)
    assert "Supported" in msg and "rough" in msg  # actionable: lists what IS valid


def test_none_kind_is_rejected_by_generator() -> None:
    with pytest.raises(ValueError):
        terrain.generate_heightfield(None)  # type: ignore[arg-type]


def test_resolution_below_two_is_rejected() -> None:
    with pytest.raises(ValueError):
        terrain.generate_heightfield("rough", resolution=1)


def test_validate_terrain_accepts_none_and_supported_rejects_unknown() -> None:
    terrain.validate_terrain(None)  # flat ground; no raise
    for kind in terrain.SUPPORTED_TERRAINS:
        terrain.validate_terrain(kind)  # no raise
    with pytest.raises(ValueError):
        terrain.validate_terrain("bogus")


def test_stairs_field_has_correct_length_and_discrete_levels() -> None:
    n = 40
    h = terrain.generate_heightfield("stairs", resolution=n)
    assert len(h) == n * n
    assert all(0.0 <= v <= 1.0 for v in h)
    assert min(h) == 0.0 and max(h) == 1.0
    # A staircase is DISCRETE: exactly TERRAIN_STAIR_STEPS distinct plateau
    # levels (this is what distinguishes it from the continuous "rough" field).
    assert len(set(h)) == terrain.TERRAIN_STAIR_STEPS


def test_stairs_field_is_deterministic_and_seed_independent() -> None:
    # Stairs are fully deterministic (no rng), so the seed must not change them.
    a = terrain.generate_heightfield("stairs", resolution=24, seed=0)
    b = terrain.generate_heightfield("stairs", resolution=24, seed=99)
    assert a == b


def test_stairs_climbs_along_x_and_is_constant_across_y() -> None:
    n = 40
    h = terrain.generate_heightfield("stairs", resolution=n)
    rows = [h[i * n : (i + 1) * n] for i in range(n)]
    # MuJoCo hfield userdata is row-major (row 0 -> min y, col 0 -> min x): the
    # staircase rises along +x (columns), so every row is identical...
    assert all(rows[i] == rows[0] for i in range(n))
    # ...and each row is a monotonically non-decreasing step function of x.
    row0 = rows[0]
    assert all(row0[j] <= row0[j + 1] for j in range(n - 1))
    assert row0[0] == 0.0 and row0[-1] == 1.0


def test_stairs_is_genuinely_stepped_not_smooth() -> None:
    # Distinguish stairs (few discrete plateaus) from rough (near-continuous).
    stairs = terrain.generate_heightfield("stairs", resolution=32)
    rough = terrain.generate_heightfield("rough", resolution=32, seed=0)
    assert len(set(stairs)) == terrain.TERRAIN_STAIR_STEPS
    assert len(set(rough)) > terrain.TERRAIN_STAIR_STEPS * 10
