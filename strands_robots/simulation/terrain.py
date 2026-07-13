"""Procedural terrain generation for rough-ground simulation worlds.

A flat ground plane is enough to smoke-test a manipulator, but a locomotion
policy is only interesting on ground it can trip on. The velocity-tracking
locomotion benchmarks (``go2_walk_forward`` / ``g1_walk_forward`` /
``t1_walk_forward`` and the omnidirectional Go2 tasks) all spawn their robot on
a flat plane, so they measure command tracking but never *robustness to
terrain* -- the whole reason legged locomotion is hard. This module generates a
deterministic rough-terrain heightfield that
:meth:`~strands_robots.simulation.base.SimEngine.create_world` can lay down
instead of the flat plane (``create_world(terrain="rough")``), so a floating-base
robot settles onto and walks over bumps. It is the ground-generation primitive a
terrain *curriculum* (progressive difficulty across resets) is built on.

The generator is intentionally backend- and MuJoCo-independent (pure stdlib, no
numpy / mujoco import) so the height data is trivially unit-testable and
deterministic given ``(kind, resolution, seed)`` -- a benchmark that evaluates a
policy on ``terrain="rough"`` regenerates the identical field on every reset.
"""

from __future__ import annotations

import random

# Supported terrain kinds. ``"rough"`` is smoothed value-noise bumps; the tuple
# is the single source of truth both backends validate against and is easy to
# extend (e.g. ``"stairs"``) without touching the create_world signature.
SUPPORTED_TERRAINS: tuple[str, ...] = ("rough",)

# Heightfield geometry (metres). The field spans +/-``TERRAIN_RADIUS`` in x and y
# (matching the flat ground plane's 5 m half-size so the reachable workspace is
# unchanged), rises up to ``TERRAIN_ELEVATION`` at its highest bump, and rests on
# a ``TERRAIN_BASE``-thick solid slab so there is never a hole under the robot.
# The surface height therefore ranges over ``[0, TERRAIN_ELEVATION]`` -- flush
# with z=0 at its lowest point, so a robot never falls below the nominal floor.
TERRAIN_RADIUS = 5.0
TERRAIN_ELEVATION = 0.08
TERRAIN_BASE = 0.1
TERRAIN_RESOLUTION = 40  # nrow == ncol grid cells (25 cm cells over the 10 m field)
TERRAIN_SEED = 0


def validate_terrain(kind: str | None) -> None:
    """Raise ``ValueError`` for an unsupported terrain kind (``None`` is a flat ground)."""
    if kind is None or kind in SUPPORTED_TERRAINS:
        return
    raise ValueError(
        f"Unknown terrain {kind!r}. Supported: {sorted(SUPPORTED_TERRAINS)} (or None / omit for a flat ground plane)."
    )


def _box_blur(grid: list[list[float]], n: int) -> list[list[float]]:
    """One 3x3 box-blur pass (edge-clamped) -- turns spiky noise into walkable bumps."""
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            acc = 0.0
            cnt = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < n and 0 <= jj < n:
                        acc += grid[ii][jj]
                        cnt += 1
            out[i][j] = acc / cnt
    return out


def _rough(n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    grid = [[rng.random() for _ in range(n)] for _ in range(n)]
    # Two blur passes -> smooth, walkable bumps rather than single-cell spikes
    # that would flip a robot on contact and render as noise.
    for _ in range(2):
        grid = _box_blur(grid, n)
    flat = [v for row in grid for v in row]
    lo, hi = min(flat), max(flat)
    span = hi - lo
    if span <= 0.0:  # degenerate (all-equal); flat field
        return [0.0] * (n * n)
    # Normalize to [0, 1]; MuJoCo scales it by the hfield's elevation size.
    return [(v - lo) / span for v in flat]


def generate_heightfield(
    kind: str,
    resolution: int = TERRAIN_RESOLUTION,
    seed: int = TERRAIN_SEED,
) -> list[float]:
    """Return a normalized ``[0, 1]`` heightfield as ``resolution * resolution`` floats.

    Row-major (``userdata`` order for a MuJoCo ``<hfield>``). Deterministic given
    ``(kind, resolution, seed)``. Raises ``ValueError`` for an unknown/None kind
    or a resolution below 2.
    """
    validate_terrain(kind)
    if kind is None:
        raise ValueError("generate_heightfield requires a terrain kind, got None.")
    n = int(resolution)
    if n < 2:
        raise ValueError(f"terrain resolution must be >= 2, got {resolution}.")
    if kind == "rough":
        return _rough(n, seed)
    # validate_terrain accepts only SUPPORTED_TERRAINS; a kind reaching here
    # means the tuple grew without a generator branch.
    raise ValueError(f"terrain kind {kind!r} has no generator implementation.")  # pragma: no cover


__all__ = [
    "SUPPORTED_TERRAINS",
    "TERRAIN_RADIUS",
    "TERRAIN_ELEVATION",
    "TERRAIN_BASE",
    "TERRAIN_RESOLUTION",
    "TERRAIN_SEED",
    "validate_terrain",
    "generate_heightfield",
]
