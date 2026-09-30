"""Isaac cameras show the state the simulation is in, and reading them advances nothing.

Two faults with one cause. Isaac's RTX camera products - every camera but the
first - refresh only on a Kit app update that advances the timeline; a
render-only ``World.render`` never lights them.

* The multi-camera observation therefore ran an extra ``SimulationApp.update()``,
  which integrated a whole ``rendering_dt`` (four physics steps) that ``sim_time``
  never counted: 150 recorded steps at 15 Hz integrated 15.03 s of physics where
  the same rollout unrecorded ran 10.0 s, ending on another pose.
* After ``reset()`` the products held no frame of the reset scene for ~6 updates,
  so the first observation of every episode showed the policy a black wrist view.

The World is now built at ``rendering_dt == physics_dt``, so a rendering step is
ONE physics step that also refreshes every camera, and ``reset()`` renders until
every camera is lit, then zeroes the velocities those ticks left.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import numpy as np

from strands_robots.simulation.isaac.simulation import IsaacSimulation, _CameraState


class _Camera:
    """An RTX product that is black for its first ``dark`` rendered updates."""

    def __init__(self, world: _World, dark: int) -> None:
        self.world, self.dark = world, dark

    def get_rgba(self) -> np.ndarray:
        lit = self.world.rendered_updates > self.dark
        return np.full((4, 4, 4), 200 if lit else 0, dtype=np.uint8)


class _World:
    def __init__(self) -> None:
        self.current_time = 0.0
        self.rendered_updates = 0
        self.render_args: list[bool] = []

    def step(self, render: bool = False) -> None:
        self.current_time += 1 / 120
        self.render_args.append(render)
        if render:
            self.rendered_updates += 1

    def reset(self) -> None:
        self.current_time = 0.0


class _Articulation:
    def __init__(self) -> None:
        self.velocities: list[np.ndarray] = []

    def get_joint_positions(self) -> np.ndarray:
        return np.zeros(6)

    def set_joint_velocities(self, v: np.ndarray) -> None:
        self.velocities.append(np.asarray(v))


def _sim(dark: tuple[int, ...]) -> tuple[IsaacSimulation, _World, _Articulation]:
    world = _World()
    sim = IsaacSimulation.__new__(IsaacSimulation)
    sim._world, sim._world_created = world, True
    sim._cameras = {}
    for i, d in enumerate(dark):
        cam = _CameraState(name=f"c{i}", prim_path=f"/c{i}", width=4, height=4)
        cam.handle = _Camera(world, d)
        sim._cameras[cam.name] = cam
    art = _Articulation()
    sim._robots = {"arm": SimpleNamespace(articulation=art)}  # type: ignore[dict-item]
    sim._objects = {}
    sim._lock = threading.RLock()
    return sim, world, art


def test_reset_renders_until_the_slowest_camera_is_lit_then_stops() -> None:
    sim, world, art = _sim(dark=(0, 5))
    sim._light_cameras_after_reset()
    assert world.render_args == [True] * 6
    assert all(IsaacSimulation._camera_frame_is_lit(cam) for cam in sim._cameras.values())
    assert art.velocities and not art.velocities[-1].any()


def test_a_camera_that_never_lights_costs_a_bounded_number_of_ticks() -> None:
    sim, world, _ = _sim(dark=(10_000,))
    sim._light_cameras_after_reset()
    assert len(world.render_args) == IsaacSimulation._RESET_LIGHT_TICKS_MAX


def test_a_frame_of_zeros_is_not_lit_and_a_missing_handle_does_not_block() -> None:
    assert not IsaacSimulation._camera_frame_is_lit(
        SimpleNamespace(handle=SimpleNamespace(get_rgba=lambda: np.zeros((2, 2, 4)))),  # type: ignore[arg-type]
    )
    assert IsaacSimulation._camera_frame_is_lit(SimpleNamespace(handle=None))  # type: ignore[arg-type]


def test_every_dynamic_object_is_left_at_rest() -> None:
    sim, _, _ = _sim(dark=(0,))
    calls: dict[str, Any] = {}
    handle = SimpleNamespace(
        set_linear_velocity=lambda v: calls.__setitem__("lin", np.asarray(v)),
        set_angular_velocity=lambda v: calls.__setitem__("ang", np.asarray(v)),
    )
    objects: dict[str, Any] = {
        "cube": SimpleNamespace(is_static=False, handle=handle),
        "table": SimpleNamespace(is_static=True, handle=None),
    }
    sim._objects = objects
    sim._settle_after_lighting()
    assert not calls["lin"].any() and not calls["ang"].any()
