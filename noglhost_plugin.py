"""Emulate a host with no usable offscreen GL context (no EGL/OSMesa).

Forces the MuJoCo backend's cached render probe to False, which is exactly the
state `_can_render()` reaches on a headless host where neither EGL nor OSMesa is
loadable. Also forces the shared test probe negative so `requires_gl` skips.
"""

import os
import pathlib


def pytest_configure(config):
    os.environ["ROBOT_TEST_MUJOCO"] = "0"
    from strands_robots.simulation.mujoco import backend

    print("NOGLHOST tree:", pathlib.Path(backend.__file__).parents[3])
    backend._rendering_available = False
    backend._can_render = lambda: False
