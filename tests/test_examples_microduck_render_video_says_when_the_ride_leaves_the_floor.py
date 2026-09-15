"""``render_video.py`` says when a ride runs off the rendered floor.

The world strips the floor a Microduck scene ships (``size="0 0 0.05"``, drawn
without limit) and lays its own ``ground`` plane, drawn to 5 m from the origin.
MuJoCo planes collide without limit, so a robot that rolls past that edge keeps
rolling - on a floor the frame no longer shows. The example's roller recipe,
``--vx 0.3 --duration 8``, did exactly that: measured on the L40S the roller
covers about 0.7 m/s at that command, crosses x = 5 m at 7.5 s and finishes at
5.41 m, so the last half-second of the clip shows the duck over the void, and
nothing said so.

The recipe now runs 6 s (finishing near 3.7 m, on the checkerboard), and the
example reports the second a ride left the drawn floor, how far it went, and what
to change. These cells pin the floor-extent read and the crossing detection on a
small model of their own; the ``mujoco`` cells need no GL, assets or ONNX.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

_EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "microduck" / "render_video.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("render_video_floor_under_test", _EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model(mujoco, ground: str):
    return mujoco.MjModel.from_xml_string(f"<mujoco><worldbody>{ground}</worldbody></mujoco>")


class TestTheFloorExtent:
    def test_it_is_the_smaller_drawn_half_extent_of_the_ground_plane(self) -> None:
        mujoco = pytest.importorskip("mujoco")
        module = _load()
        model = _model(mujoco, '<geom name="ground" type="plane" size="5 3 0.01"/>')
        assert module._floor_half_extent(mujoco, model) == 3.0

    def test_a_plane_drawn_without_limit_cannot_be_run_off(self) -> None:
        mujoco = pytest.importorskip("mujoco")
        module = _load()
        model = _model(mujoco, '<geom name="ground" type="plane" size="0 0 0.05"/>')
        assert module._floor_half_extent(mujoco, model) is None

    def test_no_ground_plane_means_no_edge(self) -> None:
        mujoco = pytest.importorskip("mujoco")
        module = _load()
        assert (
            module._floor_half_extent(mujoco, _model(mujoco, '<geom name="floor" type="plane" size="5 5 0.01"/>'))
            is None
        )
        assert (
            module._floor_half_extent(mujoco, _model(mujoco, '<geom name="ground" type="box" size="5 5 0.01"/>'))
            is None
        )


class TestTheCrossing:
    def test_the_first_tick_past_the_edge_in_either_axis_is_named(self) -> None:
        module = _load()
        path = [(0.0, 0.0), (2.0, 0.1), (4.9, 0.2), (5.05, 0.3), (5.4, 0.4)]
        assert module.first_tick_off_the_floor(path, 5.0) == 3
        assert module.first_tick_off_the_floor([(0.0, 0.0), (1.0, -5.2)], 5.0) == 1
        assert module.first_tick_off_the_floor([(0.0, 0.0), (-5.2, 1.0)], 5.0) == 1

    def test_a_ride_that_stays_on_the_floor_is_not_reported(self) -> None:
        module = _load()
        path = np.array([[0.0, 0.0, 0.12], [3.7, 0.5, 0.12]])  # z is carried and ignored
        assert module.first_tick_off_the_floor(path, 5.0) is None
        assert module.first_tick_off_the_floor([], 5.0) is None

    def test_an_unbounded_floor_is_never_left(self) -> None:
        module = _load()
        assert module.first_tick_off_the_floor([(1e6, 1e6)], None) is None


class TestTheRecipe:
    def test_the_roller_recipe_stays_on_the_floor(self) -> None:
        text = _EXAMPLE.read_text(encoding="utf-8")
        recipe = re.search(r"roller\.onnx --scene scene_rollers\.xml \\\n\s+--vx ([\d.]+) --duration ([\d.]+)", text)
        assert recipe is not None, "the roller recipe is gone"
        vx, duration = float(recipe.group(1)), float(recipe.group(2))
        # The roller covers about 0.7 m/s at --vx 0.3 (2.3x the command); the drawn floor ends at 5 m.
        assert vx / 0.3 * 0.7 * duration < 5.0

    def test_the_notice_names_the_second_and_the_remedy(self) -> None:
        text = _EXAMPLE.read_text(encoding="utf-8")
        assert "left the rendered floor" in text
        assert "shorten --duration or lower --vx" in text
