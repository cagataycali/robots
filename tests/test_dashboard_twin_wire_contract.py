"""The pose frame the twin reads has one owner for its width and one for its order.

The browser twin owns no physics: it draws the compiled model once from
``/api/sim/{id}/scene`` and then moves it with one binary frame per tick, so the
frame's layout is a contract between three files -
:func:`strands_robots.dashboard.scene.describe` publishes ``pose_row_floats``,
:func:`strands_robots.dashboard.sim_session._pack_poses` emits the rows, and
``static/twin.js`` strides through them - and reads that width off the scene
rather than restating it, so a row that grew a velocity moves the browser too.
A restated width is silent when it drifts: the frame still parses, every geom
lands on a neighbour's matrix element, and nothing raises. Row order is the
other half no size check can see.

These cells read the width from ``describe``, grade the packer against it, and
require the shipped asset to carry no copy of it.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
from collections.abc import Iterator

import numpy as np
import pytest

from strands_robots.dashboard import scene, sim_session

TWIN_JS = pathlib.Path(sim_session.__file__).parent / "static" / "twin.js"

_HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None


@pytest.fixture(scope="module")
def so101() -> Iterator[tuple[object, object, dict]]:
    """A compiled so101 and its forward-stepped data, with the scene it publishes."""
    if not _HAS_MUJOCO:
        pytest.skip("mujoco not installed")
    session = sim_session.SimSession("so101")
    try:
        assert session.wait_ready(60), "engine did not start"
        if session.snapshot.state == "error":
            pytest.skip(f"no renderer here: {session.snapshot.error}")
        engine = session._engine
        assert engine is not None, "a ready session has an engine"
        yield session.model, engine.mj_data, scene.describe(session.model)
    finally:
        session.stop()


class TestTheTwinsPoseFrame:
    def test_the_declared_row_width_is_the_width_the_packer_emits(self, so101) -> None:
        """``pose_row_floats`` is what a row costs, not a number kept beside one."""
        model, data, described = so101
        assert described["pose_row_floats"] == scene.POSE_ROW_FLOATS
        assert len(sim_session._pack_poses(data)) == int(model.ngeom) * scene.POSE_ROW_FLOATS * 4

    def test_each_row_is_that_geoms_position_then_its_rotation(self, so101) -> None:
        """Row *i* is ``geom_xpos[i]`` then row-major ``geom_xmat[i]``, in float32.

        Order is the half no size check can see: swapping the two halves keeps
        every byte count right and puts each geom at a matrix element.
        """
        model, data, described = so101
        rows = np.frombuffer(sim_session._pack_poses(data), dtype="<f4").reshape(-1, described["pose_row_floats"])
        assert rows.shape == (int(model.ngeom), scene.POSE_ROW_FLOATS)
        # float32 is the only difference from the engine's own arrays.
        assert np.array_equal(rows[:, :3], np.asarray(data.geom_xpos, dtype="<f4").reshape(-1, 3))
        assert np.array_equal(rows[:, 3:], np.asarray(data.geom_xmat, dtype="<f4").reshape(-1, 9))

    def test_the_shipped_twin_takes_the_width_from_the_scene_it_fetched(self) -> None:
        """``twin.js`` strides by the published ``pose_row_floats``, holding no copy of it.

        The asset ships in the wheel and is the only consumer of the frame. A
        width restated here would agree with the server today and drift in
        silence: the frame still parses and every geom lands on a neighbour's
        matrix element, so this grades that no copy exists.
        """
        source = TWIN_JS.read_text(encoding="utf-8")
        # Naming the field in a comment is not reading it, so grade the assignment.
        held = re.findall(r"(this\.\w+)\s*=\s*\w+\.pose_row_floats\b", source)
        assert held, f"{TWIN_JS.name} never takes the width {scene.describe.__name__} publishes as pose_row_floats"
        # Every stride the frame reader applies: `f.length / X` and `i * X`.
        strides = set(re.findall(r"(?:\.length\s*/|\bi\s*\*)\s*([\w.$]+)", source))
        assert strides, f"no row stride found in {TWIN_JS.name}; the scan grades nothing"
        assert strides == {held[0]}, (
            f"{TWIN_JS.name} strides by {sorted(strides)}; every stride must be {held[0]}, "
            "the width the scene published"
        )
