"""The pose frame the twin reads is the shape three places independently assume.

The browser twin owns no physics: it draws the compiled model once from
``/api/sim/{id}/scene`` and then moves it with one binary frame per tick, so the
frame's layout is a contract between three files that each state it separately -
:func:`strands_robots.dashboard.scene.describe` publishes ``pose_row_floats``,
:func:`strands_robots.dashboard.sim_session._pack_poses` emits the rows, and
``static/twin.js`` strides through them. Nothing computes the width from
anything else, so a change in one is silent in the other two: a row that grew a
velocity would be read as a pose, and a row packed rotation-first would place
every geom at a matrix element. Neither crashes, and no other cell notices - the
existing sizes are asserted against the same literal the code uses.

These cells read the width from ``describe`` and grade the other two against it.
"""

from __future__ import annotations

import pathlib
import re
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from strands_robots.dashboard import scene, sim_session

TWIN_JS = pathlib.Path(sim_session.__file__).parent / "static" / "twin.js"

try:  # pragma: no cover - import guard, not behaviour
    import mujoco  # noqa: F401

    _HAS_MUJOCO = True
except Exception:  # pragma: no cover
    _HAS_MUJOCO = False


@pytest.fixture(scope="module")
def so101() -> Iterator[tuple[Any, Any, dict[str, Any]]]:
    """A compiled so101 and its forward-stepped data, with the scene it publishes."""
    if not _HAS_MUJOCO:
        pytest.skip("mujoco not installed")
    session = sim_session.SimSession("so101")
    try:
        assert session.wait_ready(60), "engine did not start"
        if session.snapshot.state == "error":
            pytest.skip(f"no renderer here: {session.snapshot.error}")
        engine = session._engine
        assert engine is not None, "the session reported ready without an engine"
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

    def test_the_shipped_twin_strides_by_the_declared_width(self) -> None:
        """``twin.js`` reads rows of ``pose_row_floats``, whatever that becomes.

        The asset ships in the wheel and is the only consumer of the frame, so
        the width it assumes is graded here rather than in a browser.
        """
        source = TWIN_JS.read_text(encoding="utf-8")
        # Every stride the frame reader applies: `f.length / N` and `i * N`.
        strides = {int(n) for n in re.findall(r"(?:\.length\s*/|\bi\s*\*)\s*(\d+)", source)}
        assert strides, f"no row stride found in {TWIN_JS.name}; the scan grades nothing"
        assert strides == {scene.POSE_ROW_FLOATS}
