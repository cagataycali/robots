"""``mj_model``/``mj_data`` hand back the CURRENT objects, so a cached one is dead.

Every op that recompiles the MJCF goes through ``spec.recompile``, which
allocates a new model and data and installs them (see
``scene_ops._recompile_preserving_state``, whose own docstring says it replaces
``world._model`` and ``_data``). An external consumer that took the handles
before such an op therefore holds a pair that is not stale but DETACHED: it
keeps its own sizes, state and clock, accepts writes and ``mj_step`` without
error, and neither its reads nor its writes ever reach the engine again.

That failure is silent, and it is silent in the shape that produces plausible
numbers: an analysis script that poses the arm through the held ``qpos`` and
measures against the held model gets a self-consistent answer about a world
nobody is simulating, while the engine's own readers (``get_observation``,
``get_contacts``, ``render``) keep answering about the untouched scene. The
properties previously documented only the OTHER staleness mechanism - racing a
``PolicyRunner`` worker's ``mj_step`` - and offered "read between steps", which
cannot help here because the detached pair never agrees again.

Two things are pinned:

* **Behaviour** - the handles are replaced by ``add_object`` and by
  ``add_camera``, the replacement changes the model's sizes, and a write through
  the pre-mutation data does not reach the live engine.
* **The documentation** - both property docstrings name a triggering op, the
  recompile that causes it, and the permanence, so a consumer reading the API is
  told the mechanism rather than discovering it from a wrong measurement.
"""

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_mj_handle_sim", mesh=False)
    s.create_world(gravity=[0, 0, -9.81])
    s.add_robot("so101")
    yield s
    s.cleanup()


def _add_cube(s):
    return s.add_object(name="cube", shape="box", size=[0.028, 0.028, 0.028], position=[0.1, -0.2, 0.08])


class TestASceneRebuildReplacesTheHandles:
    def test_add_object_replaces_both_handles(self, sim):
        model, data = sim.mj_model, sim.mj_data
        _add_cube(sim)

        assert sim.mj_model is not model
        assert sim.mj_data is not data
        # The replacement is not cosmetic: the held model describes a smaller world.
        assert model.nq < sim.mj_model.nq

    def test_add_camera_replaces_both_handles(self, sim):
        model, data = sim.mj_model, sim.mj_data
        sim.add_camera(name="probe", position=[0.4, -0.4, 0.2], target=[0.1, -0.2, 0.07])

        assert sim.mj_model is not model
        assert sim.mj_data is not data
        assert model.ncam < sim.mj_model.ncam

    def test_a_write_through_the_pre_rebuild_data_never_reaches_the_engine(self, sim):
        model, data = sim.mj_model, sim.mj_data
        _add_cube(sim)

        data.qpos[1] = 0.7  # accepted, no error, on the detached pair
        mj.mj_forward(model, data)

        assert data.qpos[1] == pytest.approx(0.7)  # the write landed somewhere
        assert sim.mj_data.qpos[1] == pytest.approx(0.0)  # but not in the live scene
        live = sim.get_observation(skip_images=True)
        moved = [k for k, v in live.items() if isinstance(v, float) and abs(v - 0.7) < 1e-6]
        assert moved == []

    def test_the_detached_pair_keeps_its_own_clock(self, sim):
        model, data = sim.mj_model, sim.mj_data
        _add_cube(sim)

        for _ in range(10):
            mj.mj_step(model, data)  # steps without error, forever

        assert data.time > 0.0
        assert sim.mj_data.time == pytest.approx(0.0)


class TestThePropertiesDocumentTheRebuild:
    """The fact above is only discoverable from the docstring, so grade it."""

    @pytest.mark.parametrize("name", ["mj_model", "mj_data"])
    def test_the_docstring_names_trigger_mechanism_and_permanence(self, name):
        doc = getattr(Simulation, name).__doc__
        assert doc, f"{name} has no docstring to grade"

        assert "add_object" in doc, f"{name} does not name an op that triggers the rebuild"
        assert "recompile" in doc, f"{name} does not name the recompile that causes it"
        # Wording-tolerant on the permanence: any of these says "not momentary".
        assert any(w in doc for w in ("detached", "permanent", "for good")), (
            f"{name} does not say the replacement is permanent, so a reader is left "
            "with the momentary mj_step race as the only staleness mechanism"
        )
