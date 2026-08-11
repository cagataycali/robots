"""``remove_camera`` reports a refused recompile instead of claiming the removal.

Deleting a camera mutates the live ``MjSpec`` *before* the ``spec.recompile``
that validates the result, and a refused recompile leaves ``world._model``
untouched. So the delete has to be rolled back out of the spec and reported,
exactly as its inverse does: ``add_camera`` rolls a refused add back out
(:func:`~strands_robots.simulation.mujoco.scene_ops.inject_camera_into_scene`)
and returns ``{'status': 'error'}``.

Before the fix ``remove_camera`` logged the refusal at warning level, dropped the
registry entry anyway and returned ``{'status': 'success', ... "removed."}``.
That left three observable inconsistencies, and this module pins each one:

* the spec no longer declared the camera while the compiled model still had it,
  so ``list_cameras`` stopped naming a camera ``render`` and
  ``get_camera_params`` went on resolving - two consumers of one scene giving
  opposite answers about whether the camera exists,
* the delete landed *later*, applied by whichever unrelated mutation next
  recompiled successfully, so a camera a rollout or recording was reading from
  disappeared at an ``add_object`` call with nothing to attribute it to,
* the compiler's reason - the signal that the scene had become uncompilable -
  reached only the log, so the caller was told the opposite of what happened.

The refusal is deliberately transient here (the recompile is refused only for the
duration of the patch), which is what lets the last test in each class show the
refusal cost the caller nothing permanent: the identical removal succeeds once
the refusal clears.
"""

from __future__ import annotations

from typing import Any

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco import scene_ops  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402
from tests.simulation.mujoco._gl_probe import requires_gl  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_remove_camera_refusal", mesh=False)
    s.create_world()
    # Two cameras so the surviving one shows the refusal is scoped, and so the
    # registry's ORDER is observable: the camera under test sits in the middle.
    assert s.add_camera(name="watch", position=[0.7, -0.7, 0.5], target=[0, 0, 0.1])["status"] == "success"
    assert s.add_camera(name="keep", position=[0.0, -1.0, 0.6], target=[0, 0, 0.1])["status"] == "success"
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


def _refuse_recompiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the recompile that validates a spec mutation fail.

    Patches the binding rather than the helper so the refusal enters through the
    same call the production path makes. ``monkeypatch.undo()`` models the
    failure clearing.
    """

    def _boom(_self: object, *_a: object, **_k: object) -> object:
        raise ValueError("simulated spec.recompile refusal")

    monkeypatch.setattr(mj.MjSpec, "recompile", _boom)


def _model_cameras(sim: Any) -> list[str]:
    model = sim._world._model
    return [mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]


def _spec_cameras(sim: Any) -> list[str]:
    return [cam.name for cam in sim._world._backend_state["spec"].cameras]


class TestARefusedRecompileIsReportedNotClaimed:
    def test_the_removal_reports_an_error_naming_the_camera(self, sim: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """The caller learns the removal did not happen, and which camera."""
        _refuse_recompiles(monkeypatch)
        refused = sim.remove_camera("watch")
        monkeypatch.undo()

        assert refused["status"] == "error"
        text = refused["content"][0]["text"]
        assert "watch" in text
        # The message has to say the camera is STILL THERE, because the caller's
        # next move depends on it: pre-fix this same call said "removed."
        assert "still registered" in text

    def test_the_camera_keeps_its_registry_entry_and_its_position(
        self, sim: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The registry is restored to exactly what it was, order included.

        The entry is dropped only once the recompile is accepted, so a refusal
        needs no restore and cannot reorder the map - which a pop-then-reinsert
        rollback would, moving the camera to the end of ``list_cameras``.
        """
        before = list(sim._world.cameras)

        _refuse_recompiles(monkeypatch)
        assert sim.remove_camera("watch")["status"] == "error"
        monkeypatch.undo()

        assert list(sim._world.cameras) == before
        assert before.index("watch") < before.index("keep"), "the fixture must not order it last"


class TestTheSceneIsLeftAsItWasFound:
    def test_the_spec_and_the_model_still_agree(self, sim: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pre-fix the spec lost the camera while the live model kept it."""
        spec_before, model_before = _spec_cameras(sim), _model_cameras(sim)
        assert "watch" in spec_before and "watch" in model_before

        _refuse_recompiles(monkeypatch)
        assert sim.remove_camera("watch")["status"] == "error"
        monkeypatch.undo()

        assert _spec_cameras(sim) == spec_before
        assert _model_cameras(sim) == model_before

    def test_every_consumer_still_resolves_the_camera(self, sim: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """The registry and the intrinsics give the same answer.

        Pre-fix ``list_cameras`` stopped naming the camera while
        ``get_camera_params`` - which resolves it from the model - went on
        succeeding. Both consumers read the scene without a GL context, so this
        pin holds on every host; the renderer is the third consumer and is
        pinned separately because it additionally needs one.
        """
        _refuse_recompiles(monkeypatch)
        assert sim.remove_camera("watch")["status"] == "error"
        monkeypatch.undo()

        assert "watch" in sim.list_cameras()
        assert sim.get_camera_params(camera_name="watch") is not None

    @requires_gl
    def test_the_renderer_still_resolves_the_camera(self, sim: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """``render`` is the consumer that also needs a GL context.

        Kept apart from the registry/intrinsics pin above rather than folded in
        with it: ``render`` returns ``{'status': 'error'}`` on a host with no
        EGL/OSMesa for a reason that has nothing to do with resolving the
        camera, so asserting success unconditionally would fail off-CI with a
        bare ``'error' != 'success'`` naming neither GL nor the camera. Gating
        only this case keeps the two GL-free consumers verified everywhere.
        """
        _refuse_recompiles(monkeypatch)
        assert sim.remove_camera("watch")["status"] == "error"
        monkeypatch.undo()

        assert sim.render(camera_name="watch", width=64, height=48)["status"] == "success"

    def test_the_removal_does_not_land_at_a_later_unrelated_mutation(
        self, sim: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refused delete must not be applied by the next successful recompile.

        Pre-fix the delete sat in the live spec, so an unrelated ``add_object``
        recompiled it away: the camera vanished from the model at a call that
        never mentioned it.
        """
        _refuse_recompiles(monkeypatch)
        assert sim.remove_camera("watch")["status"] == "error"
        monkeypatch.undo()

        added = sim.add_object(name="crate", shape="box", size=[0.1, 0.1, 0.1], position=[0.4, 0, 0.05])
        assert added["status"] == "success"

        assert "watch" in _model_cameras(sim)
        assert "watch" in _spec_cameras(sim)
        assert "watch" in sim.list_cameras()

    def test_no_model_swap_means_outstanding_checkpoints_survive(
        self, sim: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refused removal installs no model, so it invalidates no checkpoint.

        ``install_compiled_model`` bumps the generation that the ``save_state`` /
        ``load_state`` fingerprint carries; a refusal never reaches it.
        """
        generation_before = sim._world._recompile_generation

        _refuse_recompiles(monkeypatch)
        assert sim.remove_camera("watch")["status"] == "error"
        monkeypatch.undo()

        assert sim._world._recompile_generation == generation_before

    def test_the_identical_removal_succeeds_once_the_refusal_clears(
        self, sim: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The scene stays mutable: the refusal cost exactly the refused call."""
        _refuse_recompiles(monkeypatch)
        assert sim.remove_camera("watch")["status"] == "error"
        monkeypatch.undo()

        assert sim.remove_camera("watch")["status"] == "success"
        assert "watch" not in sim.list_cameras()
        assert "watch" not in _spec_cameras(sim)
        assert "watch" not in _model_cameras(sim)


class TestTheHonoredRemovalIsUnchanged:
    def test_a_successful_removal_drops_it_from_registry_spec_and_model(self, sim: Any) -> None:
        """Non-vacuity: the fix does not turn a working removal into a refusal."""
        generation_before = sim._world._recompile_generation

        assert sim.remove_camera("watch")["status"] == "success"

        assert "watch" not in sim.list_cameras()
        assert "watch" not in _spec_cameras(sim)
        assert "watch" not in _model_cameras(sim)
        # The surviving camera is untouched, so the removal was scoped.
        assert "keep" in sim.list_cameras()
        assert "keep" in _model_cameras(sim)
        # A successful removal DOES swap the model, so it must invalidate
        # outstanding checkpoints - the mirror of add_camera.
        assert sim._world._recompile_generation > generation_before


class TestTheEjectHelperMirrorsItsEjectSiblings:
    def test_a_camera_the_spec_never_declared_is_not_a_failure(self, sim: Any) -> None:
        """Nothing was mutated, so there is nothing to roll back.

        Mirrors :func:`scene_ops.eject_body_from_scene`, which also reports
        success for an element the spec does not hold: the caller's registry is
        the record that matters and the spec already agrees with where it is
        heading.
        """
        assert scene_ops.eject_camera_from_scene(sim._world, "never_declared") is True
        # ... and the scene it did not touch is still exactly as it was.
        assert _spec_cameras(sim) == _model_cameras(sim)

    def test_a_refused_recompile_reports_false(self, sim: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """The bool the facade branches on is False when the scene was restored."""
        _refuse_recompiles(monkeypatch)
        result = scene_ops.eject_camera_from_scene(sim._world, "watch")
        monkeypatch.undo()

        assert result is False
        assert "watch" in _spec_cameras(sim)
