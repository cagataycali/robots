"""A camera enumeration names each camera once, however many names it answers to.

A MuJoCo camera answers to more than one name. ``add_robot`` registers a
robot's own MJCF cameras under their short alias (``wrist``) while the compiled
model holds them namespaced (``arm0/wrist``), and
``MuJoCoSimulation._camera_id`` -- which owns name resolution for every camera
surface on this backend -- maps both spellings onto one ``mjOBJ_CAMERA`` id.

``_active_camera_list`` enumerates a scene by name, and its callers spend one
unit of work per name it returns: ``render_all`` renders a frame,
``start_cameras_recording`` and ``start_cameras_recording_synchronous`` each run
an encoder writing one MP4. Enumerating by name without resolving identity
therefore captured a robot camera twice - two pixel-identical frames under two
labels, and two MP4 clips of one view.

That is precisely the outcome two sibling guards in the same call path already
refuse: ``camera_clip_name_collision_error`` for two cameras that would name one
clip, and ``name_list_error`` for a caller who repeats a name, "because a
repeated name opened a second encoder on the one output path". Neither can see
this one, because two spellings of one camera are two distinct names naming two
distinct clips.

These pin the identity rule at the enumeration: every camera is offered once,
the first spelling wins so a caller's own ordering and spelling survive, and a
name the compiled model does not answer for is never assumed to repeat another.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.models import SimCamera  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402
from tests.simulation.mujoco._gl_probe import requires_gl  # noqa: E402

#: The namespaced camera the model holds, and the short alias ``add_robot``
#: registers for it. Two names, one camera.
NAMESPACED = "arm0/wrist"
ALIAS = "wrist"


@pytest.fixture
def sim():
    s = Simulation(tool_name="camera_enumeration_test", mesh=False)
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.2)


@pytest.fixture
def aliased_scene(sim):
    """A scene holding one robot-shaped camera under both of its spellings.

    Built the way ``add_robot`` builds it - a namespaced model camera whose only
    registry entry is keyed on the short alias, with the ``SimCamera`` carrying
    the namespaced name - so the contract is exercised without depending on a
    downloadable robot asset. Dropping the key ``add_camera`` leaves behind is
    what makes the shape faithful: a robot's camera is reachable by its short
    name through the registry and by its namespaced name through the model, and
    the registry holds no namespaced key. With the free view the world always
    carries, the scene then holds exactly two cameras under three names.
    """
    sim.create_world()
    assert sim.add_camera(name=NAMESPACED, position=[0.4, 0.0, 0.5], target=[0.0, 0.0, 0.2])["status"] == "success"
    cam_id = sim._camera_id(NAMESPACED)
    sim._world.cameras[ALIAS] = SimCamera(name=NAMESPACED, camera_id=cam_id, width=64, height=48)
    del sim._world.cameras[NAMESPACED]
    assert sorted(sim._world.cameras) == ["default", ALIAS]
    return sim


class TestTheSceneWideEnumeration:
    def test_the_alias_and_the_namespaced_name_are_one_camera(self, aliased_scene) -> None:
        """The premise: dropping one of these names drops no camera from the scene."""
        assert aliased_scene._camera_id(ALIAS) == aliased_scene._camera_id(NAMESPACED) >= 0
        assert aliased_scene._world._model.ncam == 2  # the free view, plus arm0/wrist

    def test_every_camera_in_the_scene_is_offered_once(self, aliased_scene) -> None:
        """One name per camera, so a caller of "every camera" captures each once."""
        names, unresolved = aliased_scene._active_camera_list(None)
        assert unresolved == []
        ids = [aliased_scene._camera_id(n) for n in names]
        assert len(ids) == len(set(ids)), f"{names} names {len(set(ids))} camera(s)"
        assert len(names) == aliased_scene._world._model.ncam

    def test_the_namespaced_name_is_the_one_offered(self, aliased_scene) -> None:
        """The model's own spelling wins: unique per robot, and always captured under."""
        names, _ = aliased_scene._active_camera_list(None)
        assert NAMESPACED in names
        assert ALIAS not in names

    def test_a_name_the_model_does_not_answer_for_is_not_a_repeat(self, sim) -> None:
        """Unresolvable names are each kept: none can be shown to repeat another.

        Collapsing them would fold every camera the compiled model has no answer
        for into one entry, and their caller is the surface that reports them.
        """
        sim.create_world()
        for ghost in ("gone_a", "gone_b"):
            sim._world.cameras[ghost] = SimCamera(name=ghost, camera_id=-1, width=64, height=48)
            assert sim._camera_id(ghost) < 0
        names, _ = sim._active_camera_list(None)
        assert [n for n in names if n.startswith("gone_")] == ["gone_a", "gone_b"]


class TestACallerSuppliedList:
    @pytest.mark.parametrize(
        ("requested", "expected"),
        [
            pytest.param([ALIAS], [ALIAS], id="a-lone-alias-keeps-the-callers-spelling"),
            pytest.param([NAMESPACED], [NAMESPACED], id="a-lone-namespaced-name-passes-through"),
            pytest.param([ALIAS, NAMESPACED], [ALIAS], id="two-spellings-of-one-camera-collapse-to-the-first"),
            pytest.param([NAMESPACED, ALIAS], [NAMESPACED], id="collapsing-follows-the-callers-order"),
            pytest.param([NAMESPACED, "default"], [NAMESPACED, "default"], id="two-cameras-are-both-kept"),
        ],
    )
    def test_the_callers_spelling_survives_and_its_camera_is_named_once(
        self, aliased_scene, requested, expected
    ) -> None:
        """A caller is answered in their own words, with each camera named once."""
        assert aliased_scene._active_camera_list(requested) == (expected, [])


class TestTheSurfacesThatCaptureEveryCamera:
    @requires_gl
    def test_render_all_returns_one_frame_per_camera(self, aliased_scene) -> None:
        """No two frames of a snapshot are the same view under different labels."""
        result = aliased_scene.render_all(width=64, height=48)
        assert result["status"] == "success"
        summary = result["content"][0]["text"]
        assert f"{aliased_scene._world._model.ncam} requested" in summary
        labels = [b["text"] for b in result["content"][1:] if "text" in b]
        assert ALIAS not in labels
        assert len([b for b in result["content"] if "image" in b]) == aliased_scene._world._model.ncam

    @requires_gl
    def test_a_default_recording_writes_one_clip_per_camera(self, aliased_scene, tmp_path) -> None:
        """One encoder and one MP4 per camera, not per name the camera answers to."""
        start = aliased_scene.start_cameras_recording(
            output_dir=str(tmp_path), name="clip", fps=10, width=64, height=48
        )
        assert start["status"] == "success", start["content"][0]["text"]
        # The capture runs on its own thread at ``fps``, so let wall-clock time
        # pass between steps: a clip file appears only once its camera has
        # frames, which is what makes the clip list the count of encoders run.
        for _ in range(8):
            aliased_scene.step(n_steps=5)
            time.sleep(0.05)
        stop = aliased_scene.stop_cameras_recording()
        assert stop["status"] == "success"
        clips = sorted(path.name for path in tmp_path.glob("*.mp4"))
        assert clips == ["clip__arm0__wrist.mp4", "clip__default.mp4"], clips
