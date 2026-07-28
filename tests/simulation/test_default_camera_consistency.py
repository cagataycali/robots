"""Regression tests: one camera name must mean exactly one view.

``docs/simulation/overview.md`` documents ``"default"`` as "the built-in
``default`` free view", and ``render`` / ``render_depth`` / ``get_frame`` /
``get_camera_params`` all resolve that token to MuJoCo's free camera
(``cam_id = -1``). But ``create_world`` ALSO registers a ``SimCamera`` named
``"default"``, which the spec builder compiles into a real MJCF camera at
``[1.5, 1.5, 1.2]``.

``_get_sim_observation`` resolved camera names with a plain ``mj_name2id``, so it
picked up that MJCF camera - and the same declared key returned two different
images depending on which API you called:

    render("default")      marker at 0.687 of frame width
    get_observation()      marker at 0.355 of frame width
    mean |difference|      35.6 / 255

That silently decoupled a rollout video from the LeRobot dataset recorded beside
it: an eval MP4 and the training frames showed different viewpoints under one
camera key, so a policy trained on the dataset was evaluated against a video of a
different view. These tests pin that both paths agree, and that a genuinely named
camera stays a distinct view.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
Image = pytest.importorskip("PIL.Image")

from strands_robots.simulation.mujoco.rendering import _FREE_CAMERA_TOKENS  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="default_camera_consistency", mesh=False)
    s.create_world()
    # A saturated red marker gives an unambiguous per-view fingerprint.
    s.add_object(name="mark", shape="box", size=[0.04] * 3, position=[0.6, 0, 0.3], color=[1, 0, 0, 1])
    s.add_robot(name="panda")
    yield s
    s.destroy()


def _png_to_array(result) -> np.ndarray:
    for block in result["content"]:
        if "image" in block:
            return np.array(Image.open(io.BytesIO(block["image"]["source"]["bytes"])).convert("RGB"))
    raise AssertionError("render() returned no image block")


def _marker_centroid(frame: np.ndarray) -> tuple[float, float]:
    """Normalized (x, y) centroid of the red marker - a view fingerprint."""
    red = (frame[:, :, 0].astype(int) - frame[:, :, 1].astype(int) - frame[:, :, 2].astype(int)) > 60
    assert red.any(), "marker not visible in frame"
    ys, xs = np.nonzero(red)
    return float(xs.mean()) / frame.shape[1], float(ys.mean()) / frame.shape[0]


def test_create_world_really_compiles_a_camera_named_default(sim) -> None:
    """Pin the premise: the name collision genuinely exists in the model."""
    model = sim.mj_model
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
    assert "default" in names


def test_default_token_set_covers_the_documented_aliases() -> None:
    for token in (None, "", "default", "free"):
        assert token in _FREE_CAMERA_TOKENS


def test_observation_and_render_agree_for_default(sim) -> None:
    """The core defect: one key must not yield two views."""
    obs_frame = sim.get_observation("panda")["default"]
    height, width = obs_frame.shape[0], obs_frame.shape[1]
    rendered = _png_to_array(sim.render(camera_name="default", width=width, height=height))

    assert rendered.shape == obs_frame.shape
    # Pre-fix: centroids were (0.687, 0.514) vs (0.355, 0.574), mean diff 35.6.
    assert _marker_centroid(rendered) == pytest.approx(_marker_centroid(obs_frame), abs=1e-3)
    assert np.abs(rendered.astype(float) - obs_frame.astype(float)).mean() < 1.0


def test_named_camera_also_agrees_across_paths(sim) -> None:
    """A user-added camera must be consistent too (it always was; guard it)."""
    sim.add_camera(name="side", position=[0.2, -1.2, 0.5], target=[0.6, 0, 0.3])
    obs_frame = sim.get_observation("panda")["side"]
    height, width = obs_frame.shape[0], obs_frame.shape[1]
    rendered = _png_to_array(sim.render(camera_name="side", width=width, height=height))
    assert np.abs(rendered.astype(float) - obs_frame.astype(float)).mean() < 1.0


def test_named_camera_is_a_different_view_from_default(sim) -> None:
    """The fix must not collapse every camera onto the free view."""
    sim.add_camera(name="side", position=[0.2, -1.2, 0.5], target=[0.6, 0, 0.3])
    obs = sim.get_observation("panda")
    assert _marker_centroid(obs["side"]) != pytest.approx(_marker_centroid(obs["default"]), abs=1e-3)


# --- D34: the free camera must not win positional image fill -------------------
#
# ``_get_sim_observation`` built ``cameras_to_render`` from model cameras in MJCF
# declaration order, which puts the auto-created ``default`` view (registered
# unconditionally by ``create_world``) ahead of every camera the user added. Dict
# insertion order then handed ``default`` to the policy FIRST, so
# ``LerobotLocalPolicy``'s positional image fill gave the 640x480 free-camera view
# the first declared image slot and the task-relevant wrist frame was dropped:
#
#     image keys IN ORDER: [('default', (480, 640, 3)), ('wrist', (224, 224, 3))]
#     observation.images.top <- SOURCE CAMERA 'default'; wrist never reached the model
#
# Only ORDER changes - every camera is still rendered under its own name, and the
# recorded dataset schema (keyed by name) is unaffected.


def _image_keys_in_order(sim, robot_name: str = "panda") -> list[str]:
    obs = sim.get_observation(robot_name)
    return [key for key, value in obs.items() if isinstance(value, np.ndarray) and value.ndim == 3]


class TestFreeCameraIsOrderedLast:
    def test_a_user_camera_comes_before_default(self, sim):
        """The regression: 'default' was first and took the first image slot."""
        assert (
            sim.add_camera("wrist", position=[0.3, 0.0, 0.3], target=[0.0, 0.0, 0.1], width=64, height=64)["status"]
            == "success"
        )

        keys = _image_keys_in_order(sim)

        assert keys, "no image keys in the observation"
        assert keys[0] == "wrist", keys
        assert keys[-1] == "default", keys

    def test_every_camera_is_still_present(self, sim):
        """Ordering only: no camera may be dropped."""
        for index, name in enumerate(("wrist", "top", "side")):
            assert (
                sim.add_camera(name, position=[0.3, 0.1 * index, 0.3], target=[0.0, 0.0, 0.1], width=64, height=64)[
                    "status"
                ]
                == "success"
            )

        keys = _image_keys_in_order(sim)

        assert set(keys) == {"wrist", "top", "side", "default"}

    def test_user_cameras_keep_their_relative_order(self, sim):
        """A stable sort: only the free camera moves."""
        for index, name in enumerate(("wrist", "top", "side")):
            assert (
                sim.add_camera(name, position=[0.3, 0.1 * index, 0.3], target=[0.0, 0.0, 0.1], width=64, height=64)[
                    "status"
                ]
                == "success"
            )

        keys = _image_keys_in_order(sim)

        assert keys == ["wrist", "top", "side", "default"], keys

    def test_a_scene_with_only_the_default_camera_still_yields_it(self, sim):
        """Deprioritized, not discarded."""
        keys = _image_keys_in_order(sim)

        assert keys == ["default"], keys

    def test_the_frames_are_unchanged_by_the_reorder(self, sim):
        """Each name must still carry ITS OWN view, not a shuffled one."""
        # Framed on the red marker at [0.6, 0, 0.3] so the centroid fingerprint
        # the module's helpers use is available in this camera too.
        assert (
            sim.add_camera("wrist", position=[1.2, 0.0, 0.4], target=[0.6, 0.0, 0.3], width=64, height=64)["status"]
            == "success"
        )

        obs = sim.get_observation("panda")

        # render() is the independent reference for each camera.
        assert obs["wrist"].shape == (64, 64, 3)
        assert obs["default"].shape[:2] != (64, 64), "default took the user camera's resolution"
        wrist_render = _png_to_array(sim.render(camera_name="wrist", width=64, height=64))
        assert _marker_centroid(obs["wrist"]) == pytest.approx(_marker_centroid(wrist_render), abs=0.02)


class TestPositionalFillPrefersARealCamera:
    """Defence in depth on the policy side.

    The sim now yields the free camera last, but an observation can reach the
    packer from any source (a replayed dataset, a hand-built dict, a benchmark
    adapter). If ``default`` is first in THAT dict it would take the first
    declared slot again, so the positional-fill loop deprioritizes it too.
    """

    def _policy(self):
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

        policy = LerobotLocalPolicy.__new__(LerobotLocalPolicy)
        policy.positional_fallback_used = False
        policy.strict_keys = False
        policy.camera_key_map = None
        policy.robot_state_keys = ["j1"]
        # One declared image feature whose name matches NO camera, forcing the
        # positional path - the only path where source order decides routing.
        policy._input_features = {"observation.images.top": object(), "observation.state": object()}
        policy._obs_rename = {}
        policy._image_resize_warned = set()
        policy._action_dim_warned = False
        return policy

    def _route(self, policy, sources):
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

        observation = {"j1": 0.0}
        observation.update(sources)
        out = LerobotLocalPolicy._to_lerobot_observation(policy, observation)
        routed = out.get("observation.images.top")
        assert routed is not None, "nothing was routed to the declared slot"
        for name, frame in sources.items():
            if routed.shape == frame.shape:
                return name
        raise AssertionError("routed frame matches no source shape")

    def test_a_real_camera_wins_even_when_default_is_first(self):
        free = np.zeros((480, 640, 3), dtype=np.uint8)
        wrist = np.zeros((224, 224, 3), dtype=np.uint8)

        routed = self._route(self._policy(), {"default": free, "wrist": wrist})

        assert routed == "wrist"

    def test_a_real_camera_still_wins_when_it_is_first(self):
        free = np.zeros((480, 640, 3), dtype=np.uint8)
        wrist = np.zeros((224, 224, 3), dtype=np.uint8)

        routed = self._route(self._policy(), {"wrist": wrist, "default": free})

        assert routed == "wrist"

    def test_the_free_camera_is_used_when_it_is_the_only_source(self):
        """Deprioritized, not discarded - a free-camera-only scene must still run."""
        free = np.zeros((480, 640, 3), dtype=np.uint8)

        routed = self._route(self._policy(), {"default": free})

        assert routed == "default"

    def test_positional_fallback_is_still_flagged(self):
        """The reorder must not hide that positional routing happened."""
        policy = self._policy()
        self._route(
            policy,
            {"default": np.zeros((480, 640, 3), dtype=np.uint8), "wrist": np.zeros((224, 224, 3), dtype=np.uint8)},
        )

        assert policy.positional_fallback_used is True
