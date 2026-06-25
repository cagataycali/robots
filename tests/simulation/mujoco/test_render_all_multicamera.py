"""Behavior of ``Simulation.render_all`` - the multi-camera snapshot API.

``render_all`` is the multi-view counterpart to ``render``: it renders every
(or a named subset of) camera in the scene in one call and returns a single
content list of interleaved ``{"text": label}`` / ``{"image": ...}`` blocks
plus a leading summary. Several of its guarantees are pure result-shaping
logic layered on top of per-camera ``render``:

* a missing world / no cameras / unknown-camera request each return a clean
  ``status="error`` with an actionable message rather than rendering garbage;
* successful frames contribute one label + one image block, in camera order;
* a near-uniform (all-black / all-clear) frame is annotated with a
  low-variance warning so an empty render is never silently reported as ok;
* a per-camera render failure is counted and surfaced without aborting the
  whole snapshot;
* the summary tallies ok / failed / requested counts.

These pin that shaping directly. The branch tests drive ``render_all`` over a
real world but stub the per-camera ``render`` so they assert the aggregation
contract deterministically without a GPU; one GL-gated test exercises the
genuine multi-camera render end to end.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("mujoco")

os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


def _fake_render(text: str, *, variance: float = 99.0, status: str = "success", with_image: bool = True):
    """Build a stand-in ``render`` response matching production's content shape."""
    if status != "success":
        return {"status": "error", "content": [{"text": text}]}
    content: list = [{"text": text}]
    if with_image:
        content.append({"image": {"format": "png", "source": {"bytes": b"PNG"}}})
    content.append({"json": {"pixel_variance": variance, "pixel_mean": 50.0, "camera": "cam"}})
    return {"status": "success", "content": content}


@pytest.fixture
def sim():
    s = Simulation(tool_name="render_all_test", mesh=False)
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.2)


class TestRenderAllErrorPaths:
    def test_no_world_is_clean_error(self, sim: Simulation) -> None:
        """Calling render_all before create_world reports a clean error, not a crash."""
        r = sim.render_all()
        assert r["status"] == "error"
        assert "create_world" in r["content"][0]["text"]

    def test_unknown_camera_lists_available(self, sim: Simulation) -> None:
        """An explicit unknown camera name errors and names what IS available."""
        sim.create_world()
        r = sim.render_all(cameras=["does_not_exist"])
        assert r["status"] == "error"
        msg = r["content"][0]["text"]
        assert "does_not_exist" in msg
        assert "default" in msg  # available cameras are listed

    def test_no_cameras_in_scene_is_error(self, sim: Simulation, monkeypatch) -> None:
        """A scene that resolves to zero cameras yields an explicit error."""
        sim.create_world()
        monkeypatch.setattr(sim, "_active_camera_list", lambda cameras: ([], []))
        r = sim.render_all()
        assert r["status"] == "error"
        assert "No cameras" in r["content"][0]["text"]


class TestRenderAllAggregation:
    def test_each_success_yields_label_then_image_in_order(self, sim: Simulation, monkeypatch) -> None:
        """Two good cameras produce summary + (label, image) per camera, in order."""
        sim.create_world()
        monkeypatch.setattr(sim, "_active_camera_list", lambda cameras: (["cam_a", "cam_b"], []))
        monkeypatch.setattr(sim, "render", lambda camera_name, width, height: _fake_render(camera_name))

        r = sim.render_all()
        assert r["status"] == "success"
        summary = r["content"][0]["text"]
        assert "2 ok, 0 failed, 2 requested" in summary
        # Blocks after the summary: label, image, label, image.
        blocks = r["content"][1:]
        assert blocks[0]["text"] == "cam_a"
        assert "image" in blocks[1]
        assert blocks[2]["text"] == "cam_b"
        assert "image" in blocks[3]

    def test_low_variance_frame_is_flagged(self, sim: Simulation, monkeypatch) -> None:
        """A near-uniform (variance < 1) frame is annotated and tallied as low-variance."""
        sim.create_world()
        monkeypatch.setattr(sim, "_active_camera_list", lambda cameras: (["blackcam"], []))
        monkeypatch.setattr(sim, "render", lambda camera_name, width, height: _fake_render(camera_name, variance=0.0))

        r = sim.render_all()
        assert r["status"] == "success"
        assert "1 low-variance" in r["content"][0]["text"]
        # The image label carries the empty-frame warning.
        label_block = r["content"][1]
        assert "appears empty" in label_block["text"]

    def test_partial_failure_counted_and_surfaced(self, sim: Simulation, monkeypatch) -> None:
        """A failing camera is counted as failed and its error text is included,
        without aborting the other cameras; overall status stays success while
        at least one camera renders."""
        sim.create_world()
        monkeypatch.setattr(sim, "_active_camera_list", lambda cameras: (["good", "bad"], []))

        def mixed_render(camera_name, width, height):
            if camera_name == "bad":
                return _fake_render("boom: device lost", status="error")
            return _fake_render(camera_name)

        monkeypatch.setattr(sim, "render", mixed_render)
        r = sim.render_all()
        assert r["status"] == "success"  # one camera still rendered
        assert "1 ok, 1 failed, 2 requested" in r["content"][0]["text"]
        # The failed camera's error is surfaced with its name.
        texts = [b.get("text", "") for b in r["content"] if isinstance(b, dict)]
        assert any("bad" in t and "boom: device lost" in t for t in texts)

    def test_all_failed_is_error_status(self, sim: Simulation, monkeypatch) -> None:
        """If every camera fails to render, the overall status is error."""
        sim.create_world()
        monkeypatch.setattr(sim, "_active_camera_list", lambda cameras: (["a", "b"], []))
        monkeypatch.setattr(sim, "render", lambda camera_name, width, height: _fake_render("nope", status="error"))
        r = sim.render_all()
        assert r["status"] == "error"
        assert "0 ok, 2 failed" in r["content"][0]["text"]


class TestRenderAllRealRender:
    def test_two_real_cameras_render_end_to_end(self, sim: Simulation) -> None:
        """A real world with an added camera renders both views with image bytes."""
        sim.create_world()
        probe = sim.render(camera_name="default", width=64, height=48)
        if probe.get("status") != "success":
            pytest.skip(probe.get("content", [{}])[0].get("text", "render unavailable"))

        assert sim.add_camera(name="overhead", position=[0.0, 0.0, 2.0], target=[0.0, 0.0, 0.0])["status"] == "success"

        r = sim.render_all(width=64, height=48)
        assert r["status"] == "success"
        assert "2 ok, 0 failed, 2 requested" in r["content"][0]["text"]
        image_blocks = [b for b in r["content"] if isinstance(b, dict) and "image" in b]
        assert len(image_blocks) == 2
        for blk in image_blocks:
            assert blk["image"]["format"] == "png"
            assert blk["image"]["source"]["bytes"]
