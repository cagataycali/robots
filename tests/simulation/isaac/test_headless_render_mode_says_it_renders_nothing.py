"""``render_mode="headless"`` says it renders nothing, where a consumer reads.

The default ``render_mode`` is ``"headless"`` - no rendering at all - which is a
different switch from ``headless=True`` (no window). A camera added under it
rendered all-zero frames with ``status: success`` and ``get_observation`` held
no images (measured on 6.1: mean 0.0, std 0.0), so the docs' own
``create_simulation("isaac", headless=True)`` + ``add_camera`` recipe fed a
policy or a recording black frames with nothing to say so except a text tag.
``headless=True, render_mode="rtx_realtime"`` renders real frames on the same
host. The blank frame stays (CI and GR00T server flows rely on it); what changes
is that the json says ``blank_frame`` with the remedy, and ``add_camera``'s
envelope warns at the moment the caller can still change course.
"""

from __future__ import annotations

from strands_robots.simulation.isaac.simulation import _HEADLESS_RENDER_REMEDY

from .test_render_refuses_a_camera_the_scene_does_not_carry import CAM, _engine


def _json(result: dict) -> dict:
    return next(b["json"] for b in result["content"] if "json" in b)


def test_a_headless_render_marks_its_frame_blank_and_names_the_remedy() -> None:
    result = _engine(render_mode="headless").render(camera_name=CAM)

    assert result["status"] == "success"
    payload = _json(result)
    assert payload["blank_frame"] is True
    assert payload["rtx"] is False
    assert payload["pixel_mean"] == 0.0
    assert payload["remedy"] == _HEADLESS_RENDER_REMEDY


def test_an_rtx_render_is_not_marked_blank() -> None:
    payload = _json(_engine(render_mode="rtx_realtime").render(camera_name=CAM))
    assert "blank_frame" not in payload
    assert payload["rtx"] is True


def test_the_remedy_names_the_switch_that_works() -> None:
    assert 'render_mode="rtx_realtime"' in _HEADLESS_RENDER_REMEDY
    assert "headless=True" in _HEADLESS_RENDER_REMEDY
