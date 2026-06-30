"""Regression tests: recorded LeRobotDataset videos must be OpenCV-decodable.

``start_recording`` defaults to an H.264 codec because the per-camera MP4s in a
LeRobotDataset are frequently re-opened by OpenCV-backed readers (``cv2.Video
Capture``), which is what many downstream VLM video readers use to *watch* a
recorded episode. The previous default (AV1 / ``libsvtav1``) is commonly
undecodable by the FFmpeg build bundled in OpenCV wheels: ``VideoCapture`` opens
the file, reports a non-zero frame count, yet ``read()`` yields zero frames.

A second, subtler defect this guards: every LeRobot in the supported
``>=0.5.0,<0.6.0`` range validates ``vcodec`` against a codec-name allowlist
(``{"h264", "hevc", "libsvtav1", "auto"} | HW_ENCODERS``) and rejects the ffmpeg
encoder names ("libx264"/"libx265"). A recorder that forwarded the ffmpeg name
onto the flat ``vcodec`` surface had its H.264 request rejected outright - so
the codec never took effect.

Covers:
* the default codec produces an MP4 that OpenCV decodes to exactly the recorded
  frame count (the acceptance criterion);
* an explicit ffmpeg alias ("libx264") is normalized and still OpenCV-decodable;
* opt-in AV1 ("libsvtav1") is honored (the stream really is AV1), confirming the
  storage-oriented codec remains available for callers who want it.
"""

from __future__ import annotations

import glob
import os

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("lerobot")
cv2 = pytest.importorskip("cv2")

os.environ.setdefault("MUJOCO_GL", "egl")

_ROBOT_XML = """
<mujoco model="test_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.8 1"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
  </actuator>
</mujoco>
"""

_N_STEPS = 12


@pytest.fixture
def sim_with_camera(tmp_path):
    from strands_robots.simulation import Simulation

    path = tmp_path / "test_arm.xml"
    path.write_text(_ROBOT_XML)

    s = Simulation()
    s.create_world()
    s.add_robot("arm", urdf_path=str(path))
    s.add_camera(name="base", position=[0.5, 0.0, 0.35], target=[0.2, 0.0, 0.05], width=64, height=64)
    s.step(3)
    yield s
    s.destroy()


def _record_episode(sim, dataset_dir, vcodec=None):
    from strands_robots import MockPolicy

    kwargs = dict(
        repo_id="local/codec_demo",
        root=str(dataset_dir),
        fps=30,
        task="codec round-trip",
        overwrite=True,
        cameras=["base"],
    )
    if vcodec is not None:
        kwargs["vcodec"] = vcodec
    assert sim.start_recording(**kwargs)["status"] == "success"
    sim.run_policy(robot_name="arm", policy_object=MockPolicy(), n_steps=_N_STEPS)
    sim.stop_recording()
    mp4s = glob.glob(os.path.join(str(dataset_dir), "videos", "**", "*.mp4"), recursive=True)
    assert mp4s, "no per-camera MP4 was written"
    return mp4s


def _opencv_frame_count(path):
    cap = cv2.VideoCapture(path)
    try:
        assert cap.isOpened(), f"OpenCV could not open {path}"
        decoded = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            decoded += 1
        return decoded
    finally:
        cap.release()


def _fourcc(path):
    """Return the lowercased 4-char FOURCC codec tag OpenCV reports for ``path``.

    Read from container metadata, so it works even when the active OpenCV build
    cannot decode the stream (e.g. AV1 -> "av01" with zero decodable frames).
    """
    cap = cv2.VideoCapture(path)
    try:
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
    finally:
        cap.release()
    return "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)).lower()


def test_default_codec_is_opencv_decodable(sim_with_camera, tmp_path):
    """The default-codec dataset video opens in OpenCV and yields every frame."""
    mp4s = _record_episode(sim_with_camera, tmp_path / "ds_default")
    for mp4 in mp4s:
        decoded = _opencv_frame_count(mp4)
        assert decoded == _N_STEPS, f"{mp4}: OpenCV decoded {decoded} frames, expected {_N_STEPS}"
        assert _fourcc(mp4) == "h264"


def test_libx264_alias_is_normalized_and_decodable(sim_with_camera, tmp_path):
    """An ffmpeg-style ``libx264`` request is normalized to H.264, not dropped."""
    mp4s = _record_episode(sim_with_camera, tmp_path / "ds_libx264", vcodec="libx264")
    for mp4 in mp4s:
        assert _opencv_frame_count(mp4) == _N_STEPS
        assert _fourcc(mp4) == "h264"


def test_libsvtav1_optin_is_honored(sim_with_camera, tmp_path):
    """Opt-in AV1 still works: the stream really is AV1 (storage-oriented codec)."""
    mp4s = _record_episode(sim_with_camera, tmp_path / "ds_av1", vcodec="libsvtav1")
    for mp4 in mp4s:
        assert _fourcc(mp4) == "av01"
