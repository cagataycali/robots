"""Every camera name this backend offers can be rendered through this backend.

``list_cameras`` documents its return value as "camera names accepted by
:meth:`render`", and it offers two spellings for a robot's own camera:
``add_robot`` registers the cameras a robot's MJCF declares under their SHORT
name (``wrist``) - the config-level schema ``get_observation`` publishes and
``add_robot`` reports - while the compiled model holds them namespaced
(``arm0/wrist``).

Every camera surface resolved that name for itself, and they did not agree:

* ``get_observation`` read the registered ``SimCamera`` for the namespaced name
  and published a frame under the short key.
* ``render`` / ``render_depth`` / ``get_frame`` / ``get_camera_params`` looked the
  caller's string up in the compiled model only, so the short name was refused as
  "Camera 'wrist' not found" - by a message that then listed ``wrist`` among the
  available cameras, because it reports ``_list_camera_names()``, which includes
  the registry keys.
* ``start_cameras_recording`` accepted the short name (it validates against that
  same list), reported success, and produced a clip of zero frames, because each
  capture tick went through the refusing render path.

:meth:`RenderingMixin._camera_id` now owns the mapping for all of them, so the
promise ``list_cameras`` makes holds on every surface. Names that already
resolved are unaffected: the model lookup is still tried first, and the registry
is consulted only for a name it refuses.

Robots are inline MJCF written to ``tmp_path``, so no asset download is needed.
The frame-level cells need a GL context and skip without one.
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402
from tests.simulation.mujoco._gl_probe import requires_gl  # noqa: E402

# One joint, one camera, no meshes: compiles with nothing downloaded.
_ROBOT_XML = """
<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link" pos="0 0 0.2">
      <joint name="pan" type="hinge" axis="0 0 1" range="-2 2" damping="1"/>
      <geom name="body_geom" type="box" size="0.05 0.05 0.05" rgba="0.8 0.2 0.2 1"/>
      <camera name="wrist" pos="0.4 0 0.1" xyaxes="0 -1 0 0 0 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="10"/>
  </actuator>
</mujoco>
"""

_ALIAS = "wrist"
_NAMESPACED = "arm0/wrist"


@pytest.fixture
def sim(tmp_path: Path):
    """A world holding one robot whose MJCF declares one camera."""
    path = tmp_path / "arm.xml"
    path.write_text(_ROBOT_XML)
    s = Simulation(tool_name="camera_name_resolution", mesh=False)
    try:
        assert s.create_world()["status"] == "success"
        added = s.add_robot(name="arm0", urdf_path=str(path))
        assert added["status"] == "success", added
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


def _frame(sim: Simulation, camera: str) -> np.ndarray:
    """The RGB frame :meth:`render` draws for ``camera``, as an array."""
    result = sim.render(camera_name=camera, width=160, height=120)
    assert result["status"] == "success", result
    for block in result.get("content", []):
        if "image" in block:
            raw = block["image"]["source"]["bytes"]
            if isinstance(raw, str):
                raw = base64.b64decode(raw)
            from PIL import Image

            return np.asarray(Image.open(BytesIO(raw)).convert("RGB"), dtype=np.int16)
    raise AssertionError(f"render({camera!r}) returned no image")


def test_both_spellings_of_a_robot_camera_are_offered(sim: Simulation) -> None:
    """The premise: one physical camera is listed under two names."""
    listed = sim.list_cameras()
    assert _ALIAS in listed, listed
    assert _NAMESPACED in listed, listed


@requires_gl
def test_every_camera_it_lists_can_be_rendered(sim: Simulation) -> None:
    """list_cameras documents exactly the names render accepts."""
    refused = {
        name: sim.render(camera_name=name, width=160, height=120)
        for name in sim.list_cameras()
        if sim.render(camera_name=name, width=160, height=120)["status"] != "success"
    }
    assert not refused, f"list_cameras offers names render refuses: {sorted(refused)}"


def test_the_alias_names_the_same_camera_as_its_namespaced_form(sim: Simulation) -> None:
    """The alias resolves to that camera, so the two report one pose and one K."""
    via_alias = sim.get_camera_params(camera_name=_ALIAS, width=160, height=120)
    via_model = sim.get_camera_params(camera_name=_NAMESPACED, width=160, height=120)
    assert np.array_equal(via_alias.K, via_model.K)
    assert np.array_equal(via_alias.T_world_cam, via_model.T_world_cam)


@requires_gl
def test_the_alias_draws_that_camera_and_not_the_free_view(sim: Simulation) -> None:
    """Resolving must not degrade into the scene overview under a camera's name.

    A fallback to the free camera would answer every cell above with a success
    result carrying the wrong view - the failure mode ``get_observation``'s
    schema refuses by omitting the key instead.
    """
    alias, model_name, free = (_frame(sim, _ALIAS), _frame(sim, _NAMESPACED), _frame(sim, "default"))
    # Two draws of one camera differ only by GL's own last-bit noise.
    assert np.abs(alias - model_name).max() <= 2
    assert np.abs(alias - free).max() > 32


@requires_gl
def test_a_clip_recorded_under_the_alias_carries_frames(sim: Simulation, tmp_path: Path) -> None:
    """The recorder validates against the listed names, so it must be able to draw them."""
    started = sim.start_cameras_recording(cameras=[_ALIAS], output_dir=str(tmp_path / "clips"), fps=10)
    assert started["status"] == "success", started
    sim.step(n_steps=20)
    stopped = sim.stop_cameras_recording()
    text = stopped["content"][0]["text"]
    assert "0 frames" not in text, text
    assert "0 errors" in text or "errors" not in text, text


def test_an_unknown_camera_is_refused_and_offers_only_names_that_resolve(sim: Simulation) -> None:
    """The negative control: the refusal must not list the name it is refusing."""
    refusal = sim.render(camera_name="no_such_camera", width=64, height=64)
    assert refusal["status"] == "error"
    text = refusal["content"][0]["text"]
    assert "no_such_camera" in text
    offered = [name for name in sim.list_cameras() if name in text]
    assert _ALIAS in offered, text
    for name in offered:
        # Raises KeyError for a name nothing in the model answers for, which is
        # what the refusal above was offering as an alternative.
        sim.get_camera_params(camera_name=name, width=64, height=64)
