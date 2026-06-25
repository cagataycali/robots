"""Real-MuJoCo state-read coverage for the LIBERO observation bridge.

:meth:`LiberoAdapter.augment_observation` injects the Cartesian EEF pose
(``x/y/z/roll/pitch/yaw``) and the two-finger ``gripper`` vector that the
``libero_panda`` ``Gr00tDataConfig`` schema expects. Those values come from
two MuJoCo-backed readers:

* :meth:`LiberoAdapter._read_eef_pose` - a split-source read that takes the
  POSITION from a named gripper *site* (RoboSuite's ``eef_pos`` observable)
  and the ORIENTATION from the wrist *body* ``xquat`` (RoboSuite's
  ``eef_quat`` observable). The site sits ~10 cm below the wrist body, so
  the two sources differ; reading position from the body produced
  out-of-distribution state for LIBERO-trained checkpoints.
* :meth:`LiberoAdapter._read_gripper_qpos` - reads both finger joint qpos
  directly from ``data.qpos[jnt_qposadr]`` (opposite-sign values by physical
  convention) rather than duplicating a single finger reading.

The repository's existing split-source assertions live in
``test_libero_adapter.py`` but ``pytest.importorskip``-skip whenever no
cached LIBERO scene XML is present (the default on CI and fresh checkouts),
leaving the real-MuJoCo reader branches uncovered. This module pins them with
a self-contained inline MJCF: a wrist body carrying a named site offset ~10 cm
below it, plus two oppositely-ranged finger joints. No LIBERO assets, no
network, no cached scene cache required - only the always-available ``mujoco``
package.
"""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots.benchmarks.libero import LiberoAdapter
from strands_robots.simulation.base import SimEngine

mujoco = pytest.importorskip("mujoco")

# Inline MJCF mirroring LIBERO's kinematic layout: the gripper site
# (``gripper0_grip_site``) sits at the gripper tip, ~9.9 cm below the wrist
# body (``robot0_right_hand``). Two finger joints carry opposite-sign qpos.
_EEF_PROBE_XML = """
<mujoco model="eef_probe">
  <worldbody>
    <body name="robot0_right_hand" pos="0.3 0.0 0.5" quat="0 1 0 0">
      <joint name="gripper0_finger_joint1" type="slide" axis="1 0 0" range="0 0.04"/>
      <geom type="box" size="0.02 0.02 0.02"/>
      <body name="tip" pos="0 0 -0.097">
        <joint name="gripper0_finger_joint2" type="slide" axis="1 0 0" range="-0.04 0"/>
        <site name="gripper0_grip_site" pos="0 0 0" size="0.005"/>
        <geom type="box" size="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_PICK_CUBE_BDDL = """
(define (problem libero_eef_probe)
  (:language "pick up the cube")
  (:objects cube_1 - object)
  (:goal (on cube_1 table_1)))
"""

# Opposite-sign at-rest finger qpos, matching LIBERO's physical convention.
_FINGER1_QPOS = 0.0208
_FINGER2_QPOS = -0.0208


class _MjWorld:
    """Holds a compiled MuJoCo model/data pair under ``_model`` / ``_data``."""

    def __init__(self, model, data) -> None:
        self._model = model
        self._data = data
        self.robots: dict[str, object] = {}


class _MjSim(SimEngine):
    """Minimal ``SimEngine`` exposing a real compiled MuJoCo world.

    Only the members the LIBERO state readers touch are implemented; every
    other abstract method is a trivial stub so the class is instantiable.
    ``get_body_state`` is intentionally absent so the readers exercise their
    direct-mujoco path (the body-state fallback is covered elsewhere).
    """

    def __init__(self, model, data) -> None:
        self._world = _MjWorld(model, data)

    def create_world(self, timestep=None, gravity=None, ground_plane=True):
        return {"status": "success"}

    def destroy(self):
        return {"status": "success"}

    def reset(self):
        return {"status": "success"}

    def step(self, n_steps: int = 1):
        return {"status": "success"}

    def get_state(self):
        return {}

    def add_robot(self, name, **kw):
        return {"status": "success"}

    def remove_robot(self, name):
        return {"status": "success"}

    def list_robots(self):
        return []

    def robot_joint_names(self, robot_name):
        return []

    def add_object(self, name, **kw):
        return {"status": "success"}

    def remove_object(self, name):
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images=False):
        return {}

    def send_action(self, action, robot_name=None, n_substeps=1):
        return {"status": "success"}

    def physics_timestep(self):
        return 0.002

    def render(self, camera_name="default", width=640, height=480):
        return {"status": "success", "content": []}


@pytest.fixture
def mj_sim():
    """A sim bound to the inline EEF-probe MJCF with finger qpos preset."""
    model = mujoco.MjModel.from_xml_string(_EEF_PROBE_XML)
    data = mujoco.MjData(model)
    j1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper0_finger_joint1")
    j2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper0_finger_joint2")
    data.qpos[model.jnt_qposadr[j1]] = _FINGER1_QPOS
    data.qpos[model.jnt_qposadr[j2]] = _FINGER2_QPOS
    mujoco.mj_forward(model, data)
    return _MjSim(model, data)


def _make_adapter() -> LiberoAdapter:
    return LiberoAdapter.from_text(
        _PICK_CUBE_BDDL,
        install_cameras=False,
        auto_generate_scene=False,
        eef_body_name="robot0_right_hand",
        eef_state_site_name="gripper0_grip_site",
        state_gripper_joint_names=["gripper0_finger_joint1", "gripper0_finger_joint2"],
    )


def _ground_truth(model, data):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_right_hand")
    return (
        np.array(data.site_xpos[sid]),
        np.array(data.xpos[bid]),
        np.array(data.xquat[bid]),
    )


def test_read_eef_pose_position_from_site_orientation_from_body(mj_sim):
    """Position is read from the gripper site; orientation from the wrist body
    xquat - the split-source contract RoboSuite uses for eef_pos / eef_quat."""
    adapter = _make_adapter()
    site_pos, body_pos, body_quat = _ground_truth(mj_sim._world._model, mj_sim._world._data)

    # Sentinel: the inline scene must reproduce the ~10 cm site-vs-body gap
    # that motivated the split read, else the test is not meaningful.
    assert np.linalg.norm(site_pos - body_pos) > 0.05

    pos, quat = adapter._read_eef_pose(mj_sim)
    assert pos is not None and quat is not None

    # Position tracks the SITE, not the wrist body.
    np.testing.assert_allclose(pos, site_pos, atol=1e-6)
    assert abs(pos[2] - body_pos[2]) > 0.05  # would match body if read wrong

    # Orientation tracks the body xquat (wxyz), unit-norm.
    np.testing.assert_allclose(quat, body_quat, atol=1e-6)
    assert np.linalg.norm(np.array(quat)) == pytest.approx(1.0, abs=1e-6)


def test_read_gripper_qpos_returns_both_fingers_with_opposite_signs(mj_sim):
    """Both finger joint qpos are read directly and keep their opposite signs,
    not a single finger duplicated into ``[v, v]``."""
    adapter = _make_adapter()
    qpos = adapter._read_gripper_qpos(mj_sim)
    assert qpos is not None
    assert len(qpos) == 2
    assert qpos[0] == pytest.approx(_FINGER1_QPOS, abs=1e-6)
    assert qpos[1] == pytest.approx(_FINGER2_QPOS, abs=1e-6)
    # The defining property: the two fingers carry opposite-sign qpos.
    assert qpos[0] > 0 > qpos[1]


def test_read_gripper_qpos_returns_none_when_joint_absent(mj_sim):
    """An unknown finger joint name makes the direct read bail out with
    ``None`` so the caller can fall back to the legacy single-joint path."""
    adapter = LiberoAdapter.from_text(
        _PICK_CUBE_BDDL,
        install_cameras=False,
        auto_generate_scene=False,
        state_gripper_joint_names=["gripper0_finger_joint1", "no_such_joint"],
    )
    assert adapter._read_gripper_qpos(mj_sim) is None


def test_augment_observation_injects_full_cartesian_and_gripper_state(mj_sim):
    """End-to-end: the schema keys the libero_panda data_config expects are
    all present after augment_observation, sourced from the real sim state."""
    adapter = _make_adapter()
    site_pos, _, _ = _ground_truth(mj_sim._world._model, mj_sim._world._data)

    out = adapter.augment_observation(mj_sim, {})

    for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper"):
        assert key in out, f"missing state key {key!r}"

    # Position keys come from the site read.
    assert out["x"] == pytest.approx(float(site_pos[0]), abs=1e-6)
    assert out["y"] == pytest.approx(float(site_pos[1]), abs=1e-6)
    assert out["z"] == pytest.approx(float(site_pos[2]), abs=1e-6)

    # Gripper is the 2-vector of opposite-sign finger qpos.
    assert out["gripper"] == pytest.approx([_FINGER1_QPOS, _FINGER2_QPOS], abs=1e-6)


def test_augment_observation_flips_camera_images_vertically(mj_sim):
    """Rendered ``image`` / ``wrist_image`` frames are flipped vertically into
    upstream LIBERO's OffScreenRenderEnv (bottom-row-zero) convention, and the
    result is C-contiguous for downstream serialization."""
    adapter = _make_adapter()
    img = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    wrist = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3) + 100

    out = adapter.augment_observation(mj_sim, {"image": img, "wrist_image": wrist})

    np.testing.assert_array_equal(out["image"], np.ascontiguousarray(img[::-1, :]))
    np.testing.assert_array_equal(out["wrist_image"], np.ascontiguousarray(wrist[::-1, :]))
    assert out["image"].flags["C_CONTIGUOUS"]
    assert out["wrist_image"].flags["C_CONTIGUOUS"]


def test_augment_observation_noop_when_injection_disabled(mj_sim):
    """With ``inject_eef_state=False`` the observation passes through unchanged
    - no state keys are added even though the sim could supply them."""
    adapter = LiberoAdapter.from_text(
        _PICK_CUBE_BDDL,
        install_cameras=False,
        auto_generate_scene=False,
        inject_eef_state=False,
    )
    out = adapter.augment_observation(mj_sim, {"existing": 1})
    assert out == {"existing": 1}


# ---------------------------------------------------------------------------
# Fail-soft fallback contract for the MuJoCo-backed state readers.
#
# ``_read_eef_pose`` promises a split-source read (position from the gripper
# site, orientation from the wrist body) that degrades per-axis to
# ``sim.get_body_state`` and ultimately to ``(None, None)`` - never raising,
# so a single bad lookup only drops one observation key instead of aborting
# the eval. ``_read_gripper_qpos`` promises an all-or-nothing read that bails
# to ``None`` (so the caller falls back to its legacy single-joint path) on
# any failure. The happy-path tests above use a fully resolvable model; these
# pin the degrade branches that fire on broken / partial / non-RoboSuite
# scenes, which the cached-asset-gated suite leaves uncovered.
# ---------------------------------------------------------------------------


class _FallbackSim(SimEngine):
    """SimEngine whose ``_world`` carries arbitrary model/data plus a
    pluggable ``get_body_state`` so tests can drive the fallback branch.

    ``get_body_state`` is provided here (unlike ``_MjSim`` above, which omits
    it to force the direct path) so the body-state fallback is exercised. Pass
    ``gbs=None`` to make the fallback itself raise, pinning the
    "fallback also fails" degrade to ``(None, None)``.
    """

    def __init__(self, model, data, gbs=None) -> None:
        self._world = _MjWorld(model, data)
        self._gbs = gbs

    def get_body_state(self, body_name):
        if self._gbs is None:
            raise RuntimeError("get_body_state unavailable")
        return self._gbs(body_name)

    def create_world(self, timestep=None, gravity=None, ground_plane=True):
        return {"status": "success"}

    def destroy(self):
        return {"status": "success"}

    def reset(self):
        return {"status": "success"}

    def step(self, n_steps: int = 1):
        return {"status": "success"}

    def get_state(self):
        return {}

    def add_robot(self, name, **kw):
        return {"status": "success"}

    def remove_robot(self, name):
        return {"status": "success"}

    def list_robots(self):
        return []

    def robot_joint_names(self, robot_name):
        return []

    def add_object(self, name, **kw):
        return {"status": "success"}

    def remove_object(self, name):
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images=False):
        return {}

    def send_action(self, action, robot_name=None, n_substeps=1):
        return {"status": "success"}

    def physics_timestep(self):
        return 0.002

    def render(self, camera_name="default", width=640, height=480):
        return {"status": "success", "content": []}


def _body_state_payload(pos, quat):
    """Shape a ``get_body_state`` success payload the way the MuJoCo backend
    does: a ``content`` list whose last block carries the ``json`` dict."""
    return {
        "status": "success",
        "content": [
            {"text": "Body 'x'"},
            {"json": {"position": list(pos), "quaternion": list(quat)}},
        ],
    }


def test_read_eef_pose_unresolvable_names_fall_back_to_body_state(mj_sim):
    """When neither the site nor the body name resolves in the model, the
    reader degrades to ``sim.get_body_state`` and returns its pose - the
    documented fallback for non-RoboSuite scenes lacking the canonical site."""
    captured = {}

    def gbs(body_name):
        captured["body_name"] = body_name
        return _body_state_payload([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0])

    # Real model, but names that do not exist in it -> both direct lookups
    # return -1 and the reader must use the fallback.
    adapter = LiberoAdapter.from_text(
        _PICK_CUBE_BDDL,
        install_cameras=False,
        auto_generate_scene=False,
        eef_body_name="no_such_body",
        eef_state_site_name="no_such_site",
    )
    sim = _FallbackSim(mj_sim._world._model, mj_sim._world._data, gbs=gbs)

    pos, quat = adapter._read_eef_pose(sim)
    assert pos == [1.0, 2.0, 3.0]
    assert quat == [1.0, 0.0, 0.0, 0.0]
    # The fallback is queried for the configured EEF body name.
    assert captured["body_name"] == "no_such_body"


def test_read_eef_pose_mixes_direct_site_with_fallback_quat(mj_sim):
    """A partial resolve (site present, body absent) mixes sources per-axis:
    position comes from the direct site read, orientation from the body-state
    fallback. This is the documented "each axis populated by whichever source
    succeeded first" contract, not an all-or-nothing switch."""
    sid = mujoco.mj_name2id(mj_sim._world._model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
    site_gt = np.array(mj_sim._world._data.site_xpos[sid])

    def gbs(body_name):
        return _body_state_payload([9.0, 9.0, 9.0], [0.0, 1.0, 0.0, 0.0])

    adapter = LiberoAdapter.from_text(
        _PICK_CUBE_BDDL,
        install_cameras=False,
        auto_generate_scene=False,
        eef_body_name="no_such_body",  # body lookup fails -> quat from fallback
        eef_state_site_name="gripper0_grip_site",  # site resolves -> pos direct
    )
    sim = _FallbackSim(mj_sim._world._model, mj_sim._world._data, gbs=gbs)

    pos, quat = adapter._read_eef_pose(sim)
    # Position tracks the real site read, NOT the fallback's [9, 9, 9].
    np.testing.assert_allclose(pos, site_gt, atol=1e-6)
    # Orientation comes from the fallback because the body name is unresolvable.
    assert quat == [0.0, 1.0, 0.0, 0.0]


def test_read_eef_pose_returns_none_pair_when_everything_fails():
    """No resolvable model and a failing ``get_body_state`` yields
    ``(None, None)`` rather than raising - the caller then injects no EEF
    state keys for that step."""
    adapter = LiberoAdapter.from_text(
        _PICK_CUBE_BDDL,
        install_cameras=False,
        auto_generate_scene=False,
        eef_body_name="robot0_right_hand",
        eef_state_site_name="gripper0_grip_site",
    )
    # Non-mujoco "model"/"data" objects: mj_name2id raises TypeError, caught
    # internally; the fallback then raises too (gbs=None).
    sim = _FallbackSim(object(), object(), gbs=None)
    assert adapter._read_eef_pose(sim) == (None, None)


def test_read_gripper_qpos_none_when_joint_lookup_raises():
    """A model object that ``mj_name2id`` cannot consume (TypeError) is caught
    and the gripper read bails to ``None`` so the caller's legacy single-joint
    fallback takes over - the read never propagates the error."""
    adapter = LiberoAdapter.from_text(
        _PICK_CUBE_BDDL,
        install_cameras=False,
        auto_generate_scene=False,
        state_gripper_joint_names=["gripper0_finger_joint1", "gripper0_finger_joint2"],
    )
    sim = _FallbackSim(object(), object())
    assert adapter._read_gripper_qpos(sim) is None


def test_read_gripper_qpos_none_when_qpos_index_fails(mj_sim):
    """With a real model but a data object lacking ``qpos``, the joint names
    resolve yet the per-finger qpos read raises IndexError/AttributeError,
    which is caught and collapses the whole read to ``None``."""

    class _NoQposData:
        pass

    adapter = LiberoAdapter.from_text(
        _PICK_CUBE_BDDL,
        install_cameras=False,
        auto_generate_scene=False,
        state_gripper_joint_names=["gripper0_finger_joint1", "gripper0_finger_joint2"],
    )
    sim = _FallbackSim(mj_sim._world._model, _NoQposData())
    assert adapter._read_gripper_qpos(sim) is None


def test_read_gripper_qpos_none_when_no_joint_names_configured(mj_sim):
    """An empty ``state_gripper_joint_names`` (e.g. a scene with no gripper)
    short-circuits to ``None`` before any model lookup."""
    adapter = LiberoAdapter.from_text(
        _PICK_CUBE_BDDL,
        install_cameras=False,
        auto_generate_scene=False,
        state_gripper_joint_names=[],
    )
    assert adapter._read_gripper_qpos(mj_sim) is None


def test_read_eef_pose_inner_array_read_failure_falls_back(mj_sim):
    """When the site/body NAMES resolve against the model but the per-array
    read (``data.site_xpos`` / ``data.xquat``) raises - e.g. a data object
    that lacks those arrays - each inner read is caught and the reader
    degrades to the body-state fallback rather than propagating the error."""

    class _NoArraysData:
        pass

    def gbs(body_name):
        return _body_state_payload([5.0, 5.0, 5.0], [0.0, 0.0, 1.0, 0.0])

    adapter = LiberoAdapter.from_text(
        _PICK_CUBE_BDDL,
        install_cameras=False,
        auto_generate_scene=False,
        eef_body_name="robot0_right_hand",  # resolves in model -> name2id ok
        eef_state_site_name="gripper0_grip_site",  # resolves in model -> name2id ok
    )
    # Real model (name lookups succeed) but a data object without site_xpos /
    # xquat (the array reads raise AttributeError, caught internally).
    sim = _FallbackSim(mj_sim._world._model, _NoArraysData(), gbs=gbs)

    pos, quat = adapter._read_eef_pose(sim)
    assert pos == [5.0, 5.0, 5.0]
    assert quat == [0.0, 0.0, 1.0, 0.0]
