"""The ``microduck`` asset compiles to the shape ``robots.json`` declares.

The registry-side companion (``tests/registry``) grades the entry's fields from
``robots.json`` alone. It cannot compile the asset: that directory's conftest
repoints ``STRANDS_ASSETS_DIR`` at a per-test temp dir for host isolation, so no
downloaded asset is ever visible there. These cells live here so the declared
figures are checked against the model that supplies them.

Both the joint order and the pose are stated once, in the registry-side module,
and imported here - so there is one statement of each and this file grades it,
rather than two copies that agree until one is edited.

The order matters beyond bookkeeping. A consumer that reads joint positions as a
flat ``qpos[7:21]`` slice gets exactly this order from the declared model, and
upstream ships a variant that does not share it:
``robot_allcollisions_rollers.xml`` inserts two passive wheel joints after
``left_ankle``, which moves nine of the fourteen actuated joints to a different
``qpos`` index, so the same slice reads different joints there. The actuator
order is identical across the variants, so a policy writing ``ctrl`` is
unaffected - only a position read is.
:class:`TestTheEntryPointsAtTheDocumentedLayout` pins that the entry keeps
naming the fourteen-hinge model, which is the one the catalog describes.

The asset is not downloaded here. Fetching it would clone an external repository
during a test run and turn a host with no network into a failure rather than a
skip, so the search paths are read directly and every cell skips when the asset
is absent - which is the case on a clean CI checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.registry.test_microduck_declares_its_asset_shape import (
    DOCUMENTED_ORDER,
    FLOATING_BASE_JOINTS,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ROBOTS_JSON = REPO_ROOT / "strands_robots" / "registry" / "robots.json"

#: The ``STAND`` pose in radians, in :data:`DOCUMENTED_ORDER`, quoted to the four
#: decimals the catalog states. :data:`POSE_TOLERANCE` is that rounding.
DOCUMENTED_STAND_POSE: tuple[float, ...] = (
    0.0,
    -0.0873,
    -0.4579,
    -0.0049,
    0.4530,
    0.3491,
    0.3491,
    0.0,
    0.0,
    0.0,
    0.0873,
    0.4579,
    0.0049,
    -0.4530,
)

POSE_TOLERANCE = 5e-5


def _entry() -> dict:
    return json.loads(ROBOTS_JSON.read_text(encoding="utf-8"))["robots"]["microduck"]


def _asset_file(key: str) -> Path:
    """Return a declared asset file present on disk, or skip.

    Reads the search paths rather than calling
    :func:`~strands_robots.assets.manager.resolve_model_path`, which downloads a
    missing asset. A test that fetches 26 MB from a third-party repository fails
    on a host with no network instead of skipping, and does it once per cell.
    """
    from strands_robots.utils import get_search_paths

    asset = _entry()["asset"]
    present = next(
        (candidate for root in get_search_paths() if (candidate := Path(root) / asset["dir"] / asset[key]).exists()),
        None,
    )
    if present is None:
        pytest.skip(f"microduck {key} is not downloaded, so the compiled shape cannot be read")
    return present


def _model(key: str = "scene_xml"):
    mujoco = pytest.importorskip("mujoco")
    return mujoco, mujoco.MjModel.from_xml_path(str(_asset_file(key)))


def _joint_names(mujoco, model) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]


class TestTheDeclaredShapeMatchesTheCompiledAsset:
    """The declared figures, re-derived from the model they describe."""

    def test_the_declared_count_is_the_models_joint_total(self) -> None:
        """``joints: 15`` is the model's ``njnt``, floating base included."""
        mujoco, model = _model()
        assert model.njnt == _entry()["joints"]

    def test_every_documented_joint_is_driven_by_one_actuator(self) -> None:
        mujoco, model = _model()
        assert model.nu == len(DOCUMENTED_ORDER)
        driven = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i, 0]))
            for i in range(model.nu)
        )
        assert driven == DOCUMENTED_ORDER

    def test_the_position_slice_reads_the_documented_order(self) -> None:
        """A flat ``qpos[7:]`` read gets :data:`DOCUMENTED_ORDER`, in order."""
        mujoco, model = _model()
        hinges = tuple(
            name
            for name, kind in zip(_joint_names(mujoco, model), model.jnt_type, strict=True)
            if int(kind) == int(mujoco.mjtJoint.mjJNT_HINGE)
        )
        assert hinges == DOCUMENTED_ORDER

    def test_the_base_is_the_only_joint_no_actuator_drives(self) -> None:
        mujoco, model = _model()
        free = [k for k in model.jnt_type if int(k) == int(mujoco.mjtJoint.mjJNT_FREE)]
        assert len(free) == FLOATING_BASE_JOINTS
        assert model.njnt == model.nu + FLOATING_BASE_JOINTS


class TestTheShippedStandKeyframeIsTheDocumentedPose:
    """The pose is read from the asset, so an upstream retune fails here."""

    def test_the_keyframe_is_still_named_stand(self) -> None:
        mujoco, model = _model()
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(model.nkey)]
        assert "STAND" in names, f"asset no longer ships a STAND keyframe: {names}"

    def test_the_keyframe_holds_the_documented_pose(self) -> None:
        mujoco, model = _model()
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(model.nkey)]
        pose = model.key_qpos[names.index("STAND")][-len(DOCUMENTED_ORDER) :]
        for joint, actual, documented in zip(DOCUMENTED_ORDER, pose, DOCUMENTED_STAND_POSE, strict=True):
            assert abs(float(actual) - documented) < POSE_TOLERANCE, joint

    def test_the_keyframe_commands_the_pose_it_holds(self) -> None:
        """``ctrl`` equals ``qpos`` there, so spawning at STAND commands STAND."""
        mujoco, model = _model()
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(model.nkey)]
        index = names.index("STAND")
        pose = model.key_qpos[index][-len(DOCUMENTED_ORDER) :]
        command = model.key_ctrl[index]
        for joint, held, commanded in zip(DOCUMENTED_ORDER, pose, command, strict=True):
            assert abs(float(held) - float(commanded)) < POSE_TOLERANCE, joint


class TestTheEntryPointsAtTheDocumentedLayout:
    """The entry must not name a variant that renumbers the position slice."""

    def test_the_declared_model_carries_no_passive_wheel(self) -> None:
        """The rollers variant adds wheels mid-list, moving nine joints.

        Nothing here objects to that variant - a caller can load it by path.
        This is about what a bare ``Robot("microduck")`` resolves, which is the
        model whose ``qpos`` layout the catalog documents.
        """
        mujoco, model = _model()
        assert not [name for name in _joint_names(mujoco, model) if "wheel" in name]

    def test_the_model_and_the_scene_describe_one_robot(self) -> None:
        """``model_xml`` and ``scene_xml`` agree, so either entry point is safe."""
        mujoco, scene = _model("scene_xml")
        _, bare = _model("model_xml")
        assert _joint_names(mujoco, bare) == _joint_names(mujoco, scene)

    def test_only_the_scene_carries_the_pose_keyframes(self) -> None:
        """The bare model ships none, which is why the entry declares a scene."""
        mujoco, scene = _model("scene_xml")
        _, bare = _model("model_xml")
        assert bare.nkey == 0
        assert scene.nkey > 0
