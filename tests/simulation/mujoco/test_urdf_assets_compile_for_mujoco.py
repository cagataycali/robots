"""A ``robot_descriptions`` URDF becomes a MuJoCo asset the backend loads like a Menagerie one.

The 84 URDF-only descriptions have no MJCF, no actuators, no floor, Collada
meshes and ``package://`` mesh paths. :mod:`strands_robots.assets.urdf` turns
one into ``robot.xml`` + ``scene.xml`` + meshes on first resolution. These tests
build the fixture ``tests/fixtures/urdf/three_link.urdf`` (an STL under a
``package://`` URI, an OBJ by relative path, a hand-written Collada triangle
whose texture image does not exist, a revolute and a prismatic joint, a Gazebo
plugin element) and grade what the brief promises: the artifact compiles, the
meshes are where the model says, every joint has an actuator with the URDF's
effort as its force limit, finger joints are capped, the root is lifted onto
the floor, the counts in ``urdf_asset.json`` are the compiled model's, refusals
are the fixed sentences, and a curated name is never shadowed. No network: the
``robot_descriptions`` import is stubbed where it is reached at all.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from strands_robots.assets import urdf as urdf_assets

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("trimesh")

_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "urdf" / "three_link.urdf"


@pytest.fixture
def built(tmp_path: Path) -> tuple[Path, urdf_assets.UrdfAssetInfo]:
    dest = tmp_path / "three_link_description"
    info = urdf_assets.build_from_urdf(
        _FIXTURE,
        dest,
        name="three_link",
        module="three_link_description",
        tags={"arm"},
        package_dir=_FIXTURE.parent,
        repo_dir=_FIXTURE.parent,
    )
    return dest, info


def test_artifact_layout_is_the_menagerie_one(built: tuple[Path, urdf_assets.UrdfAssetInfo]) -> None:
    dest, _ = built
    for name in (urdf_assets.ROBOT_URDF, urdf_assets.ROBOT_XML, urdf_assets.SCENE_XML, urdf_assets.ASSET_JSON):
        assert (dest / name).is_file(), name
    meshes = sorted(p.name for p in (dest / urdf_assets.MESH_DIR).iterdir())
    # STL copied, OBJ copied, Collada converted to STL (one triangle is far below the OBJ threshold).
    assert meshes == ["base.stl", "link.obj", "tip.stl"]


def test_scene_compiles_and_steps_without_nan(built: tuple[Path, urdf_assets.UrdfAssetInfo]) -> None:
    import numpy as np

    dest, info = built
    model = mujoco.MjModel.from_xml_path(str(dest / urdf_assets.SCENE_XML))
    data = mujoco.MjData(model)
    for _ in range(200):
        mujoco.mj_step(model, data)
    assert not np.isnan(data.qpos).any()
    assert model.nu == info.nu == 2
    assert model.nq == info.nq == 2
    # The floor plane and the light the URDF never had.
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0
    assert model.nlight >= 1


def test_every_joint_has_a_position_actuator_sized_from_the_urdf_effort(
    built: tuple[Path, urdf_assets.UrdfAssetInfo],
) -> None:
    dest, info = built
    model = mujoco.MjModel.from_xml_path(str(dest / urdf_assets.ROBOT_XML))
    assert info.joints == ["shoulder", "finger_slide"]
    shoulder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "shoulder")
    finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_slide")
    assert shoulder >= 0 and finger >= 0
    # kp = effort (40) for the arm joint; position actuator means bias = -kp.
    assert model.actuator_gainprm[shoulder, 0] == pytest.approx(40.0)
    assert model.actuator_biasprm[shoulder, 1] == pytest.approx(-40.0)
    assert list(model.actuator_forcerange[shoulder]) == pytest.approx([-40.0, 40.0])
    assert list(model.actuator_ctrlrange[shoulder]) == pytest.approx([-1.5, 1.5])
    # A finger joint is capped at the finger gain even though its effort is 50.
    assert model.actuator_gainprm[finger, 0] == pytest.approx(urdf_assets._KP_FINGER_MAX)
    assert list(model.actuator_forcerange[finger]) == pytest.approx([-50.0, 50.0])
    assert list(model.actuator_ctrlrange[finger]) == pytest.approx([0.0, 0.04])


def test_fixed_base_is_not_floating_and_root_is_lifted_onto_the_floor(
    built: tuple[Path, urdf_assets.UrdfAssetInfo],
) -> None:
    dest, info = built
    model = mujoco.MjModel.from_xml_path(str(dest / urdf_assets.SCENE_XML))
    assert info.floating is False
    assert not any(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in range(model.njnt))
    # The 10 cm base box is centred on the root frame, so it starts 5 cm under
    # the floor; the builder lifts it clear by the clearance.
    assert info.root_offset_z == pytest.approx(0.05 + urdf_assets._FLOOR_CLEARANCE, abs=1e-3)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert urdf_assets._lowest_point(model, data) >= 0.0


def test_floating_tags_add_a_freejoint(tmp_path: Path) -> None:
    info = urdf_assets.build_from_urdf(
        _FIXTURE,
        tmp_path / "q",
        name="three_link",
        tags={"quadruped"},
        package_dir=_FIXTURE.parent,
        repo_dir=_FIXTURE.parent,
    )
    assert info.floating is True
    assert info.category == "mobile"
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "q" / urdf_assets.SCENE_XML))
    assert model.nq == 7 + 2  # freejoint + two actuated joints
    assert model.nu == 2


def test_asset_json_reports_the_compiled_counts_and_the_conversion(
    built: tuple[Path, urdf_assets.UrdfAssetInfo],
) -> None:
    dest, info = built
    written = json.loads((dest / urdf_assets.ASSET_JSON).read_text(encoding="utf-8"))
    assert written["nu"] == 2 and written["nq"] == 2 and written["joints"] == ["shoulder", "finger_slide"]
    assert written["converted"] == 1 and written["streamable"] is False
    assert [m["ext"] for m in written["meshes"]] == [".stl", ".obj", ".dae"]
    assert urdf_assets.read_asset_info(dest) == written
    assert info.to_dict() == written


def test_rewritten_urdf_drops_ros_only_elements_and_keeps_visual_meshes(
    built: tuple[Path, urdf_assets.UrdfAssetInfo],
) -> None:
    dest, _ = built
    text = (dest / urdf_assets.ROBOT_URDF).read_text(encoding="utf-8")
    assert "gazebo" not in text and ".so" not in text
    assert 'discardvisual="false"' in text and 'fusestatic="false"' in text
    assert "package://" not in text
    model = mujoco.MjModel.from_xml_path(str(dest / urdf_assets.ROBOT_XML))
    # Three mesh files, all referenced: the visual meshes survived.
    assert model.nmesh == 3
    # The tip link had a visual only; with fusestatic off it keeps its body.
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tip") >= 0


def test_second_build_reuses_converted_meshes(tmp_path: Path) -> None:
    dest = tmp_path / "again"
    urdf_assets.build_from_urdf(_FIXTURE, dest, name="a", tags={"arm"}, package_dir=_FIXTURE.parent)
    tip = dest / urdf_assets.MESH_DIR / "tip.stl"
    stamp = tip.stat().st_mtime_ns
    urdf_assets.build_from_urdf(_FIXTURE, dest, name="a", tags={"arm"}, package_dir=_FIXTURE.parent)
    assert tip.stat().st_mtime_ns == stamp


def test_missing_mesh_is_refused_with_the_fixed_sentence(tmp_path: Path) -> None:
    urdf = tmp_path / "bad.urdf"
    urdf.write_text(
        '<robot name="bad"><link name="a"><visual><geometry><mesh filename="package://nope/x.stl"/></geometry>'
        '</visual></link><joint name="j" type="revolute"><parent link="a"/><child link="b"/><axis xyz="0 0 1"/>'
        '</joint><link name="b"/></robot>',
        encoding="utf-8",
    )
    with pytest.raises(urdf_assets.UrdfBuildError) as excinfo:
        urdf_assets.build_from_urdf(urdf, tmp_path / "out", name="bad", package_dir=tmp_path)
    assert str(excinfo.value).startswith(urdf_assets.REFUSAL_MESH_MISSING)


def test_unknown_mesh_format_is_refused_with_the_fixed_sentence(tmp_path: Path) -> None:
    (tmp_path / "m.xyz").write_bytes(b"not a mesh")
    urdf = tmp_path / "fmt.urdf"
    urdf.write_text(
        '<robot name="fmt"><link name="a"><visual><geometry><mesh filename="m.xyz"/></geometry></visual></link>'
        '<joint name="j" type="revolute"><parent link="a"/><child link="b"/><axis xyz="0 0 1"/></joint>'
        '<link name="b"/></robot>',
        encoding="utf-8",
    )
    with pytest.raises(urdf_assets.UrdfBuildError) as excinfo:
        urdf_assets.build_from_urdf(urdf, tmp_path / "out", name="fmt")
    assert str(excinfo.value) == urdf_assets.REFUSAL_MESH_FORMAT.format(ext=".xyz")


def test_a_robot_with_no_joint_is_refused(tmp_path: Path) -> None:
    urdf = tmp_path / "static.urdf"
    urdf.write_text(
        '<robot name="s"><link name="a"><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link></robot>'
    )
    with pytest.raises(urdf_assets.UrdfBuildError) as excinfo:
        urdf_assets.build_from_urdf(urdf, tmp_path / "out", name="s")
    assert str(excinfo.value) == urdf_assets.REFUSAL_NO_JOINTS


def test_resolve_mesh_uri_covers_package_file_and_relative(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "pkg_a" / "meshes").mkdir(parents=True)
    (repo / "pkg_a" / "meshes" / "m.stl").write_bytes(b"x")
    (repo / "deep" / "pkg_b").mkdir(parents=True)
    (repo / "deep" / "pkg_b" / "n.stl").write_bytes(b"x")
    urdf_dir = repo / "pkg_a" / "urdf"
    urdf_dir.mkdir()
    (urdf_dir / "local.stl").write_bytes(b"x")
    r = urdf_assets.resolve_mesh_uri
    assert r("package://pkg_a/meshes/m.stl", urdf_dir, repo / "pkg_a", repo) == repo / "pkg_a" / "meshes" / "m.stl"
    assert r("package://pkg_b/n.stl", urdf_dir, repo / "pkg_a", repo) == repo / "deep" / "pkg_b" / "n.stl"
    assert r("local.stl", urdf_dir, repo / "pkg_a", repo) == urdf_dir / "local.stl"
    assert r(f"file://{urdf_dir / 'local.stl'}", urdf_dir, None, None) == urdf_dir / "local.stl"
    assert r("package://pkg_a/meshes/absent.stl", urdf_dir, repo / "pkg_a", repo) is None


def test_category_mapping_lets_the_base_tag_outrank_the_arm_tag() -> None:
    assert urdf_assets.category_for_tags({"dual_arm", "mobile_manipulator"}) == "mobile_manip"
    assert urdf_assets.category_for_tags({"arm"}) == "arm"
    assert urdf_assets.category_for_tags({"end_effector"}) == "hand"
    assert urdf_assets.category_for_tags({"biped"}) == "humanoid"
    assert urdf_assets.category_for_tags(set()) == "arm"
    assert urdf_assets.is_floating({"humanoid"}) and not urdf_assets.is_floating({"arm", "educational"})


def test_build_urdf_asset_reaches_the_builder_through_the_description_module(tmp_path: Path, monkeypatch) -> None:
    """The heavy entry point reads URDF_PATH/PACKAGE_PATH/REPOSITORY_PATH and the upstream pin."""
    fake = SimpleNamespace(
        URDF_PATH=str(_FIXTURE),
        PACKAGE_PATH=str(_FIXTURE.parent),
        REPOSITORY_PATH=str(_FIXTURE.parent),
    )
    monkeypatch.setattr(urdf_assets, "_description_pin", lambda mod: ("owner/three_link", "abc123"))
    from strands_robots import _description_cache

    monkeypatch.setattr(_description_cache, "import_description", lambda module: fake)
    info = urdf_assets.build_urdf_asset("three_link", "three_link_description", tmp_path)
    assert (tmp_path / "three_link_description" / urdf_assets.SCENE_XML).is_file()
    assert (info.repository, info.commit) == ("owner/three_link", "abc123")
    assert info.module == "three_link_description"


def test_download_hook_builds_a_urdf_entry_instead_of_linking(tmp_path: Path, monkeypatch) -> None:
    from strands_robots.assets import download

    calls: list[tuple[str, str, Path]] = []

    def fake_build(name: str, module: str, dest_dir: Path) -> None:
        calls.append((name, module, Path(dest_dir)))
        shutil.copy(_FIXTURE, Path(dest_dir) / "marker")

    monkeypatch.setattr(urdf_assets, "build_urdf_asset", fake_build)
    monkeypatch.setattr(download, "get_assets_dir", lambda: tmp_path)
    monkeypatch.setattr(download, "resolve_robot_name", lambda n: n)
    info = {"asset": {"dir": "x_description", "robot_descriptions_module": "x_description", "source": {"type": "urdf"}}}
    assert download.auto_download_robot("x", info) is True
    assert calls == [("x", "x_description", tmp_path)]


def test_download_hook_reports_a_refusal_as_false(tmp_path: Path, monkeypatch) -> None:
    from strands_robots.assets import download

    def refuse(name: str, module: str, dest_dir: Path) -> None:
        raise urdf_assets.UrdfBuildError(urdf_assets.REFUSAL_NO_JOINTS)

    monkeypatch.setattr(urdf_assets, "build_urdf_asset", refuse)
    monkeypatch.setattr(download, "get_assets_dir", lambda: tmp_path)
    monkeypatch.setattr(download, "resolve_robot_name", lambda n: n)
    info = {"asset": {"dir": "x_description", "robot_descriptions_module": "x_description", "source": {"type": "urdf"}}}
    assert download.auto_download_robot("x", info) is False


def test_module_does_not_import_trimesh_or_mujoco_at_import_time() -> None:
    """Both are optional; the loader names the extra only when a conversion is needed."""
    src = Path(urdf_assets.__file__).read_text(encoding="utf-8")
    head = src.split("\ndef ", 1)[0]
    assert "import trimesh" not in head and "import mujoco" not in head
    assert "sim-urdf" in urdf_assets.REFUSAL_NO_TRIMESH
    assert "strands_robots.assets.urdf" in sys.modules


_REAL_MESH = next(p for p in sorted((_FIXTURE.parent / "meshes").iterdir()) if p.suffix.lower() == ".stl")


class TestAMeshOutsideTheDescriptionTreeIsRefused:
    """A cloned description is untrusted: its meshes stay under the URDF, package and repository directories.

    Before: ``resolve_mesh_uri`` expanded ``~`` and took any absolute or
    ``file://`` path, and ``rewrite_urdf`` followed a symlink wherever it
    pointed, so ``<mesh filename="~/secret.stl"/>`` or
    ``meshes/base.dae -> ~/.ssh/...`` copied a host file into the asset cache
    with the build reporting success. ``_copy_external_tree`` already refuses
    this on the download route; the build route now holds the same line.
    """

    @staticmethod
    def _urdf(tmp_path: Path, filename: str) -> Path:
        urdf = tmp_path / "desc" / "urdf" / "r.urdf"
        urdf.parent.mkdir(parents=True, exist_ok=True)
        urdf.write_text(
            f'<robot name="r"><link name="a"><visual><geometry><mesh filename="{filename}"/></geometry></visual>'
            '</link><joint name="j" type="revolute"><parent link="a"/><child link="b"/><axis xyz="0 0 1"/></joint>'
            '<link name="b"/></robot>',
            encoding="utf-8",
        )
        return urdf

    @staticmethod
    def _host_mesh(tmp_path: Path) -> Path:
        host = tmp_path / "host" / "secret.stl"
        host.parent.mkdir(parents=True)
        shutil.copy2(_REAL_MESH, host)
        return host

    def test_a_tilde_path_is_not_expanded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        (tmp_path / "home").mkdir()
        (tmp_path / "home" / "secret.stl").write_bytes(b"solid x\nendsolid x\n")
        urdf = self._urdf(tmp_path, "~/secret.stl")
        assert urdf_assets.resolve_mesh_uri("~/secret.stl", urdf.parent, urdf.parent.parent, tmp_path / "desc") is None
        with pytest.raises(urdf_assets.UrdfBuildError) as excinfo:
            urdf_assets.build_from_urdf(urdf, tmp_path / "out", name="r", package_dir=urdf.parent.parent)
        assert str(excinfo.value).startswith(urdf_assets.REFUSAL_MESH_MISSING)
        assert not (tmp_path / "out" / "meshes").exists() or not any((tmp_path / "out" / "meshes").iterdir())

    @pytest.mark.parametrize("scheme", ["", "file://"])
    def test_an_absolute_host_path_is_refused_before_it_is_read(self, tmp_path: Path, scheme: str) -> None:
        host = self._host_mesh(tmp_path)
        urdf = self._urdf(tmp_path, f"{scheme}{host}")
        with pytest.raises(urdf_assets.UrdfBuildError) as excinfo:
            urdf_assets.build_from_urdf(
                urdf, tmp_path / "out", name="r", package_dir=urdf.parent.parent, repo_dir=tmp_path / "desc"
            )
        assert str(excinfo.value).startswith(urdf_assets.REFUSAL_MESH_OUTSIDE_TREE)
        assert not list((tmp_path / "out" / "meshes").glob("*")), "nothing of the host file reached the cache"

    def test_a_symlink_out_of_the_tree_is_judged_by_where_it_points(self, tmp_path: Path) -> None:
        host = self._host_mesh(tmp_path)
        urdf = self._urdf(tmp_path, "../meshes/base.stl")
        (tmp_path / "desc" / "meshes").mkdir()
        (tmp_path / "desc" / "meshes" / "base.stl").symlink_to(host)
        with pytest.raises(urdf_assets.UrdfBuildError) as excinfo:
            urdf_assets.build_from_urdf(
                urdf, tmp_path / "out", name="r", package_dir=tmp_path / "desc", repo_dir=tmp_path / "desc"
            )
        assert str(excinfo.value).startswith(urdf_assets.REFUSAL_MESH_OUTSIDE_TREE)

    def test_a_symlink_inside_the_tree_is_fine(self, tmp_path: Path) -> None:
        inside = tmp_path / "desc" / "shared" / "link.stl"
        inside.parent.mkdir(parents=True)
        inside.write_bytes(_REAL_MESH.read_bytes())
        urdf = self._urdf(tmp_path, "../meshes/base.stl")
        (tmp_path / "desc" / "meshes").mkdir()
        (tmp_path / "desc" / "meshes" / "base.stl").symlink_to(inside)
        info = urdf_assets.build_from_urdf(
            urdf, tmp_path / "out", name="r", package_dir=tmp_path / "desc", repo_dir=tmp_path / "desc"
        )
        assert info.meshes and info.meshes[0].source == str(inside.resolve())

    def test_the_trusted_door_is_explicit_and_a_strict_boolean(self, tmp_path: Path) -> None:
        host = self._host_mesh(tmp_path)
        urdf = self._urdf(tmp_path, str(host))
        info = urdf_assets.build_from_urdf(
            urdf, tmp_path / "out", name="r", package_dir=urdf.parent.parent, allow_outside_tree=True
        )
        assert info.meshes and info.meshes[0].source == str(host.resolve())
        with pytest.raises(ValueError, match="allow_outside_tree"):
            urdf_assets.build_from_urdf(urdf, tmp_path / "out2", name="r", allow_outside_tree="yes")  # type: ignore[arg-type]

    def test_build_urdf_asset_never_opens_the_door(self) -> None:
        import inspect

        source = inspect.getsource(urdf_assets.build_urdf_asset)
        assert "allow_outside_tree" not in source, "a cloned description never gets the trusted-caller flag"
