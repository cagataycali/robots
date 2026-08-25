"""Geometry extraction contracts for the Isaac description-file loaders.

These pin the pure-stdlib parsing paths of
:mod:`strands_robots.simulation.isaac.loaders` -- the halves that turn a
URDF / MJCF / LIBERO scene into ``ProceduralRobot`` / ``SceneObject`` values
without needing Isaac Sim, ``pxr``, or ``mujoco`` installed.

Focus areas (verified against the shipped loader behavior):
  * ``load_mjcf_scene_objects`` -- the LIBERO scene -> box-AABB extraction:
    floor / robot skipping, static-vs-movable classification, nested-body
    AABB folding, per-primitive AABB math, and the mesh-only fallback.
  * ``load_mjcf`` / ``load_urdf`` -- per-primitive shape extraction across
    box / sphere / cylinder / capsule / mesh, joint-type mapping, and the
    explicit fail-loud guards (no silent phantom robots -- see the loader
    module docstring).
  * ``load_usd`` -- the file-existence guard and the ``pxr`` import gate.
"""

from __future__ import annotations

import math

import pytest

from strands_robots.simulation.isaac.loaders import (
    _URDF_DEFAULT_JOINT_AXIS,
    SceneObject,
    load_mjcf,
    load_mjcf_scene_objects,
    load_urdf,
    load_usd,
)


def _write(tmp_path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text)
    return str(path)


# A robosuite-compiled LIBERO scene exercising every classification branch:
# floor (exact skip), robot0_* (prefix skip), a nested static fixture whose
# collision geom lives in a child body, a movable object via <freejoint>, a
# movable object via <joint type="free">, and a mesh-only body (no analytic
# geometry -> default-box fallback).
_LIBERO_SCENE = """<mujoco model="libero_task">
  <worldbody>
    <body name="floor"><geom type="plane" size="5 5 0.1"/></body>
    <body name="robot0_base" pos="0 0 0"><geom type="box" size="0.1 0.1 0.1" group="0"/></body>
    <body name="living_room_table" pos="1 0 0.4" quat="0.707 0 0 0.707">
      <geom type="box" size="0.5 0.3 0.02" group="1"/>
      <body name="table_col" pos="0 0 -0.4">
        <geom type="box" size="0.5 0.3 0.4" group="0"/>
      </body>
    </body>
    <body name="porcelain_mug_1_main" pos="1 0 0.9">
      <freejoint/>
      <geom type="cylinder" size="0.04 0.05" group="0"/>
    </body>
    <body name="mesh_only_obj" pos="2 0 0.5">
      <geom type="mesh" mesh="something"/>
    </body>
    <body name="ball_obj" pos="0 1 0.5">
      <joint type="free"/>
      <geom type="sphere" size="0.06" group="0"/>
    </body>
  </worldbody>
</mujoco>"""


class TestSceneObjectExtraction:
    """``load_mjcf_scene_objects`` -> box-AABB approximation of LIBERO scenes."""

    @pytest.fixture
    def objects(self, tmp_path) -> dict[str, SceneObject]:
        path = _write(tmp_path, "scene.xml", _LIBERO_SCENE)
        return {o.name: o for o in load_mjcf_scene_objects(path)}

    def test_floor_and_robot_bodies_are_skipped(self, objects):
        # The floor is created by create_world; the robot is added separately.
        assert "floor" not in objects
        assert "robot0_base" not in objects
        # Only the four real task bodies survive.
        assert set(objects) == {
            "living_room_table",
            "porcelain_mug_1_main",
            "mesh_only_obj",
            "ball_obj",
        }

    def test_nested_fixture_aabb_folds_child_body_offset(self, objects):
        # The table's collision geom lives in a child body offset by z=-0.4;
        # the AABB must fold that offset in (union of parent group=1 box and
        # the child group=0 box), and the world position is body pos + centre.
        table = objects["living_room_table"]
        assert table.is_static is True
        assert table.size == pytest.approx((1.0, 0.6, 0.82))
        assert table.position == pytest.approx((1.0, 0.0, 0.01))
        # The body-level quat is reported as [w, x, y, z], normalized the way
        # MuJoCo's compiler normalizes it: the fixture's rounded 0.707 spelling
        # of a quarter turn is reported as the quarter turn it means.
        assert table.quat == pytest.approx((math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)))
        # The body-frame AABB-centre offset is recorded (#1820): pose
        # appliers recompose the prim pose from a NEW body pose as
        # ``body_pos + R(body_quat) @ offset``, which is unrecoverable
        # from ``position`` alone once the body moves.
        assert table.offset == pytest.approx((0.0, 0.0, -0.39))
        assert table.position == pytest.approx(tuple(p + o for p, o in zip((1.0, 0.0, 0.4), table.offset)))

    def test_freejoint_body_is_movable_with_cylinder_aabb(self, objects):
        mug = objects["porcelain_mug_1_main"]
        assert mug.is_static is False
        # cylinder size (radius=0.04, half-length=0.05) -> full box 0.08 x 0.08 x 0.10.
        assert mug.size == pytest.approx((0.08, 0.08, 0.10))
        assert mug.position == pytest.approx((1.0, 0.0, 0.9))

    def test_free_joint_type_also_marks_movable(self, objects):
        ball = objects["ball_obj"]
        assert ball.is_static is False
        # sphere radius 0.06 -> full box 0.12 on each axis.
        assert ball.size == pytest.approx((0.12, 0.12, 0.12))

    def test_mesh_only_body_falls_back_to_default_box(self, objects):
        # No analytic collision geom -> a small default box at the body origin.
        obj = objects["mesh_only_obj"]
        assert obj.size == pytest.approx((0.05, 0.05, 0.05))
        assert obj.position == pytest.approx((2.0, 0.0, 0.5))
        assert obj.is_static is True

    def test_capsule_extends_by_radius_ellipsoid_and_short_cylinder(self, tmp_path):
        # These primitives are not covered by the main scene; pin their AABBs.
        scene = """<mujoco model="prims"><worldbody>
          <body name="cap"><freejoint/><geom type="capsule" size="0.1 0.2" group="0"/></body>
          <body name="elli"><geom type="ellipsoid" size="0.1 0.2 0.3" group="0"/></body>
          <body name="cyl1"><geom type="cylinder" size="0.05" group="0"/></body>
        </worldbody></mujoco>"""
        objs = {o.name: o for o in load_mjcf_scene_objects(_write(tmp_path, "p.xml", scene))}
        # capsule half-length 0.2 grows by radius 0.1 along z -> full z = 0.6.
        assert objs["cap"].size == pytest.approx((0.2, 0.2, 0.6))
        # ellipsoid uses its three semi-axes directly.
        assert objs["elli"].size == pytest.approx((0.2, 0.4, 0.6))
        # A cylinder carrying one size component is not a shape MuJoCo compiles
        # ("size 1 must be positive in geom") -- that spelling only means
        # something alongside ``fromto``, which supplies the axis extent. With no
        # ``fromto`` there is no segment to measure, so the loader leaves the
        # caller's fallback rather than inventing a radius-sized cube.
        assert objs["cyl1"].size == pytest.approx((0.05, 0.05, 0.05))

    def test_malformed_scene_and_missing_worldbody_fail_loud(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_mjcf_scene_objects("/nonexistent/scene.xml")
        with pytest.raises(ValueError, match="root element must be <mujoco>"):
            load_mjcf_scene_objects(_write(tmp_path, "bad.xml", "<foo/>"))
        with pytest.raises(ValueError, match="has no <worldbody>"):
            load_mjcf_scene_objects(_write(tmp_path, "nowb.xml", "<mujoco/>"))


class TestMjcfRobotLoader:
    """``load_mjcf`` -> single-robot body/joint topology + shape extraction."""

    def test_walks_bodies_and_maps_joint_types(self, tmp_path):
        mjcf = """<mujoco model="arm"><worldbody>
          <body name="base"><geom type="sphere" size="0.1"/>
            <body name="link1" pos="0 0 0.1">
              <joint name="j1" type="hinge" axis="0 0 1" range="-1 1" damping="0.5" armature="0.02"/>
              <geom type="cylinder" size="0.02 0.1"/>
              <body name="link2" pos="0 0 0.1">
                <joint name="j2" type="slide" axis="1 0 0"/>
                <geom type="box" size="0.03 0.03 0.03"/>
              </body>
            </body>
          </body>
        </worldbody></mujoco>"""
        robot = load_mjcf(_write(tmp_path, "arm.xml", mjcf))
        # A synthetic "world" root body precedes the file's bodies.
        assert [b.name for b in robot.bodies] == ["world", "base", "link1", "link2"]
        # hinge -> revolute, slide -> prismatic; parent/child resolved to indices.
        jtypes = {j.name: j.joint_type for j in robot.joints}
        assert jtypes == {"j1": "revolute", "j2": "prismatic"}
        j1 = next(j for j in robot.joints if j.name == "j1")
        assert (j1.limit_lower, j1.limit_upper) == pytest.approx((-1.0, 1.0))
        # slide joint without a <range> keeps the default +/- pi limits.
        j2 = next(j for j in robot.joints if j.name == "j2")
        assert (j2.limit_lower, j2.limit_upper) == pytest.approx((-3.14159, 3.14159))

    def test_geom_primitives_map_to_shapes(self, tmp_path):
        mjcf = """<mujoco model="shapes"><worldbody>
          <body name="a"><geom type="sphere" size="0.1"/></body>
          <body name="b"><geom type="box" size="0.03 0.04 0.05"/></body>
          <body name="c"><geom type="capsule" size="0.02 0.1"/></body>
          <body name="d"><geom type="plane" size="1 1 1"/></body>
        </worldbody></mujoco>"""
        robot = load_mjcf(_write(tmp_path, "s.xml", mjcf))
        shapes = {b.name: (b.shape, b.shape_size) for b in robot.bodies}
        assert shapes["a"] == ("sphere", (0.1,))
        assert shapes["b"] == ("box", (0.03, 0.04, 0.05))
        assert shapes["c"] == ("capsule", (0.02, 0.1))
        # non-analytic geoms (plane) degrade to a small kinematic box.
        assert shapes["d"] == ("box", (0.05, 0.05, 0.05))

    def test_unknown_joint_type_is_rejected(self, tmp_path):
        mjcf = """<mujoco><worldbody><body name="b">
          <joint name="j" type="weird"/><geom type="box" size="1 1 1"/>
        </body></worldbody></mujoco>"""
        with pytest.raises(ValueError, match="unknown joint type"):
            load_mjcf(_write(tmp_path, "j.xml", mjcf))

    def test_empty_worldbody_is_a_phantom_robot_guard(self, tmp_path):
        with pytest.raises(ValueError, match="phantom robot guard"):
            load_mjcf(_write(tmp_path, "e.xml", "<mujoco><worldbody/></mujoco>"))
        with pytest.raises(ValueError, match="root element must be <mujoco>"):
            load_mjcf(_write(tmp_path, "b.xml", "<foo/>"))

    def test_two_dof_joint_on_one_body_is_rejected_as_duplicate_edge(self, tmp_path):
        # A single <body> carrying two <joint> children (an idiomatic MJCF
        # 2-DOF compound joint, e.g. a hip with a roll and a pitch axis)
        # produces two joints that share the same (parent, child) body edge.
        # A tree articulation requires each non-root link to have exactly one
        # inbound joint, so the loader must reject this fail-fast at load time
        # rather than let it surface as a cryptic articulation error two layers
        # down. The 2-DOF axis must instead be split with an intermediate
        # massless link body (the pattern _build_unitree_g1 uses).
        mjcf = """<mujoco model="two_dof"><worldbody>
          <body name="base"><geom type="sphere" size="0.1"/>
            <body name="hip" pos="0 0 0.1">
              <joint name="hip_roll" type="hinge" axis="1 0 0"/>
              <joint name="hip_pitch" type="hinge" axis="0 1 0"/>
              <geom type="box" size="0.03 0.03 0.03"/>
            </body>
          </body>
        </worldbody></mujoco>"""
        with pytest.raises(ValueError, match="duplicate parent->child body edges") as exc:
            load_mjcf(_write(tmp_path, "two_dof.xml", mjcf))
        # The message names both offending joints so the author can locate and
        # split the compound joint from the traceback alone.
        msg = str(exc.value)
        assert "hip_roll" in msg
        assert "hip_pitch" in msg
        assert "intermediate" in msg


class TestUrdfRobotLoader:
    """``load_urdf`` -> link/joint topology, shape extraction, fail-loud guards."""

    def test_joint_types_limits_and_shapes(self, tmp_path):
        urdf = """<robot name="ur">
          <link name="base"><inertial><mass value="2.0"/></inertial>
            <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>
          <link name="l1"><visual><geometry><cylinder radius="0.02" length="0.3"/></geometry></visual></link>
          <link name="l2"><collision><geometry><sphere radius="0.05"/></geometry></collision></link>
          <link name="l3"><collision><geometry><mesh filename="x.stl"/></geometry></collision></link>
          <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
            <axis xyz="0 1 0"/><limit lower="-2" upper="2"/><dynamics damping="0.3"/></joint>
          <joint name="j2" type="continuous"><parent link="l1"/><child link="l2"/></joint>
          <joint name="j3" type="prismatic"><parent link="l2"/><child link="l3"/>
            <limit lower="0" upper="0.5"/></joint>
        </robot>"""
        robot = load_urdf(_write(tmp_path, "ur.urdf", urdf))
        joints = {j.name: j for j in robot.joints}
        # continuous collapses onto revolute; its limits stay at the default +/- pi.
        assert joints["j2"].joint_type == "revolute"
        assert (joints["j2"].limit_lower, joints["j2"].limit_upper) == pytest.approx((-3.14159, 3.14159))
        assert joints["j3"].joint_type == "prismatic"
        # j1 picks up its explicit axis, limits, and damping.
        assert joints["j1"].axis == pytest.approx((0.0, 1.0, 0.0))
        assert (joints["j1"].limit_lower, joints["j1"].limit_upper) == pytest.approx((-2.0, 2.0))
        assert joints["j1"].damping == pytest.approx(0.3)
        # Per-link shapes: box / cylinder / sphere primitives + mesh -> default box.
        shapes = {b.name: (b.shape, b.shape_size) for b in robot.bodies}
        assert shapes["base"] == ("box", (0.1, 0.1, 0.1))
        assert shapes["l1"] == ("cylinder", (0.02, 0.3))
        assert shapes["l2"] == ("sphere", (0.05,))
        assert shapes["l3"] == ("box", (0.05, 0.05, 0.05))
        # Inertial mass is read from the <mass> element (default 1.0 otherwise).
        masses = {b.name: b.mass for b in robot.bodies}
        assert masses["base"] == pytest.approx(2.0)
        assert masses["l1"] == pytest.approx(1.0)

    def test_fail_loud_guards(self, tmp_path):
        with pytest.raises(ValueError, match="root element must be <robot>"):
            load_urdf(_write(tmp_path, "root.urdf", "<foo/>"))
        with pytest.raises(ValueError, match="duplicate <link"):
            load_urdf(_write(tmp_path, "dup.urdf", '<robot name="r"><link name="a"/><link name="a"/></robot>'))
        with pytest.raises(ValueError, match="phantom robot guard"):
            load_urdf(_write(tmp_path, "empty.urdf", '<robot name="r"></robot>'))
        with pytest.raises(ValueError, match="unknown joint type"):
            load_urdf(
                _write(
                    tmp_path,
                    "jt.urdf",
                    '<robot name="r"><link name="a"/><link name="b"/>'
                    '<joint name="j" type="screw"><parent link="a"/><child link="b"/></joint></robot>',
                )
            )
        with pytest.raises(ValueError, match="unknown parent link"):
            load_urdf(
                _write(
                    tmp_path,
                    "pl.urdf",
                    '<robot name="r"><link name="a"/>'
                    '<joint name="j" type="fixed"><parent link="x"/><child link="a"/></joint></robot>',
                )
            )
        with pytest.raises(ValueError, match="malformed XML"):
            load_urdf(_write(tmp_path, "mal.urdf", "<robot name='r'><link"))


class TestUsdLoaderGate:
    """``load_usd`` guards run before any Isaac/pxr dependency is touched."""

    def test_missing_file_raises_before_import(self):
        with pytest.raises(FileNotFoundError, match="file not found"):
            load_usd("/nonexistent/stage.usd")

    def test_pxr_import_gate_has_install_hint(self, tmp_path):
        # On hosts without pxr, loading an existing file must raise ImportError
        # with an actionable install hint (never a silent phantom robot). When
        # pxr is installed the gate does not apply, so skip.
        try:
            import pxr  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("pxr is installed; the import gate does not apply")
        with pytest.raises(ImportError, match=r"usd-core") as excinfo:
            load_usd(_write(tmp_path, "stage.usda", "#usda 1.0\n"))
        # The hint must point at the real extra that ships usd-core --
        # ``strands-robots[sim-isaac]`` (see pyproject + isaac/_install.py),
        # never a nonexistent ``strands-robots-sim[isaac]`` combination that
        # would leave a following user still without the dependency.
        msg = str(excinfo.value)
        assert "strands-robots[sim-isaac]" in msg
        assert "strands-robots-sim[isaac]" not in msg
        # The pinned floor in the sim-isaac extra is usd-core>=25.5, not 24.5.
        assert "usd-core>=25.5" in msg


class TestParseFallbacks:
    """Malformed numeric attributes degrade to documented defaults, never crash.

    ``_parse_axis`` / ``_parse_xyz`` / ``_safe_float`` are the shared attribute
    parsers behind every loader. Bad input (wrong arity, non-numeric tokens)
    must fall back to the documented default rather than raise mid-parse, so a
    slightly-off description file still yields a usable kinematic structure.

    The default is the *reading format's* own, not the parser's: a URDF joint
    axis defaults to +X where an MJCF one defaults to +Z. Which default each
    loader lands on is graded against MuJoCo in
    ``test_joint_axis_format_default.py``; this class only pins that malformed
    input degrades rather than raising.
    """

    def test_axis_wrong_arity_and_non_numeric_fall_back_to_the_urdf_default(self, tmp_path):
        # A 2-component axis (wrong arity) and a non-numeric axis both fall
        # back to URDF's own default axis rather than raising.
        urdf = (
            '<robot name="r">'
            '<link name="a"/><link name="b"/><link name="c"/>'
            '<joint name="short" type="revolute"><parent link="a"/><child link="b"/>'
            '<axis xyz="1 0"/></joint>'
            '<joint name="bad" type="revolute"><parent link="b"/><child link="c"/>'
            '<axis xyz="p q r"/></joint>'
            "</robot>"
        )
        robot = load_urdf(_write(tmp_path, "axis.urdf", urdf))
        axes = {j.name: j.axis for j in robot.joints}
        assert axes["short"] == pytest.approx(_URDF_DEFAULT_JOINT_AXIS)
        assert axes["bad"] == pytest.approx(_URDF_DEFAULT_JOINT_AXIS)

    def test_position_wrong_arity_and_non_numeric_fall_back_to_origin(self, tmp_path):
        # MJCF body pos with too few components / non-numeric tokens both
        # degrade to the origin default.
        mjcf = """<mujoco><worldbody>
          <body name="short" pos="1 0"><geom type="box" size="0.1 0.1 0.1"/></body>
          <body name="bad" pos="a b c"><geom type="box" size="0.1 0.1 0.1"/></body>
        </worldbody></mujoco>"""
        robot = load_mjcf(_write(tmp_path, "pos.xml", mjcf))
        positions = {b.name: b.position for b in robot.bodies}
        assert positions["short"] == pytest.approx((0.0, 0.0, 0.0))
        assert positions["bad"] == pytest.approx((0.0, 0.0, 0.0))

    def test_non_numeric_limit_keeps_default(self, tmp_path):
        # A non-numeric URDF <limit lower=...> keeps the default -pi limit
        # (``_safe_float`` swallows the parse error).
        urdf = (
            '<robot name="r"><link name="a"/><link name="b"/>'
            '<joint name="j" type="revolute"><parent link="a"/><child link="b"/>'
            '<limit lower="foo" upper="1.0"/></joint></robot>'
        )
        robot = load_urdf(_write(tmp_path, "lim.urdf", urdf))
        j = robot.joints[0]
        assert j.limit_lower == pytest.approx(-3.14159)
        assert j.limit_upper == pytest.approx(1.0)


class TestUrdfFailLoudGuardsExtra:
    """URDF guards not exercised by the main happy-path suite."""

    def test_directory_path_is_rejected_as_not_a_regular_file(self, tmp_path):
        # A directory exists but is not a regular file -> explicit guard.
        with pytest.raises(FileNotFoundError, match="not a regular file"):
            load_urdf(str(tmp_path))

    def test_link_without_name_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="<link> without name attribute"):
            load_urdf(_write(tmp_path, "noname.urdf", '<robot name="r"><link/></robot>'))

    def test_joint_without_name_is_rejected(self, tmp_path):
        urdf = '<robot name="r"><link name="a"/><joint type="fixed"/></robot>'
        with pytest.raises(ValueError, match="<joint> without name attribute"):
            load_urdf(_write(tmp_path, "jnn.urdf", urdf))

    def test_joint_missing_parent_or_child_is_rejected(self, tmp_path):
        urdf = '<robot name="r"><link name="a"/><joint name="j" type="fixed"><parent link="a"/></joint></robot>'
        with pytest.raises(ValueError, match="missing <parent> or <child>"):
            load_urdf(_write(tmp_path, "mpc.urdf", urdf))

    def test_parent_child_missing_link_attribute_is_rejected(self, tmp_path):
        urdf = (
            '<robot name="r"><link name="a"/><link name="b"/>'
            '<joint name="j" type="fixed"><parent/><child link="b"/></joint></robot>'
        )
        with pytest.raises(ValueError, match="missing 'link' attribute"):
            load_urdf(_write(tmp_path, "mla.urdf", urdf))

    def test_unknown_child_link_is_rejected(self, tmp_path):
        urdf = (
            '<robot name="r"><link name="a"/>'
            '<joint name="j" type="fixed"><parent link="a"/><child link="ghost"/></joint></robot>'
        )
        with pytest.raises(ValueError, match="unknown child link 'ghost'"):
            load_urdf(_write(tmp_path, "ucl.urdf", urdf))

    def test_collision_without_geometry_falls_back_to_default_box(self, tmp_path):
        # A <collision> with no <geometry> child is skipped; with no <visual>
        # either, the link degrades to the default kinematic box.
        urdf = '<robot name="r"><link name="a"><collision/></link></robot>'
        robot = load_urdf(_write(tmp_path, "cng.urdf", urdf))
        body = robot.bodies[0]
        assert (body.shape, body.shape_size) == ("box", (0.05, 0.05, 0.05))


class TestMjcfShapeAndJointFallbacks:
    """``load_mjcf`` shape/joint extraction defaults for degenerate ``<geom>`` sizes."""

    def test_missing_worldbody_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="has no <worldbody>"):
            load_mjcf(_write(tmp_path, "nowb.xml", "<mujoco/>"))

    def test_inertial_mass_is_read(self, tmp_path):
        mjcf = """<mujoco><worldbody>
          <body name="b"><inertial mass="3.5"/><geom type="box" size="0.1 0.1 0.1"/></body>
        </worldbody></mujoco>"""
        robot = load_mjcf(_write(tmp_path, "inertial.xml", mjcf))
        masses = {b.name: b.mass for b in robot.bodies}
        assert masses["b"] == pytest.approx(3.5)

    def test_malformed_joint_range_keeps_default_limits(self, tmp_path):
        mjcf = """<mujoco><worldbody>
          <body name="b"><joint name="j" type="hinge" range="a b"/><geom type="box" size="0.1 0.1 0.1"/></body>
        </worldbody></mujoco>"""
        robot = load_mjcf(_write(tmp_path, "range.xml", mjcf))
        j = robot.joints[0]
        assert (j.limit_lower, j.limit_upper) == pytest.approx((-3.14159, 3.14159))

    def test_geomless_body_and_degenerate_sizes_default(self, tmp_path):
        # A body with no <geom> -> default box; malformed/short primitive
        # sizes each fall back to the primitive's default extents.
        mjcf = """<mujoco><worldbody>
          <body name="nogeom"><joint name="j" type="hinge"/></body>
          <body name="badsize"><geom type="box" size="x y z"/></body>
          <body name="shortbox"><geom type="box" size="0.1"/></body>
          <body name="nosizesphere"><geom type="sphere" size=""/></body>
          <body name="cyl1"><geom type="cylinder" size="0.03"/></body>
          <body name="cyl0"><geom type="cylinder" size=""/></body>
        </worldbody></mujoco>"""
        robot = load_mjcf(_write(tmp_path, "shapes.xml", mjcf))
        shapes = {b.name: (b.shape, b.shape_size) for b in robot.bodies}
        assert shapes["nogeom"] == ("box", (0.05, 0.05, 0.05))
        assert shapes["badsize"] == ("box", (0.05, 0.05, 0.05))
        assert shapes["shortbox"] == ("box", (0.05, 0.05, 0.05))
        assert shapes["nosizesphere"] == ("sphere", (0.05,))
        # single-size cylinder pads the half-length; no-size cylinder full default.
        assert shapes["cyl1"] == ("cylinder", (0.03, 0.05))
        assert shapes["cyl0"] == ("cylinder", (0.05, 0.05))


class TestSceneObjectGeomFallbacks:
    """``load_mjcf_scene_objects`` AABB helpers degrade cleanly on odd geoms.

    A body whose only collision geoms are non-analytic (mesh) or have
    malformed / under-specified sizes yields no analytic AABB, so the object
    falls back to the small default box instead of a NaN-sized primitive.
    """

    def test_malformed_quat_and_wrong_arity_quat_are_identity(self, tmp_path):
        scene = """<mujoco><worldbody>
          <body name="bad_quat" pos="0 0 0" quat="a b c d"><geom type="box" size="0.1 0.1 0.1" group="0"/></body>
          <body name="short_quat" pos="0 0 0" quat="1 0 0"><geom type="box" size="0.1 0.1 0.1" group="0"/></body>
        </worldbody></mujoco>"""
        objs = {o.name: o for o in load_mjcf_scene_objects(_write(tmp_path, "q.xml", scene))}
        assert objs["bad_quat"].quat == pytest.approx((1.0, 0.0, 0.0, 0.0))
        assert objs["short_quat"].quat == pytest.approx((1.0, 0.0, 0.0, 0.0))

    def test_non_analytic_and_degenerate_geoms_fall_back_to_default_box(self, tmp_path):
        # Each body's collision geometry gives no usable AABB, so all fall
        # back to the 0.05 default box at the body origin.
        scene = """<mujoco><worldbody>
          <body name="badsize"><geom type="box" size="x y" group="0"/></body>
          <body name="shortbox"><geom type="box" size="0.1" group="0"/></body>
          <body name="nosizesphere"><geom type="sphere" size="" group="0"/></body>
          <body name="shortelli"><geom type="ellipsoid" size="0.1 0.2" group="0"/></body>
          <body name="nosizecyl"><geom type="cylinder" size="" group="0"/></body>
          <body name="meshonly"><geom type="mesh" mesh="m" group="0"/></body>
        </worldbody></mujoco>"""
        objs = {o.name: o for o in load_mjcf_scene_objects(_write(tmp_path, "d.xml", scene))}
        for name in ("badsize", "shortbox", "nosizesphere", "shortelli", "nosizecyl", "meshonly"):
            assert objs[name].size == pytest.approx((0.05, 0.05, 0.05)), name


class TestMjcfFromToLinkExtent:
    """A ``fromto`` link's extent comes from its endpoints, not the size default.

    MJCF spells a capsule / cylinder two ways: ``pos`` + ``size="radius
    half-length"`` along the geom's local z, or ``fromto`` + ``size="radius"``
    with the endpoints carrying the axis extent. Both compile. Read as ``size``
    alone, the second spelling reports whatever the radius-only branch pads the
    half-length to (0.05 m) however long the segment is, so a 0.40 m link is
    published as a 0.10 m stub - the same loss ``_geom_aabb`` consults
    ``_parse_fromto`` first to avoid on the scene-object side of this module.

    ``load_mjcf`` is exported and documented (``docs/simulation/isaac.md``) and
    the ``ProceduralRobot`` it returns *is* its product, so a link extent it
    reports wrongly is the value a caller builds an articulation from.
    """

    @staticmethod
    def _shapes(tmp_path, name: str, mjcf: str) -> dict:
        robot = load_mjcf(_write(tmp_path, name, mjcf))
        return {b.name: (b.shape, b.shape_size) for b in robot.bodies}

    def test_a_fromto_capsule_link_reports_the_endpoint_half_length(self, tmp_path):
        # The spelling a robosuite-compiled scene and the Menagerie humanoids
        # use: the endpoints span 0.40 m, so the half-length is 0.20 m.
        mjcf = """<mujoco><worldbody>
          <body name="thigh"><joint name="j" type="hinge"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.4" size="0.03"/></body>
        </worldbody></mujoco>"""
        shape, size = self._shapes(tmp_path, "fromto.xml", mjcf)["thigh"]
        if shape != "capsule" or abs(size[-1] - 0.20) > 1e-12:
            raise AssertionError(
                f"'thigh' spells a 0.40 m capsule with fromto and the loader reported "
                f"({shape}, {size}) - the endpoints were not read, so the link is published "
                f"with the 0.05 m default half-length instead of 0.20 m"
            )
        assert size == pytest.approx((0.03, 0.20))

    @pytest.mark.parametrize(
        ("gtype", "fromto", "expected_half"),
        [
            ("capsule", "0 0 0 0 0 0.4", 0.20),  # along z, the common case
            ("cylinder", "0 0 0 0.6 0 0", 0.30),  # along x - a rotated axis
            ("capsule", "0 0 0 0.3 0.4 0", 0.25),  # 3-4-5 diagonal
            ("capsule", "-0.1 0 0 0.1 0 0", 0.10),  # straddling the origin
        ],
    )
    def test_the_half_length_is_half_the_segment_for_every_axis(self, tmp_path, gtype, fromto, expected_half):
        mjcf = f"""<mujoco><worldbody>
          <body name="link"><geom type="{gtype}" fromto="{fromto}" size="0.02"/></body>
        </worldbody></mujoco>"""
        got = self._shapes(tmp_path, f"{gtype}_{expected_half}.xml", mjcf)["link"]
        assert got[0] == gtype
        assert got[1] == pytest.approx((0.02, expected_half))

    def test_a_fromto_geom_without_a_size_keeps_the_default_radius(self, tmp_path):
        # ``size`` carries only the radius for this spelling, so an absent one
        # leaves the radius at the documented default - but the length is in
        # the endpoints either way and must still be reported.
        mjcf = """<mujoco><worldbody>
          <body name="link"><geom type="capsule" fromto="0 0 0 0 0 1.0"/></body>
        </worldbody></mujoco>"""
        assert self._shapes(tmp_path, "noradius.xml", mjcf)["link"][1] == pytest.approx((0.05, 0.5))

    def test_the_pos_and_size_spelling_is_unchanged(self, tmp_path):
        # The half of the contract that already worked: passes on both trees,
        # so it fails if the fromto branch is made unconditional.
        mjcf = """<mujoco><worldbody>
          <body name="a"><geom type="capsule" pos="0 0 0.2" size="0.03 0.2"/></body>
          <body name="b"><geom type="cylinder" size="0.04 0.15"/></body>
          <body name="c"><geom type="cylinder" size="0.03"/></body>
        </worldbody></mujoco>"""
        shapes = self._shapes(tmp_path, "possize.xml", mjcf)
        assert shapes["a"] == ("capsule", (0.03, 0.2))
        assert shapes["b"] == ("cylinder", (0.04, 0.15))
        # One size and no fromto still pads the half-length to the default.
        assert shapes["c"] == ("cylinder", (0.03, 0.05))

    def test_a_degenerate_fromto_keeps_the_size_reading(self, tmp_path):
        # MuJoCo refuses coincident endpoints ("fromto points too close"), so
        # there is no length to report; a zero-length link would be worse than
        # the default, and the radius reading must survive.
        mjcf = """<mujoco><worldbody>
          <body name="link"><geom type="capsule" fromto="0.1 0.2 0.3 0.1 0.2 0.3" size="0.03"/></body>
        </worldbody></mujoco>"""
        assert self._shapes(tmp_path, "degenerate.xml", mjcf)["link"][1] == pytest.approx((0.03, 0.05))

    def test_a_malformed_fromto_keeps_the_size_reading(self, tmp_path):
        # Not six numbers -> MuJoCo refuses the file outright, so there is no
        # shape to approximate and the size reading is left alone.
        mjcf = """<mujoco><worldbody>
          <body name="short"><geom type="capsule" fromto="0 0 0 0 0" size="0.03 0.4"/></body>
          <body name="words"><geom type="capsule" fromto="a b c d e f" size="0.03 0.4"/></body>
        </worldbody></mujoco>"""
        shapes = self._shapes(tmp_path, "malformed.xml", mjcf)
        assert shapes["short"][1] == pytest.approx((0.03, 0.4))
        assert shapes["words"][1] == pytest.approx((0.03, 0.4))

    def test_fromto_on_a_box_keeps_the_default_box(self, tmp_path):
        # ``fromto`` on a box or ellipsoid squares the cross-section and needs
        # the rotated-box bound, which this module deliberately does not
        # compute (the same exclusion ``_geom_aabb`` documents). Reading the
        # segment here would assert an approximation nothing derived.
        mjcf = """<mujoco><worldbody>
          <body name="bx"><geom type="box" fromto="0 0 0 0 0 0.5" size="0.02 0.03"/></body>
          <body name="el"><geom type="ellipsoid" fromto="0 0 0 0 0 0.5" size="0.02 0.03"/></body>
        </worldbody></mujoco>"""
        shapes = self._shapes(tmp_path, "boxfromto.xml", mjcf)
        assert shapes["bx"] == ("box", (0.05, 0.05, 0.05))
        assert shapes["el"] == ("box", (0.05, 0.05, 0.05))

    def test_both_fromto_consumers_read_the_length_through_one_helper(self):
        # The scene-object AABB and the robot link extent are the two places a
        # fromto segment's length is needed. Re-deriving it in either is how
        # they would drift; a shared helper is what keeps one answer.
        import inspect

        from strands_robots.simulation.isaac import loaders as mod

        for fn in (mod._extract_mjcf_shape, mod._segment_aabb):
            src = inspect.getsource(fn)
            assert "_segment_length(" in src, f"{fn.__name__} does not read the length through the shared helper"
            assert "sum(d * d" not in src, f"{fn.__name__} re-derives the segment length instead of sharing it"

        # The helper keeps the exact expression ``_segment_aabb`` used before it
        # was extracted: ``d * d`` and ``x ** 2`` disagree in the last ULP, so
        # spelling it differently would move the scene-object AABBs this shares
        # with (measured: 2 of 8000 random segments).
        assert "sum(d * d" in inspect.getsource(mod._segment_length)
