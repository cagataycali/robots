"""A URDF link's pose, graded against MuJoCo's own compiler.

``load_urdf`` gave every link ``position=(0.0, 0.0, 0.0)`` and left
``BodyDef.orientation`` at its identity default, discarding the
``<joint><origin>`` that is the only place URDF states where a link sits. The
comment justifying the discard said the absolute pose would be "computed by
joint chain at instantiation time", but ``JointDef`` carries no origin, so the
offsets were not held anywhere in the returned ``ProceduralRobot`` -- a
seven-link arm and a seven-link pile at the origin were the same report, and the
load succeeded either way.

The sibling reader in the same module, ``load_mjcf``, reports both halves of a
link's pose in its parent's frame. This is the same quantity read by the same
module two different ways, and the MJCF reader's answer is the one MuJoCo
compiles.

Every expectation here is derived from ``mujoco.MjModel``: MuJoCo parses URDF,
and ``body_pos`` / ``body_quat`` are the body's frame relative to its parent --
the frame ``BodyDef.position`` reports -- so the compiler is a direct oracle and
no expected pose is restated by hand. Quaternions are compared up to sign,
because ``q`` and ``-q`` are one rotation.

Two properties of MuJoCo's URDF import shape the fixtures:

* a moving body must have mass and inertia, so every ``<link>`` carries an
  ``<inertial>``; and
* a link reached by a ``fixed`` joint is *welded* into its parent rather than
  kept as a body, so the fixtures that grade a pose directly use movable joints.
  The loader keeps every ``<link>`` as its own body, and that documented
  difference is graded by composition in
  :class:`TestAWeldedFixedLinkComposesToTheCompilersFrame`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from strands_robots.simulation.isaac.loaders import load_urdf

mujoco = pytest.importorskip("mujoco")

#: MuJoCo refuses a moving body with no mass, so every fixture link carries this.
INERTIAL = '<inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>'

#: The pose a reader that asks about no ``<origin>`` reports.
ORIGIN_POSITION = (0.0, 0.0, 0.0)
IDENTITY = (1.0, 0.0, 0.0, 0.0)

#: ``rpy`` triples covering a single axis each, a non-commuting combination (so
#: composing in the wrong order cannot pass), negative angles, and an angle past
#: pi. All are radians -- URDF has no angle-unit declaration.
RPY_DECLARATIONS = (
    "0.4 0 0",
    "0 0.4 0",
    "0 0 0.4",
    "0.3 0.2 0.1",
    "-0.7 0.5 -1.2",
    "2.9 -0.6 1.9",
)


def _write_chain(tmp_path, origins: list[tuple[str, str]], *, jtype: str = "revolute", name: str = "arm") -> str:
    """A serial chain of ``len(origins) + 1`` links, one joint per origin.

    Each entry is the ``(xyz, rpy)`` of the joint reaching the next link, so
    ``origins[i]`` places ``link{i + 1}`` in ``link{i}``'s frame.
    """
    links = [f'<link name="link0">{INERTIAL}</link>']
    joints = []
    for i, (xyz, rpy) in enumerate(origins):
        links.append(f'<link name="link{i + 1}">{INERTIAL}</link>')
        joints.append(
            f'<joint name="j{i}" type="{jtype}">'
            f'<parent link="link{i}"/><child link="link{i + 1}"/>'
            f'<origin xyz="{xyz}" rpy="{rpy}"/>'
            f'<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>'
            f"</joint>"
        )
    path = tmp_path / f"{name}.urdf"
    path.write_text(f'<robot name="{name}">{"".join(links)}{"".join(joints)}</robot>', encoding="utf-8")
    return str(path)


def _compiler_frames(path: str) -> dict[str, tuple[tuple[float, ...], tuple[float, ...]]]:
    """``{body name: (body_pos, body_quat)}`` as MuJoCo compiled the URDF.

    Keyed by name, so the map is asserted non-degenerate: an unnamed body would
    collapse every entry onto one key and silently grade nothing.
    """
    model = mujoco.MjModel.from_xml_path(path)
    frames = {}
    for i in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        assert name, f"body {i} of {path} compiled without a name; the oracle would be degenerate"
        frames[name] = (
            tuple(float(v) for v in model.body_pos[i]),
            tuple(float(v) for v in model.body_quat[i]),
        )
    return frames


def _quat_gap(a, b) -> float:
    """Largest component difference between two wxyz quaternions, up to sign."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(min(np.abs(a - b).max(), np.abs(a + b).max()))


def _mj_parent(path: str) -> dict[str, str]:
    """``{body name: parent body name}`` as MuJoCo compiled the URDF."""
    model = mujoco.MjModel.from_xml_path(path)
    out = {}
    for i in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        out[name] = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.body_parentid[i]) or "world"
    return out


class TestALinkPoseIsReadFromTheReachingJointsOrigin:
    """Both halves of the ``<origin>`` reach the child link's ``BodyDef``."""

    def test_a_translated_chain_matches_the_compiler(self, tmp_path) -> None:
        path = _write_chain(tmp_path, [("0.1 0.2 0.3", "0 0 0"), ("0 0 0.4", "0 0 0")])
        truth = _compiler_frames(path)
        reported = {b.name: b for b in load_urdf(path).bodies}
        assert set(truth) <= set(reported), (sorted(truth), sorted(reported))
        for name, (pos, _quat) in truth.items():
            assert reported[name].position == pytest.approx(pos, abs=1e-9), name

    def test_a_rotated_chain_matches_the_compiler(self, tmp_path) -> None:
        path = _write_chain(tmp_path, [("0.1 0 0", "0 0 1.2"), ("0 0.2 0", "0.3 0.2 0.1")])
        truth = _compiler_frames(path)
        reported = {b.name: b for b in load_urdf(path).bodies}
        for name, (pos, quat) in truth.items():
            assert reported[name].position == pytest.approx(pos, abs=1e-9), name
            assert _quat_gap(reported[name].orientation, quat) < 1e-9, (
                name,
                reported[name].orientation,
                quat,
            )

    @pytest.mark.parametrize("rpy", RPY_DECLARATIONS)
    def test_every_rpy_triple_matches_the_compiler(self, tmp_path, rpy: str) -> None:
        """URDF ``rpy`` is fixed-axis roll-pitch-yaw in radians.

        Composing the three rotations in the wrong order, or about the moving
        axes, disagrees with the compiler for the non-commuting triples above.
        """
        path = _write_chain(tmp_path, [("0.05 0.06 0.07", rpy)], name=f"rpy{abs(hash(rpy))}")
        quat = _compiler_frames(path)["link1"][1]
        reported = {b.name: b for b in load_urdf(path).bodies}["link1"].orientation
        assert _quat_gap(reported, quat) < 1e-9, (rpy, reported, quat)

    def test_a_prismatic_joint_places_its_child_the_same_way(self, tmp_path) -> None:
        """The origin places the link whatever the joint's own type is."""
        path = _write_chain(tmp_path, [("0 0.3 0.1", "0.2 0 0.5")], jtype="prismatic", name="slider")
        pos, quat = _compiler_frames(path)["link1"]
        link = {b.name: b for b in load_urdf(path).bodies}["link1"]
        assert link.position == pytest.approx(pos, abs=1e-9)
        assert _quat_gap(link.orientation, quat) < 1e-9

    def test_the_root_link_keeps_the_identity_pose(self, tmp_path) -> None:
        """No joint reaches the root, so nothing declares its placement."""
        path = _write_chain(tmp_path, [("0.1 0.2 0.3", "0.4 0.5 0.6")])
        root = {b.name: b for b in load_urdf(path).bodies}["link0"]
        assert root.position == ORIGIN_POSITION
        assert root.orientation == IDENTITY


class TestAWeldedFixedLinkComposesToTheCompilersFrame:
    """MuJoCo welds a ``fixed``-joint link away; the loader keeps it as a body.

    The two reports then name different parents for the link below, so they are
    reconciled by composing the loader's chain rather than compared directly.
    This is the shape every residual disagreement over the shipped URDF corpus
    turned out to have.
    """

    def test_the_composed_chain_matches_the_compiler(self, tmp_path) -> None:
        path = tmp_path / "welded.urdf"
        path.write_text(
            f'<robot name="welded">'
            f'<link name="base">{INERTIAL}</link>'
            f'<link name="mount">{INERTIAL}</link>'
            f'<link name="tip">{INERTIAL}</link>'
            f'<joint name="j0" type="revolute"><parent link="base"/><child link="mount"/>'
            f'<origin xyz="0 0 0.11" rpy="0 0 -0.7853981634"/>'
            f'<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>'
            f'<joint name="weld" type="fixed"><parent link="mount"/><child link="tip"/>'
            f'<origin xyz="0 0 0.06" rpy="0 0 0"/></joint>'
            f"</robot>",
            encoding="utf-8",
        )
        parents = _mj_parent(str(path))
        # MuJoCo re-parents twice here: the root link becomes the world body,
        # and the fixed-joint link is welded into the body above it.
        assert "base" not in parents, f"MuJoCo kept the root link as a body: {parents}"
        assert parents["mount"] == "world", parents
        assert "tip" not in parents, f"MuJoCo kept the welded link as a body: {parents}"

        reported = {b.name: b for b in load_urdf(str(path)).bodies}
        assert "tip" in reported, "the loader must keep every <link> as its own body"
        mount_pos, mount_quat = _compiler_frames(str(path))["mount"]
        assert reported["mount"].position == pytest.approx(mount_pos, abs=1e-9)
        assert _quat_gap(reported["mount"].orientation, mount_quat) < 1e-9
        # The welded link's own frame is still the one its joint declares, in
        # the parent the loader kept - which is what makes the chain composable.
        assert reported["tip"].position == pytest.approx((0.0, 0.0, 0.06), abs=1e-9)
        assert reported["tip"].orientation == pytest.approx(IDENTITY, abs=1e-9)


class TestTheFixtureIsNotVacuous:
    """Premises: each fixture really distinguishes a read from the fallback."""

    def test_every_graded_pose_differs_from_the_fallback(self, tmp_path) -> None:
        path = _write_chain(tmp_path, [("0.1 0 0", "0 0 1.2"), ("0 0.2 0", "0.3 0.2 0.1")])
        truth = _compiler_frames(path)
        assert truth, "premise: the oracle graded no body at all"
        for name, (pos, quat) in truth.items():
            assert pos != pytest.approx(ORIGIN_POSITION, abs=1e-6), (
                f"premise: {name}'s compiled position is the pre-fix fallback, so the fixture "
                f"cannot tell a read origin from a dropped one"
            )
            assert _quat_gap(quat, IDENTITY) > 1e-6, (
                f"premise: {name}'s compiled rotation is identity, the pre-fix fallback"
            )

    @pytest.mark.parametrize("rpy", RPY_DECLARATIONS)
    def test_every_rpy_triple_is_a_real_rotation(self, tmp_path, rpy: str) -> None:
        path = _write_chain(tmp_path, [("0.05 0.06 0.07", rpy)], name=f"nv{abs(hash(rpy))}")
        quat = _compiler_frames(path)["link1"][1]
        assert _quat_gap(quat, IDENTITY) > 1e-6, f"premise: rpy={rpy!r} compiles to identity"


class TestOnlyTheJointsOwnOriginIsRead:
    """The pose comes from that one element, and from both of its halves.

    A malformed ``rpy`` costs only the rotation, and the ``<origin>`` elements
    that place geometry *inside* a link are not the link's own placement.
    """

    @pytest.mark.parametrize("rpy", ["0 0", "nope 0 0", "", "0 0 0 0"])
    def test_a_malformed_rpy_reads_as_identity(self, tmp_path, rpy: str) -> None:
        """Tolerant, like ``_parse_xyz``'s reading of the sibling attribute."""
        path = _write_chain(tmp_path, [("0.1 0.2 0.3", rpy)], name=f"bad{abs(hash(rpy))}")
        link = {b.name: b for b in load_urdf(path).bodies}["link1"]
        assert link.orientation == IDENTITY
        # The xyz half is unaffected by a malformed rpy.
        assert link.position == pytest.approx((0.1, 0.2, 0.3), abs=1e-9)

    def test_an_origin_on_a_visual_or_collision_is_not_the_links_pose(self, tmp_path) -> None:
        """Those origins place geometry inside the link, not the link itself."""
        path = tmp_path / "geom_origin.urdf"
        path.write_text(
            f'<robot name="geom_origin">'
            f'<link name="link0">{INERTIAL}</link>'
            f'<link name="link1">{INERTIAL}'
            f'<visual><origin xyz="9 9 9" rpy="1 1 1"/><geometry><box size="0.1 0.1 0.1"/></geometry></visual>'
            f'<collision><origin xyz="8 8 8" rpy="1 1 1"/><geometry><box size="0.1 0.1 0.1"/></geometry></collision>'
            f"</link>"
            f'<joint name="j" type="revolute"><parent link="link0"/><child link="link1"/>'
            f'<origin xyz="0.1 0.2 0.3" rpy="0 0 0"/>'
            f'<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>'
            f"</robot>",
            encoding="utf-8",
        )
        link = {b.name: b for b in load_urdf(str(path)).bodies}["link1"]
        assert link.position == pytest.approx((0.1, 0.2, 0.3), abs=1e-9)
        assert link.orientation == IDENTITY


class TestTheReadingIsNotWidened:
    """Controls: what the loader reads is the joint's own ``<origin>``, only."""

    def test_a_joint_declaring_no_origin_leaves_the_link_at_the_parent_frame(self, tmp_path) -> None:
        path = tmp_path / "bare.urdf"
        path.write_text(
            f'<robot name="bare">'
            f'<link name="link0">{INERTIAL}</link><link name="link1">{INERTIAL}</link>'
            f'<joint name="j" type="revolute"><parent link="link0"/><child link="link1"/>'
            f'<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>'
            f"</robot>",
            encoding="utf-8",
        )
        link = {b.name: b for b in load_urdf(str(path)).bodies}["link1"]
        assert link.position == ORIGIN_POSITION
        assert link.orientation == IDENTITY

    def test_the_joints_other_readings_are_unchanged(self, tmp_path) -> None:
        """Reading the origin does not disturb axis, limits, damping or topology."""
        path = tmp_path / "joint_fields.urdf"
        path.write_text(
            f'<robot name="joint_fields">'
            f'<link name="link0">{INERTIAL}</link><link name="link1">{INERTIAL}</link>'
            f'<joint name="j" type="prismatic"><parent link="link0"/><child link="link1"/>'
            f'<origin xyz="0.1 0.2 0.3" rpy="0.4 0.5 0.6"/>'
            f'<axis xyz="0 1 0"/><limit lower="-0.25" upper="0.75" effort="1" velocity="1"/>'
            f'<dynamics damping="7.5"/></joint>'
            f"</robot>",
            encoding="utf-8",
        )
        robot = load_urdf(str(path))
        (joint,) = robot.joints
        assert joint.joint_type == "prismatic"
        assert joint.axis == (0.0, 1.0, 0.0)
        assert (joint.limit_lower, joint.limit_upper) == (-0.25, 0.75)
        assert joint.damping == pytest.approx(7.5)
        assert (joint.parent_body, joint.child_body) == (0, 1)

    def test_a_link_reached_by_two_joints_is_still_refused(self, tmp_path) -> None:
        """The tree guard runs before any pose is applied, so no origin is picked."""
        path = tmp_path / "two_parents.urdf"
        path.write_text(
            f'<robot name="two_parents">'
            f'<link name="a">{INERTIAL}</link><link name="b">{INERTIAL}</link><link name="c">{INERTIAL}</link>'
            f'<joint name="j0" type="revolute"><parent link="a"/><child link="c"/>'
            f'<origin xyz="1 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>'
            f'<limit lower="-1" upper="1" effort="1" velocity="1"/></joint>'
            f'<joint name="j1" type="revolute"><parent link="b"/><child link="c"/>'
            f'<origin xyz="0 2 0" rpy="0 0 0"/><axis xyz="0 0 1"/>'
            f'<limit lower="-1" upper="1" effort="1" velocity="1"/></joint>'
            f"</robot>",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="reached by more than one joint"):
            load_urdf(str(path))


class TestTheRpyConventionIsStated:
    """The convention is fixed-axis, not the moving-axis reading of the same triple."""

    def test_fixed_and_moving_axis_readings_differ_for_the_fixture(self, tmp_path) -> None:
        """Premise for the parametrized grading: the two readings are distinguishable."""
        roll, pitch, yaw = 0.3, 0.2, 0.1

        def about(axis, angle):
            half = angle / 2.0
            return np.array(
                [math.cos(half), *[a * math.sin(half) for a in axis]],
                dtype=float,
            )

        def mul(a, b):
            aw, ax, ay, az = a
            bw, bx, by, bz = b
            return np.array(
                [
                    aw * bw - ax * bx - ay * by - az * bz,
                    aw * bx + ax * bw + ay * bz - az * by,
                    aw * by - ax * bz + ay * bw + az * bx,
                    aw * bz + ax * by - ay * bx + az * bw,
                ],
                dtype=float,
            )

        fixed = mul(about((0, 0, 1), yaw), mul(about((0, 1, 0), pitch), about((1, 0, 0), roll)))
        moving = mul(about((1, 0, 0), roll), mul(about((0, 1, 0), pitch), about((0, 0, 1), yaw)))
        assert _quat_gap(fixed, moving) > 1e-3, "premise: the two conventions agree on this triple"

        path = _write_chain(tmp_path, [("0 0 0.1", f"{roll} {pitch} {yaw}")], name="convention")
        quat = _compiler_frames(path)["link1"][1]
        assert _quat_gap(quat, fixed) < 1e-9, "URDF rpy is the fixed-axis reading"
        reported = {b.name: b for b in load_urdf(path).bodies}["link1"].orientation
        assert _quat_gap(reported, fixed) < 1e-9
