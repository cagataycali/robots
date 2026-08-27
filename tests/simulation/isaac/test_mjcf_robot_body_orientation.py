"""A robot link's rotation, graded against MuJoCo's own compiler.

``load_mjcf`` reads each ``<body>``'s ``pos`` into ``BodyDef.position`` but read
no rotation at all, so every link of a robot the model rotates was reported
upright while ``BodyDef.orientation`` kept its identity default. Identity is a
valid orientation, so no caller could tell a link the model never rotates from
one whose rotation was dropped -- the load reports success either way.

The sibling reader in the same module, ``load_mjcf_scene_objects``, resolves all
five of MJCF's mutually exclusive spellings (``quat``, ``euler``, ``axisangle``,
``xyaxes`` and ``zaxis``) through ``_parse_orientation``. This is the same
format feature read by the same module two different ways, and the scene
reader's answer is the one MuJoCo compiles.

Every expectation here is derived from ``mujoco.MjModel``: the fixture is
compiled and the loader is compared against the ``body_quat`` the compiler
stored, so no expected quaternion is restated by hand. ``body_quat`` is the
body's frame relative to its parent, which is the same frame
``BodyDef.position`` reports, so it is a direct oracle for nested links too.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from strands_robots.simulation.isaac.loaders import load_mjcf, load_mjcf_scene_objects

mujoco = pytest.importorskip("mujoco")

#: MJCF's five mutually exclusive orientation spellings.
SPELLINGS = ("quat", "euler", "axisangle", "xyaxes", "zaxis")

#: One declaration per spelling, all naming a rotation far from identity so a
#: reader that fell back to identity cannot pass by coincidence.
DECLARATIONS = {
    "quat": "0.707106781 0 0 0.707106781",
    "euler": "30 -40 55",
    "axisangle": "0 1 0 -70",
    "xyaxes": "0 1 0 -1 0 0",
    "zaxis": "1 0 1",
}

#: The orientation a reader that asks about no spelling reports.
IDENTITY = (1.0, 0.0, 0.0, 0.0)


def _write_robot(tmp_path, body_attrs: str = "", compiler: str = "", *, name: str = "robot") -> str:
    """A one-link articulated robot, compiled by both MuJoCo and the loader.

    The link carries a hinge joint because ``load_mjcf`` refuses a model with no
    articulation, and a box geom because a mesh geom's compiled frame is not the
    frame the file declared.
    """
    path = tmp_path / f"{name}.xml"
    path.write_text(
        f"<mujoco>{compiler}<worldbody>"
        f'<body name="link" pos="0.1 0.2 0.3" {body_attrs}>'
        f'<joint name="j" type="hinge" axis="0 0 1"/>'
        f'<geom name="g" type="box" size="0.1 0.05 0.02"/>'
        f"</body></worldbody></mujoco>",
        encoding="utf-8",
    )
    return str(path)


def _write_nested_robot(tmp_path, parent_attrs: str, child_attrs: str) -> str:
    """A two-link chain, so the nested link's parent-relative frame is graded too."""
    path = tmp_path / "chain.xml"
    path.write_text(
        f'<mujoco><compiler angle="degree"/><worldbody>'
        f'<body name="link" pos="0.1 0.2 0.3" {parent_attrs}>'
        f'<joint name="j" type="hinge" axis="0 0 1"/>'
        f'<geom name="g" type="box" size="0.1 0.05 0.02"/>'
        f'<body name="forearm" pos="0.3 0 0" {child_attrs}>'
        f'<joint name="j2" type="hinge" axis="0 1 0"/>'
        f'<geom name="g2" type="box" size="0.1 0.05 0.02"/>'
        f"</body></body></worldbody></mujoco>",
        encoding="utf-8",
    )
    return str(path)


def _mujoco_body_quat(path: str, name: str = "link") -> np.ndarray:
    """The parent-relative orientation MuJoCo's compiler stored for one body."""
    model = mujoco.MjModel.from_xml_path(path)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert body > 0, f"premise: the fixture declares a body named {name!r}"
    return np.asarray(model.body_quat[body], dtype=float)


def _same_rotation(got, expected) -> float:
    """Agreement of two wxyz quaternions, treating ``q`` and ``-q`` as one rotation."""
    a = np.asarray(got, dtype=float)
    b = np.asarray(expected, dtype=float)
    return float(min(np.abs(a - b).max(), np.abs(a + b).max()))


def _link(path: str, name: str = "link"):
    """The ``BodyDef`` ``load_mjcf`` reports for one link, via the public loader."""
    robot = load_mjcf(path)
    matches = [b for b in robot.bodies if b.name == name]
    assert len(matches) == 1, f"premise: one body named {name!r}, got {[b.name for b in robot.bodies]}"
    return matches[0]


class TestEveryRotationSpellingIsReadForARobotLink:
    """The regression: every spelling was reported as identity, ``quat`` included."""

    @pytest.mark.parametrize("spelling", SPELLINGS)
    def test_a_links_orientation_matches_the_compiler(self, tmp_path, spelling):
        path = _write_robot(tmp_path, body_attrs=f'{spelling}="{DECLARATIONS[spelling]}"')
        expected = _mujoco_body_quat(path)
        assert _same_rotation(expected, IDENTITY) > 0.1, (
            "premise: the declaration is far from identity, so a reader that fell "
            "back to identity cannot pass by coincidence"
        )
        got = _link(path).orientation
        assert _same_rotation(got, expected) < 1e-6, (
            f"a link declaring {spelling}={DECLARATIONS[spelling]!r} was reported as "
            f"{tuple(round(v, 6) for v in got)}; MuJoCo compiles "
            f"{tuple(round(float(v), 6) for v in expected)}"
        )

    def test_the_position_and_the_rotation_are_read_from_the_same_element(self, tmp_path):
        """``pos`` was always read, so reporting identity dropped half of one pose."""
        path = _write_robot(tmp_path, body_attrs='euler="0 0 90"')
        link = _link(path)
        assert link.position == pytest.approx((0.1, 0.2, 0.3))
        assert _same_rotation(link.orientation, _mujoco_body_quat(path)) < 1e-6

    @pytest.mark.parametrize("spelling", SPELLINGS)
    def test_a_nested_links_parent_relative_frame_matches_the_compiler(self, tmp_path, spelling):
        """A nested link's rotation is relative to its parent, as its position is."""
        path = _write_nested_robot(tmp_path, 'euler="0 0 25"', f'{spelling}="{DECLARATIONS[spelling]}"')
        expected = _mujoco_body_quat(path, "forearm")
        assert _same_rotation(expected, IDENTITY) > 0.1, "premise: the child's rotation is far from identity"
        got = _link(path, "forearm").orientation
        assert _same_rotation(got, expected) < 1e-6, (
            f"the nested link declaring {spelling} was reported as {tuple(round(v, 6) for v in got)}; "
            f"MuJoCo compiles {tuple(round(float(v), 6) for v in expected)} relative to its parent"
        )

    def test_a_rotated_parent_does_not_absorb_its_childs_rotation(self, tmp_path):
        """Each link reports its own declaration, so the two must differ here."""
        path = _write_nested_robot(tmp_path, 'euler="0 0 90"', 'euler="0 90 0"')
        parent = _link(path, "link").orientation
        child = _link(path, "forearm").orientation
        assert _same_rotation(parent, child) > 0.1, "premise: the fixture rotates the two links differently"
        assert _same_rotation(parent, _mujoco_body_quat(path, "link")) < 1e-6
        assert _same_rotation(child, _mujoco_body_quat(path, "forearm")) < 1e-6


class TestTheAngleUnitsComeFromTheSplicedModel:
    """``<compiler angle>`` and ``eulerseq`` are model-global, so an ``<include>`` supplies them."""

    def test_the_degree_default_reads_ninety_as_a_quarter_turn(self, tmp_path):
        path = _write_robot(tmp_path, body_attrs='euler="0 0 90"')
        expected = _mujoco_body_quat(path)
        assert _same_rotation(_link(path).orientation, expected) < 1e-6
        assert _same_rotation(expected, (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))) < 1e-6

    def test_radians_are_honoured(self, tmp_path):
        path = _write_robot(tmp_path, body_attrs='euler="0 0 -1.5708"', compiler='<compiler angle="radian"/>')
        expected = _mujoco_body_quat(path)
        got = _link(path).orientation
        assert _same_rotation(got, expected) < 1e-6, (
            f"euler='0 0 -1.5708' under angle='radian' was reported as {tuple(round(v, 6) for v in got)}; "
            f"MuJoCo compiles {tuple(round(float(v), 6) for v in expected)}. Read as degrees it would be "
            "a rotation of about one degree instead of a quarter turn."
        )

    def test_radians_declared_in_an_include_are_honoured(self, tmp_path):
        (tmp_path / "units.xml").write_text('<mujoco><compiler angle="radian"/></mujoco>', encoding="utf-8")
        path = tmp_path / "robot.xml"
        path.write_text(
            '<mujoco><include file="units.xml"/><worldbody>'
            '<body name="link" pos="0.1 0.2 0.3" euler="0 0 -1.5708">'
            '<joint name="j" type="hinge" axis="0 0 1"/>'
            '<geom name="g" type="box" size="0.1 0.05 0.02"/>'
            "</body></worldbody></mujoco>",
            encoding="utf-8",
        )
        expected = _mujoco_body_quat(str(path))
        got = _link(str(path)).orientation
        assert _same_rotation(got, expected) < 1e-6, (
            "angle='radian' declared in an <include> was not honoured: reported "
            f"{tuple(round(v, 6) for v in got)}, MuJoCo compiles {tuple(round(float(v), 6) for v in expected)}"
        )

    @pytest.mark.parametrize("sequence", ["xyz", "zyx", "yxz", "XYZ", "ZYX", "YXZ"])
    def test_the_euler_sequence_and_its_case_are_honoured(self, tmp_path, sequence):
        path = _write_robot(
            tmp_path,
            body_attrs='euler="0.3 -0.7 1.1"',
            compiler=f'<compiler angle="radian" eulerseq="{sequence}"/>',
        )
        expected = _mujoco_body_quat(path)
        got = _link(path).orientation
        assert _same_rotation(got, expected) < 1e-6, (
            f"eulerseq='{sequence}' composes the three rotations in an order this reader did not follow: "
            f"reported {tuple(round(v, 6) for v in got)}, MuJoCo compiles "
            f"{tuple(round(float(v), 6) for v in expected)}"
        )


class TestTheTwoReadersOfOneModelAgree:
    """One ``<body>``, read by both loaders in this module, reports one rotation."""

    @pytest.mark.parametrize("spelling", SPELLINGS)
    def test_the_robot_and_scene_readers_report_the_same_rotation(self, tmp_path, spelling):
        path = _write_robot(tmp_path, body_attrs=f'{spelling}="{DECLARATIONS[spelling]}"')
        as_link = _link(path).orientation
        objects = [o for o in load_mjcf_scene_objects(path) if o.name == "link"]
        assert len(objects) == 1, "premise: the scene reader reports the same top-level body"
        assert _same_rotation(as_link, objects[0].quat) < 1e-6, (
            f"the robot-link reader reported {tuple(round(v, 6) for v in as_link)} for {spelling} while the "
            f"scene reader in the same module reported {tuple(round(v, 6) for v in objects[0].quat)}"
        )


class TestAnAmbiguousRotationIsRefused:
    """Two spellings on one body is a model MuJoCo refuses, so there is no rotation to pick."""

    def test_mujoco_refuses_two_orientation_specifiers(self, tmp_path):
        path = _write_robot(tmp_path, body_attrs='quat="1 0 0 0" euler="0 0 90"')
        with pytest.raises(ValueError, match="orientation"):
            mujoco.MjModel.from_xml_path(path)

    def test_the_loader_refuses_them_too(self, tmp_path):
        path = _write_robot(tmp_path, body_attrs='quat="1 0 0 0" euler="0 0 90"')
        with pytest.raises(ValueError, match="mutually exclusive"):
            load_mjcf(path)


class TestTheReportedRotationIsNotWidened:
    """Behaviour that must not move: these hold before and after the change."""

    def test_a_link_declaring_no_rotation_is_identity(self, tmp_path):
        path = _write_robot(tmp_path)
        assert _link(path).orientation == pytest.approx(IDENTITY)

    def test_the_synthetic_world_body_is_identity(self, tmp_path):
        """MJCF's world is implicit, so the reader's stand-in for it declares nothing."""
        path = _write_robot(tmp_path, body_attrs='euler="0 0 90"')
        world = [b for b in load_mjcf(path).bodies if b.name == "world"]
        assert len(world) == 1
        assert world[0].orientation == pytest.approx(IDENTITY)

    @pytest.mark.parametrize("spelling", SPELLINGS)
    def test_a_malformed_declaration_is_identity(self, tmp_path, spelling):
        path = _write_robot(tmp_path, body_attrs=f'{spelling}="not numbers at all"')
        assert _link(path).orientation == pytest.approx(IDENTITY), (
            "a value that cannot be read stays identity, the historical reading for a malformed quat"
        )

    def test_the_other_reported_link_fields_are_unchanged(self, tmp_path):
        path = _write_robot(tmp_path, body_attrs='euler="0 0 90"')
        link = _link(path)
        assert link.position == pytest.approx((0.1, 0.2, 0.3))
        assert link.shape == "box"
        assert link.shape_size == pytest.approx((0.1, 0.05, 0.02))


class TestANonUnitQuatIsNormalized:
    """A ``quat`` is reported normalized, the way MuJoCo's compiler reads it.

    A quaternion describes a rotation only at unit norm, and MJCF's idiomatic
    quarter turn ``quat="1 -1 0 0"`` is not one -- reporting the four components
    as written handed the caller a value that scales the frame it rotates. The
    reading is graded across both loaders and both readers of the shared helper
    in :mod:`tests.simulation.isaac.test_mjcf_quat_normalization`; what belongs
    here is the link-level statement, beside the other four spellings.
    """

    def test_a_non_unit_quat_is_reported_as_mujoco_compiles_it(self, tmp_path):
        path = _write_robot(tmp_path, body_attrs='quat="1 -1 0 0"')
        expected = _mujoco_body_quat(path)
        assert expected == pytest.approx((math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0))
        assert _link(path).orientation == pytest.approx(expected, abs=1e-12)
        assert _same_rotation(_link(path).orientation, expected) < 1e-12
