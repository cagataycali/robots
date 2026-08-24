"""An MJCF ``quat`` is reported normalized, as MuJoCo's compiler reads it.

A quaternion describes a rotation only at unit norm. MJCF authors routinely
spell one non-unit -- ``quat="1 -1 0 0"`` is the idiomatic quarter turn, and the
shipped asset corpus uses it on hundreds of robot links -- and MuJoCo's compiler
normalizes what it reads. The loaders reported the four components as written,
so what reached the caller was not a rotation: MuJoCo's own ``mju_quat2Mat``
builds a matrix from the components as given, and that matrix scales by
``|q|**2`` (twice size for the quarter turn above) on top of rotating.

Every expectation here is derived from ``mujoco.MjModel``: the fixture is
compiled and the loader compared against the ``body_quat`` the compiler stored,
so no expected quaternion is restated by hand. The fixtures are shared with the
orientation-spelling suite next door, because normalizing is a property of the
one helper both readers resolve a ``quat`` through rather than of either reader.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from strands_robots.utils import coerce_orientation_quaternion

from .test_mjcf_orientation_spellings import DECLARATIONS, _only, _write, _write_mesh_scene
from .test_mjcf_robot_body_orientation import _link, _write_robot

mujoco = pytest.importorskip("mujoco")

#: ``quat`` declarations MuJoCo compiles, none of them a unit quaternion: the
#: idiomatic quarter turn, a scaled identity, an unnormalized diagonal, a
#: rounded quarter turn (the shape a hand-written asset produces) and a
#: Pythagorean triple whose norm is exactly 5.
NON_UNIT = ("1 -1 0 0", "2 0 0 0", "1 1 1 1", "0.707 0 0 0.707", "3 0 -4 0")

#: Unit declarations, which must be reported unchanged rather than re-scaled.
ALREADY_UNIT = ("1 0 0 0", "0.5 0.5 0.5 0.5", "0 1 0 0")

#: ``quat`` values no reader can use, each already read as identity: the wrong
#: number of components, or components that are not numbers at all.
MALFORMED = ("1 0 0", "1 0 0 0 0", "not numbers", "")

#: The one unusable value that has four numeric components. It has no direction
#: to normalize onto, so it joins :data:`MALFORMED` rather than being scaled.
ZERO = "0 0 0 0"

IDENTITY = (1.0, 0.0, 0.0, 0.0)


def _compiled_body_quat(path: str, name: str) -> np.ndarray:
    """The orientation MuJoCo's compiler stored for the fixture's body."""
    model = mujoco.MjModel.from_xml_path(path)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert body >= 0, f"premise: the fixture declares a body named {name!r}"
    return np.asarray(model.body_quat[body], dtype=float)


def _rotation_matrix(quat) -> np.ndarray:
    """The matrix MuJoCo builds from a quaternion, without normalizing it first.

    ``mju_quat2Mat`` is what a consumer applying a reported orientation reaches
    for, and it uses the components as given -- which is why the norm of the
    reported value is load-bearing rather than cosmetic.
    """
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(quat, dtype=float))
    return rot.reshape(3, 3)


def _norm(quat) -> float:
    return math.sqrt(sum(float(c) * float(c) for c in quat))


class TestTheFixturesAreWhatTheyClaim:
    """Non-vacuity: the declarations really are non-unit, unit and unusable."""

    @pytest.mark.parametrize("declaration", NON_UNIT)
    def test_a_non_unit_declaration_is_not_already_normalized(self, declaration):
        deviation = abs(_norm(declaration.split()) - 1.0)
        assert deviation > 1e-6, (
            f"premise: {declaration!r} differs from unit norm by {deviation}, far above the 1e-12 the "
            "comparisons below allow, so normalizing it moves the reported value"
        )

    @pytest.mark.parametrize("declaration", ALREADY_UNIT)
    def test_a_unit_declaration_is_already_normalized(self, declaration):
        assert _norm(declaration.split()) == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("declaration", NON_UNIT)
    def test_mujoco_compiles_every_non_unit_declaration(self, tmp_path, declaration):
        """The oracle exists: MuJoCo reads these models rather than refusing them."""
        quat = _compiled_body_quat(_write_robot(tmp_path, body_attrs=f'quat="{declaration}"'), "link")
        assert _norm(quat) == pytest.approx(1.0, abs=1e-12), "premise: MuJoCo's own answer is a unit quaternion"


class TestARobotLinksQuatIsNormalized:
    """The regression for ``load_mjcf``: a link's reported ``quat`` was the file's."""

    @pytest.mark.parametrize("declaration", NON_UNIT)
    def test_the_reported_orientation_is_the_one_mujoco_compiled(self, tmp_path, declaration):
        path = _write_robot(tmp_path, body_attrs=f'quat="{declaration}"')
        assert _link(path).orientation == pytest.approx(_compiled_body_quat(path, "link"), abs=1e-12)

    @pytest.mark.parametrize("declaration", NON_UNIT)
    def test_the_reported_orientation_is_a_unit_quaternion(self, tmp_path, declaration):
        path = _write_robot(tmp_path, body_attrs=f'quat="{declaration}"')
        assert _norm(_link(path).orientation) == pytest.approx(1.0, abs=1e-12)

    def test_the_reported_orientation_rotates_without_scaling(self, tmp_path):
        """What the norm costs a caller: the frame the reported value describes.

        A quaternion applied as a rotation without being normalized first scales
        the frame by ``|q|**2``. The quarter turn below is spelled at norm
        ``sqrt(2)``, so the matrix built from the components as written doubles
        every axis -- a link twice its size, from an orientation.
        """
        path = _write_robot(tmp_path, body_attrs='quat="1 -1 0 0"')
        as_written = _rotation_matrix((1.0, -1.0, 0.0, 0.0))
        assert np.linalg.norm(as_written @ np.array([1.0, 0.0, 0.0])) == pytest.approx(2.0), (
            "premise: applying the declaration as written scales the frame, so the fix is not cosmetic"
        )
        reported = _rotation_matrix(_link(path).orientation)
        assert np.linalg.det(reported) == pytest.approx(1.0, abs=1e-9)
        for axis in np.eye(3):
            assert np.linalg.norm(reported @ axis) == pytest.approx(1.0, abs=1e-9)

    def test_a_nested_links_quat_is_normalized_too(self, tmp_path):
        """The reported frame is parent-relative, and both levels are normalized."""
        path = str(tmp_path / "nested.xml")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                '<mujoco><worldbody><body name="link" pos="0.1 0 0" quat="1 -1 0 0">'
                '<joint name="j" type="hinge" axis="0 0 1"/><geom type="box" size="0.1 0.05 0.02"/>'
                '<body name="child" pos="0 0.2 0" quat="0 2 0 2">'
                '<joint name="k" type="hinge" axis="0 1 0"/><geom type="box" size="0.05 0.05 0.05"/>'
                "</body></body></worldbody></mujoco>"
            )
        for name in ("link", "child"):
            got = _link(path, name).orientation
            assert got == pytest.approx(_compiled_body_quat(path, name), abs=1e-12)
            assert _norm(got) == pytest.approx(1.0, abs=1e-12)


class TestASceneObjectsQuatIsNormalized:
    """The same helper feeds ``load_mjcf_scene_objects``, so both readers move together."""

    @pytest.mark.parametrize("declaration", NON_UNIT)
    def test_the_reported_object_orientation_is_the_one_mujoco_compiled(self, tmp_path, declaration):
        path = _write(tmp_path, body_attrs=f'quat="{declaration}"')
        assert _only(path).quat == pytest.approx(_compiled_body_quat(path, "obj"), abs=1e-12)

    def test_a_mesh_geoms_reported_frame_is_normalized(self, tmp_path):
        """The third reader of the same helper: a mesh geom's frame within its body.

        Graded transitively against the body reading of the same declaration --
        MuJoCo folds a mesh's principal-inertia alignment into ``geom_quat``, so
        the compiled value is not the frame the file declared and cannot serve
        as the oracle here.
        """
        got = _only(_write_mesh_scene(tmp_path, 'quat="1 -1 0 0"')).mesh_quat
        expected = _compiled_body_quat(_write(tmp_path, body_attrs='quat="1 -1 0 0"'), "obj")
        assert got == pytest.approx(expected, abs=1e-12)
        assert _norm(got) == pytest.approx(1.0, abs=1e-12)

    def test_both_readers_report_one_declaration_the_same_way(self, tmp_path):
        """One rotation, two public readers, one answer."""
        robot = _write_robot(tmp_path, body_attrs='quat="1 -1 0 0"')
        scene = _write(tmp_path, body_attrs='quat="1 -1 0 0"')
        assert _link(robot).orientation == pytest.approx(_only(scene).quat, abs=1e-12)


class TestAZeroQuatIsReadAsMalformed:
    """The one non-unit ``quat`` that cannot be normalized was reported verbatim.

    Four zeros are not a direction, so there is nothing to normalize onto and
    the reading is identity -- this loader's answer for every ``quat`` it cannot
    use. It is also the one orientation the library refuses on its write side, so
    reporting it was a reader handing out a value its own writers reject.
    """

    def test_the_reported_orientation_is_identity(self, tmp_path):
        path = _write_robot(tmp_path, body_attrs=f'quat="{ZERO}"')
        assert _link(path).orientation == pytest.approx(IDENTITY)

    def test_the_scene_reader_agrees(self, tmp_path):
        path = _write(tmp_path, body_attrs=f'quat="{ZERO}"')
        assert _only(path).quat == pytest.approx(IDENTITY)

    def test_the_verbatim_reading_is_refused_on_the_write_side(self):
        """The value the loaders used to report, offered to the library's own guard."""
        accepted, error = coerce_orientation_quaternion("add_object", "orientation", [0.0, 0.0, 0.0, 0.0])
        assert accepted is None
        assert error is not None and "zero norm" in error

    def test_mujoco_refuses_a_zero_quaternion(self, tmp_path):
        """No compiled model is affected: MuJoCo refuses such a file outright."""
        path = _write_robot(tmp_path, body_attrs=f'quat="{ZERO}"')
        with pytest.raises(ValueError, match="zero quaternion is not allowed"):
            mujoco.MjModel.from_xml_path(path)


class TestTheReadingIsNotWidened:
    """What normalizing must not disturb: unit input, the other four spellings, refusals."""

    @pytest.mark.parametrize("declaration", ALREADY_UNIT)
    def test_a_unit_quat_is_reported_unchanged(self, tmp_path, declaration):
        expected = tuple(float(part) for part in declaration.split())
        path = _write_robot(tmp_path, body_attrs=f'quat="{declaration}"')
        assert _link(path).orientation == pytest.approx(expected, abs=1e-15)

    @pytest.mark.parametrize("declaration", MALFORMED)
    def test_a_quat_no_reader_can_use_is_identity(self, tmp_path, declaration):
        path = _write_robot(tmp_path, body_attrs=f'quat="{declaration}"')
        assert _link(path).orientation == pytest.approx(IDENTITY), (
            "identity is this loader's historical reading for a malformed orientation"
        )

    @pytest.mark.parametrize("spelling", [s for s in DECLARATIONS if s != "quat"])
    def test_the_alternative_spellings_still_match_mujoco(self, tmp_path, spelling):
        """The four constructed spellings were already unit and are untouched."""
        path = _write_robot(tmp_path, body_attrs=f'{spelling}="{DECLARATIONS[spelling]}"')
        got = _link(path).orientation
        expected = _compiled_body_quat(path, "link")
        assert min(np.abs(np.array(got) - expected).max(), np.abs(np.array(got) + expected).max()) < 1e-9
        assert _norm(got) == pytest.approx(1.0, abs=1e-12)

    def test_the_other_reported_link_fields_are_unchanged(self, tmp_path):
        link = _link(_write_robot(tmp_path, body_attrs='quat="1 -1 0 0"'))
        assert link.position == pytest.approx((0.1, 0.2, 0.3))
        assert link.shape == "box"
        assert link.shape_size == pytest.approx((0.1, 0.05, 0.02))
