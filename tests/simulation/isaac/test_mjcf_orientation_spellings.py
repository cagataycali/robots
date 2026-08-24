"""MJCF's five orientation spellings, graded against MuJoCo's own compiler.

MJCF gives a body or geom five mutually exclusive ways to state one rotation --
``quat``, ``euler``, ``axisangle``, ``xyaxes`` and ``zaxis`` -- and the scene
loader read only ``quat``, so an object a model rotates by any of the other four
was placed unrotated with the load reporting success. That is the shape of the
``<default>`` and ``<compiler meshdir>`` gaps before them: a format feature the
reader never asked about, where the fallback is a plausible value rather than a
refusal.

Every expectation here is derived from ``mujoco.MjModel``: the fixture is
compiled and the loader is compared against the ``body_quat`` the compiler
stored, so no expected quaternion is restated by hand. A non-mesh geom is used
for the geom cases on purpose -- MuJoCo bakes a mesh's principal-inertia
alignment into ``geom_quat``, so a mesh geom's compiled value is not the
authored frame and cannot serve as an oracle for what the file declared.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from strands_robots.simulation.isaac.loaders import load_mjcf_scene_objects

mujoco = pytest.importorskip("mujoco")

#: The five spellings, in the order the loader reports a multiply-declared element.
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


#: A tetrahedron: MuJoCo refuses a mesh with fewer than four vertices.
_TETRA_OBJ = "v 0 0 0\nv 0.4 0 0\nv 0 0.3 0\nv 0 0 0.2\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"


def _write(tmp_path, body_attrs: str = "", geom_attrs: str = "", compiler: str = "") -> str:
    """A one-body scene with a box geom, compiled by both MuJoCo and the loader."""
    path = tmp_path / "scene.xml"
    path.write_text(
        f"<mujoco>{compiler}<worldbody>"
        f'<body name="obj" pos="0.1 0.2 0.3" {body_attrs}>'
        f'<geom name="g" type="box" size="0.1 0.05 0.02" {geom_attrs}/>'
        f"</body></worldbody></mujoco>",
        encoding="utf-8",
    )
    return str(path)


def _write_mesh_scene(tmp_path, geom_attrs: str) -> str:
    """The same scene with a mesh geom, the only kind whose frame is reported."""
    (tmp_path / "shape.obj").write_text(_TETRA_OBJ, encoding="utf-8")
    path = tmp_path / "mesh_scene.xml"
    path.write_text(
        '<mujoco><asset><mesh name="shape" file="shape.obj"/></asset><worldbody>'
        '<body name="obj" pos="0.1 0.2 0.3">'
        f'<geom name="g" type="mesh" mesh="shape" {geom_attrs}/>'
        "</body></worldbody></mujoco>",
        encoding="utf-8",
    )
    return str(path)


def _mujoco_body_quat(path: str) -> np.ndarray:
    """The orientation MuJoCo's compiler stored for the fixture's one body."""
    model = mujoco.MjModel.from_xml_path(path)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj")
    assert body > 0, "premise: the fixture declares a body named 'obj'"
    return np.asarray(model.body_quat[body], dtype=float)


def _same_rotation(got, expected) -> float:
    """Angular agreement of two wxyz quaternions, treating q and -q as one rotation."""
    a = np.asarray(got, dtype=float)
    b = np.asarray(expected, dtype=float)
    return float(min(np.abs(a - b).max(), np.abs(a + b).max()))


def _only(path: str):
    objects = load_mjcf_scene_objects(path)
    assert len(objects) == 1, f"premise: the fixture declares one object, got {[o.name for o in objects]}"
    return objects[0]


class TestEverySpellingIsReadAsTheRotationMujocoCompiles:
    """The regression: four of the five spellings were reported as identity."""

    @pytest.mark.parametrize("spelling", SPELLINGS)
    def test_a_body_orientation_matches_the_compiler(self, tmp_path, spelling):
        path = _write(tmp_path, body_attrs=f'{spelling}="{DECLARATIONS[spelling]}"')
        expected = _mujoco_body_quat(path)
        assert _same_rotation(expected, (1.0, 0.0, 0.0, 0.0)) > 0.1, (
            "premise: the declaration is far from identity, so a reader that fell "
            "back to identity cannot pass by coincidence"
        )
        got = _only(path).quat
        delta = _same_rotation(got, expected)
        assert delta < 1e-6, (
            f'body {spelling}="{DECLARATIONS[spelling]}" was reported as '
            f"{tuple(round(v, 6) for v in got)}, but MuJoCo compiles that model with "
            f"{tuple(round(float(v), 6) for v in expected)} (|delta| {delta:.3e})"
        )

    @pytest.mark.parametrize("spelling", SPELLINGS)
    def test_a_mesh_geom_reads_the_same_declaration_as_a_body(self, tmp_path, spelling):
        """One declaration, one rotation, wherever MJCF permits it.

        MuJoCo folds a mesh's principal-inertia alignment into ``geom_quat``, so
        the compiled value is not the frame the file declared and cannot be the
        oracle here. The body reading is graded against the compiler above, so
        grading the geom against it carries that guarantee across.
        """
        declaration = f'{spelling}="{DECLARATIONS[spelling]}"'
        body_path = _write(tmp_path, body_attrs=declaration)
        expected = _mujoco_body_quat(body_path)
        assert _same_rotation(_only(body_path).quat, expected) < 1e-6, (
            "premise: the body reading of this declaration matches the compiler"
        )
        got = _only(_write_mesh_scene(tmp_path, declaration)).mesh_quat
        delta = _same_rotation(got, expected)
        assert delta < 1e-6, (
            f"geom {declaration} was reported as {tuple(round(v, 6) for v in got)}, but the "
            f"same declaration on a body is {tuple(round(float(v), 6) for v in expected)} "
            f"(|delta| {delta:.3e})"
        )


class TestTheAngleUnitsComeFromTheSplicedModel:
    """``<compiler angle>`` is model-global, so an ``<include>`` supplies it."""

    def test_radians_are_honoured(self, tmp_path):
        path = _write(tmp_path, body_attrs='euler="0 0 -1.5708"', compiler='<compiler angle="radian"/>')
        expected = _mujoco_body_quat(path)
        got = _only(path).quat
        assert _same_rotation(got, expected) < 1e-6, (
            f"euler='0 0 -1.5708' under angle='radian' was reported as "
            f"{tuple(round(v, 6) for v in got)}; MuJoCo compiles "
            f"{tuple(round(float(v), 6) for v in expected)}. Read as degrees it would be "
            "a rotation of about one degree instead of a quarter turn."
        )

    def test_radians_declared_in_an_include_are_honoured(self, tmp_path):
        (tmp_path / "units.xml").write_text('<mujoco><compiler angle="radian"/></mujoco>', encoding="utf-8")
        path = tmp_path / "scene.xml"
        path.write_text(
            '<mujoco><include file="units.xml"/><worldbody>'
            '<body name="obj" pos="0 0 0" euler="0 0 -1.5708">'
            '<geom name="g" type="box" size="0.1 0.05 0.02"/>'
            "</body></worldbody></mujoco>",
            encoding="utf-8",
        )
        expected = _mujoco_body_quat(str(path))
        got = _only(str(path)).quat
        assert _same_rotation(got, expected) < 1e-6, (
            "angle='radian' declared in an <include> was not honoured: reported "
            f"{tuple(round(v, 6) for v in got)}, MuJoCo compiles "
            f"{tuple(round(float(v), 6) for v in expected)}"
        )

    def test_the_last_compiler_element_wins(self, tmp_path):
        path = _write(
            tmp_path,
            body_attrs='euler="0 0 90"',
            compiler='<compiler angle="radian"/><compiler angle="degree"/>',
        )
        expected = _mujoco_body_quat(path)
        got = _only(path).quat
        assert _same_rotation(got, expected) < 1e-6, (
            "the last <compiler angle> in document order must win, as it does for meshdir"
        )

    def test_the_degree_default_reads_ninety_as_a_quarter_turn(self, tmp_path):
        """MJCF's default is degrees, so a bare model reads 90 as a quarter turn."""
        path = _write(tmp_path, body_attrs='euler="0 0 90"')
        expected = _mujoco_body_quat(path)
        assert _same_rotation(_only(path).quat, expected) < 1e-6
        assert _same_rotation(expected, (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))) < 1e-6

    @pytest.mark.parametrize("sequence", ["xyz", "zyx", "yxz", "XYZ", "ZYX", "YXZ"])
    def test_the_euler_sequence_and_its_case_are_honoured(self, tmp_path, sequence):
        path = _write(
            tmp_path,
            body_attrs='euler="0.3 -0.7 1.1"',
            compiler=f'<compiler angle="radian" eulerseq="{sequence}"/>',
        )
        expected = _mujoco_body_quat(path)
        got = _only(path).quat
        delta = _same_rotation(got, expected)
        assert delta < 1e-6, (
            f"eulerseq='{sequence}' composes the three rotations in an order this "
            f"reader did not follow: reported {tuple(round(v, 6) for v in got)}, "
            f"MuJoCo compiles {tuple(round(float(v), 6) for v in expected)} (|delta| {delta:.3e}). "
            "A lower-case sequence rotates about the fixed axes and an upper-case one "
            "about the moving axes, which is the same rotations composed in reverse."
        )


class TestMultipleSpellingsAreRefused:
    """MuJoCo refuses such a model, so there is no rotation to pick."""

    @pytest.mark.parametrize(
        "pair", [("euler", "quat"), ("euler", "axisangle"), ("xyaxes", "zaxis"), ("quat", "zaxis")]
    )
    def test_mujoco_refuses_two_orientation_specifiers(self, tmp_path, pair):
        first, second = pair
        path = _write(
            tmp_path,
            body_attrs=f'{first}="{DECLARATIONS[first]}" {second}="{DECLARATIONS[second]}"',
        )
        with pytest.raises(Exception) as caught:  # noqa: PT011 - mujoco raises its own FatalError
            mujoco.MjModel.from_xml_path(path)
        assert "orientation" in str(caught.value).lower(), (
            f"premise: MuJoCo refuses two orientation specifiers; it said {caught.value}"
        )

    @pytest.mark.parametrize("pair", [("euler", "quat"), ("xyaxes", "zaxis")])
    def test_the_loader_refuses_them_too(self, tmp_path, pair):
        first, second = pair
        path = _write(
            tmp_path,
            body_attrs=f'{first}="{DECLARATIONS[first]}" {second}="{DECLARATIONS[second]}"',
        )
        try:
            objects = load_mjcf_scene_objects(path)
        except ValueError as refused:
            message = str(refused)
            assert first in message and second in message, (
                f"the refusal must name the spellings that collided; it said {message}"
            )
            assert "mutually exclusive" in message, f"the refusal must say why; it said {message}"
            return
        raise AssertionError(
            f"a body declaring both {first} and {second} was accepted and reported "
            f"{[tuple(round(v, 4) for v in o.quat) for o in objects]}, but MuJoCo "
            "refuses that model outright, so the reported rotation is a guess"
        )


class TestTheReadingIsNotWidened:
    """Behaviour that must not move: these pass before and after the fix."""

    def test_a_quat_is_reported_as_written(self, tmp_path):
        path = _write(tmp_path, body_attrs='quat="0.5 0.5 0.5 0.5"')
        assert _only(path).quat == pytest.approx((0.5, 0.5, 0.5, 0.5))

    def test_no_orientation_is_identity(self, tmp_path):
        path = _write(tmp_path)
        got = _only(path)
        assert got.quat == pytest.approx((1.0, 0.0, 0.0, 0.0))
        assert got.mesh_quat == pytest.approx((1.0, 0.0, 0.0, 0.0))

    @pytest.mark.parametrize("spelling", SPELLINGS)
    def test_a_malformed_declaration_is_identity(self, tmp_path, spelling):
        path = _write(tmp_path, body_attrs=f'{spelling}="not numbers at all"')
        assert _only(path).quat == pytest.approx((1.0, 0.0, 0.0, 0.0)), (
            "a value that cannot be read stays identity, the historical reading for a malformed quat"
        )

    def test_a_zero_length_axis_is_identity(self, tmp_path):
        path = _write(tmp_path, body_attrs='zaxis="0 0 0"')
        assert _only(path).quat == pytest.approx((1.0, 0.0, 0.0, 0.0))
