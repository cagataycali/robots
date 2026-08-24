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

import itertools
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


def _write_class_scene(tmp_path, class_attrs: str, geom_attrs: str, *, name: str) -> str:
    """A box geom taking ``class_attrs`` from a ``<default>`` and declaring ``geom_attrs``.

    A box's compiled ``geom_quat`` is the frame the file authored, so MuJoCo's
    own answer for a class/element pair can be read straight off it. A mesh
    geom's cannot: MuJoCo folds the mesh's principal-inertia alignment into it.
    """
    path = tmp_path / f"{name}.xml"
    path.write_text(
        f'<mujoco><compiler angle="degree"/>'
        f'<default><default class="rot">'
        f'<geom {class_attrs} type="box" size="0.1 0.05 0.02"/></default></default>'
        f'<worldbody><body name="obj" pos="0.1 0.2 0.3">'
        f'<geom name="g" class="rot" {geom_attrs}/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    return str(path)


def _write_nested_class_scene(tmp_path, parent: str, child: str, geom_attrs: str, *, name: str) -> str:
    """The same, with the class nested inside another that also declares an orientation."""
    path = tmp_path / f"{name}.xml"
    path.write_text(
        f'<mujoco><compiler angle="degree"/>'
        f'<default><default class="outer">'
        f'<geom {parent} type="box" size="0.1 0.05 0.02"/>'
        f'<default class="inner"><geom {child}/></default></default></default>'
        f'<worldbody><body name="obj" pos="0.1 0.2 0.3">'
        f'<geom name="g" class="inner" {geom_attrs}/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    return str(path)


def _write_mesh_class_scene(tmp_path, class_attrs: str, geom_attrs: str) -> str:
    """A mesh geom taking ``class_attrs`` from a ``<default>``: the reported frame."""
    (tmp_path / "shape.obj").write_text(_TETRA_OBJ, encoding="utf-8")
    path = tmp_path / "mesh_class_scene.xml"
    path.write_text(
        f'<mujoco><compiler angle="degree"/>'
        f'<asset><mesh name="shape" file="shape.obj"/></asset>'
        f'<default><default class="rot"><geom {class_attrs} type="mesh" mesh="shape"/></default></default>'
        f'<worldbody><body name="obj" pos="0.1 0.2 0.3">'
        f'<geom name="g" class="rot" {geom_attrs}/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    return str(path)


def _mujoco_geom_quat(path: str) -> np.ndarray:
    """The orientation MuJoCo's compiler stored for the fixture's one geom."""
    model = mujoco.MjModel.from_xml_path(path)
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g")
    assert geom >= 0, "premise: the fixture declares a geom named 'g'"
    return np.asarray(model.geom_quat[geom], dtype=float)


def _reported_mesh_quat(tmp_path, class_attrs: str, geom_attrs: str, *, name: str) -> tuple[float, ...]:
    """The frame ``load_mjcf_scene_objects`` reports for a mesh geom under ``class_attrs``.

    The public path, so this measures what a caller receives rather than the
    internals it composes.
    """
    (tmp_path / "shape.obj").write_text(_TETRA_OBJ, encoding="utf-8")
    path = tmp_path / f"mesh_{name}.xml"
    path.write_text(
        f'<mujoco><compiler angle="degree"/>'
        f'<asset><mesh name="shape" file="shape.obj"/></asset>'
        f'<default><default class="rot"><geom {class_attrs} type="mesh" mesh="shape"/></default></default>'
        f'<worldbody><body name="obj" pos="0.1 0.2 0.3">'
        f'<geom name="g" class="rot" {geom_attrs}/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    return _only(str(path)).mesh_quat


def _reported_nested_mesh_quat(tmp_path, parent: str, child: str, geom_attrs: str, *, name: str) -> tuple[float, ...]:
    """The same, with the geom's class nested inside another that also declares one."""
    (tmp_path / "shape.obj").write_text(_TETRA_OBJ, encoding="utf-8")
    path = tmp_path / f"mesh_{name}.xml"
    path.write_text(
        f'<mujoco><compiler angle="degree"/>'
        f'<asset><mesh name="shape" file="shape.obj"/></asset>'
        f'<default><default class="outer"><geom {parent} type="mesh" mesh="shape"/>'
        f'<default class="inner"><geom {child}/></default></default></default>'
        f'<worldbody><body name="obj" pos="0.1 0.2 0.3">'
        f'<geom name="g" class="inner" {geom_attrs}/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    return _only(str(path)).mesh_quat


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


class TestTheClassAndElementPrecedenceMatchesMujoco:
    """A ``<default>`` class and the element may spell the orientation differently.

    MuJoCo compiles such a model. It keeps the four non-``quat`` spellings in one
    slot separate from ``quat`` and resolves that slot first, so an inner
    declaration of one of them replaces an outer declaration of another, and any
    of them beats a ``quat`` from any level. Reading the merged attributes as a
    flat mapping instead loses both rules: it cannot say which alternative
    spelling is in effect, and it reads a legal pair as the mutually exclusive
    case MuJoCo refuses.

    Graded against ``geom_quat`` for a box, which MuJoCo stores as the file
    authored it; the mesh passthrough is graded end to end below.
    """

    @pytest.mark.parametrize("from_class", SPELLINGS)
    @pytest.mark.parametrize("on_element", SPELLINGS)
    def test_every_class_and_element_pair_matches_mujoco(self, tmp_path, from_class, on_element):
        path = _write_class_scene(
            tmp_path,
            f'{from_class}="{DECLARATIONS[from_class]}"',
            f'{on_element}="{DECLARATIONS[on_element]}"',
            name=f"{from_class}_{on_element}",
        )
        expected = _mujoco_geom_quat(path)
        got = _reported_mesh_quat(
            tmp_path,
            f'{from_class}="{DECLARATIONS[from_class]}"',
            f'{on_element}="{DECLARATIONS[on_element]}"',
            name=f"{from_class}_{on_element}",
        )
        assert _same_rotation(got, expected) < 1e-6, (
            f"class {from_class} + element {on_element}: the loader read "
            f"{tuple(round(v, 5) for v in got)} where MuJoCo compiled "
            f"{tuple(round(v, 5) for v in expected)}"
        )

    def test_every_nested_class_chain_matches_mujoco(self, tmp_path):
        """An inner class's spelling replaces its parent's, and the element's replaces both."""
        offenders = []
        graded = 0
        for outer, inner, element in itertools.product(SPELLINGS, ("", *SPELLINGS), ("", *SPELLINGS)):
            path = _write_nested_class_scene(
                tmp_path,
                f'{outer}="{DECLARATIONS[outer]}"',
                f'{inner}="{DECLARATIONS[inner]}"' if inner else "",
                f'{element}="{DECLARATIONS[element]}"' if element else "",
                name=f"n_{outer}_{inner or 'none'}_{element or 'none'}",
            )
            expected = _mujoco_geom_quat(path)
            graded += 1
            try:
                got = _reported_nested_mesh_quat(
                    tmp_path,
                    f'{outer}="{DECLARATIONS[outer]}"',
                    f'{inner}="{DECLARATIONS[inner]}"' if inner else "",
                    f'{element}="{DECLARATIONS[element]}"' if element else "",
                    name=f"n_{outer}_{inner or 'none'}_{element or 'none'}",
                )
            except ValueError as refused:
                offenders.append(f"outer={outer} inner={inner or '-'} element={element or '-'}: refused ({refused})")
                continue
            if _same_rotation(got, expected) >= 1e-6:
                offenders.append(
                    f"outer={outer} inner={inner or '-'} element={element or '-'}: "
                    f"loader {tuple(round(v, 5) for v in got)} vs mujoco {tuple(round(v, 5) for v in expected)}"
                )
        assert graded == len(SPELLINGS) * 6 * 6, f"premise: every chain compiles, graded {graded}"
        assert not offenders, f"{len(offenders)} of {graded} nested chains disagree with MuJoCo:\n" + "\n".join(
            offenders[:8]
        )

    def test_a_class_spelling_beats_the_elements_own_quat(self, tmp_path):
        """The reported mesh frame is the class's euler, not the element's quat."""
        reference = _write_class_scene(
            tmp_path, 'euler="90 0 0"', 'quat="0.7071068 0 0.7071068 0"', name="ref_class_euler"
        )
        expected = _mujoco_geom_quat(reference)
        assert _same_rotation(expected, (0.7071068, 0.7071068, 0.0, 0.0)) < 1e-6, (
            f"premise: MuJoCo resolves the class euler, not the element quat; it compiled {expected}"
        )
        got = _reported_mesh_quat(tmp_path, 'euler="90 0 0"', 'quat="0.7071068 0 0.7071068 0"', name="class_euler")
        assert _same_rotation(got, expected) < 1e-6, (
            f"a class euler must beat the element's own quat: reported {tuple(round(v, 5) for v in got)}, "
            f"MuJoCo compiled {tuple(round(v, 5) for v in expected)}"
        )

    def test_an_element_spelling_beats_the_classs_quat(self, tmp_path):
        """And the other way round: the element's euler, not the class's quat."""
        reference = _write_class_scene(
            tmp_path, 'quat="0.7071068 0 0.7071068 0"', 'euler="90 0 0"', name="ref_elem_euler"
        )
        expected = _mujoco_geom_quat(reference)
        assert _same_rotation(expected, (0.7071068, 0.7071068, 0.0, 0.0)) < 1e-6, (
            f"premise: MuJoCo resolves the element euler, not the class quat; it compiled {expected}"
        )
        got = _reported_mesh_quat(tmp_path, 'quat="0.7071068 0 0.7071068 0"', 'euler="90 0 0"', name="elem_euler")
        assert _same_rotation(got, expected) < 1e-6, (
            f"the element's euler must beat the class's quat: reported {tuple(round(v, 5) for v in got)}, "
            f"MuJoCo compiled {tuple(round(v, 5) for v in expected)}"
        )

    def test_a_class_and_element_pair_loads(self, tmp_path):
        """It is a model MuJoCo compiles, so the scene must load rather than be refused."""
        path = _write_mesh_class_scene(tmp_path, 'euler="90 0 0"', 'quat="0.7071068 0 0.7071068 0"')
        objects = load_mjcf_scene_objects(path)
        assert [o.name for o in objects] == ["obj"], (
            "a class supplying one spelling while the element declares another is not the "
            f"mutually exclusive case; the loader reported {[o.name for o in objects]}"
        )

    def test_only_an_elements_own_attributes_are_mutually_exclusive(self, tmp_path):
        """Two spellings across the class boundary are legal; on one element they are not."""
        across = _write_mesh_class_scene(tmp_path, 'euler="90 0 0"', 'zaxis="1 0 0"')
        assert len(load_mjcf_scene_objects(across)) == 1, "MuJoCo compiles a class/element pair"
        together = _write(tmp_path, geom_attrs='euler="90 0 0" zaxis="1 0 0"')
        with pytest.raises(Exception) as caught:  # noqa: PT011 - mujoco raises its own FatalError
            mujoco.MjModel.from_xml_path(together)
        assert "orientation" in str(caught.value).lower(), f"premise: MuJoCo refuses it; it said {caught.value}"


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
