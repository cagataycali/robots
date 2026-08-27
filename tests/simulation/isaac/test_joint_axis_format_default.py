"""Each description format's own default joint axis, graded against MuJoCo.

``_parse_axis`` is the shared 3-vector parser behind both XML loaders in
:mod:`strands_robots.simulation.isaac.loaders`, and it carried a single default
of ``(0, 0, 1)`` in its own signature. That is MJCF's default: an MJCF
``<joint>`` that omits ``axis`` acts about +Z. URDF's default is +X, and
``<axis>`` is optional there, so ``load_urdf`` -- which called the parser
without naming a default -- reported a URDF joint that omits ``<axis>`` as
acting about +Z.

Both are valid axes, so nothing downstream could tell a joint the file declares
in one plane from one whose axis was replaced by the other format's: the load
reported success either way, and ``JointDef.axis`` is the whole product of the
read.

Every expectation here is derived from ``mujoco.MjModel``: MuJoCo parses URDF as
well as MJCF, so ``jnt_axis`` answers the same question for both formats and no
expected axis is restated by hand. The MJCF cases are graded by the same oracle
in the same file, which is what makes "MJCF's default was right and the URDF
call site was borrowing it" a measurement rather than a claim.

The parser's *tolerance* is deliberately unchanged: an ``<axis>`` stating a
vector it cannot read still degrades to the format's default rather than
raising, even though MuJoCo refuses such a model outright. That boundary is
pinned by :class:`TestTheToleranceBoundaryIsUnchanged` so it is moved
explicitly if it is ever moved at all; only *which* default it lands on
changed, and that is graded by
:class:`TestAnUnreadableUrdfAxisIsTheUrdfDefault`.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import Any

import pytest

from strands_robots.simulation.isaac import loaders
from strands_robots.simulation.isaac.loaders import (
    _MJCF_DEFAULT_JOINT_AXIS,
    _URDF_DEFAULT_JOINT_AXIS,
    load_mjcf,
    load_urdf,
)

from .test_urdf_link_pose import INERTIAL

mujoco = pytest.importorskip("mujoco")

#: The URDF joint types that carry a 1-DOF axis, so an axis default is
#: observable on each. ``fixed`` has no axis to report.
MOVABLE_URDF_TYPES = ("revolute", "continuous", "prismatic")

#: The MJCF joint types that carry a 1-DOF axis.
MOVABLE_MJCF_TYPES = ("hinge", "slide")

#: ``<axis>`` spellings that state no vector this parser can read: absent
#: entirely, present with no ``xyz``, empty, wrong arity, and non-numeric. Each
#: must land on the reading format's default.
UNREADABLE_AXES = (
    "",
    "<axis/>",
    '<axis xyz=""/>',
    '<axis xyz="1 0"/>',
    '<axis xyz="p q r"/>',
)


def _write_urdf(tmp_path, *, jtype: str, axis_fragment: str, name: str = "axis") -> str:
    """A two-link URDF whose single joint carries ``axis_fragment`` verbatim."""
    limit = '<limit lower="-1" upper="1" effort="1" velocity="1"/>' if jtype in ("revolute", "prismatic") else ""
    path = tmp_path / f"{name}.urdf"
    path.write_text(
        f'<robot name="{name}">'
        f'<link name="base">{INERTIAL}</link><link name="arm">{INERTIAL}</link>'
        f'<joint name="j" type="{jtype}">'
        f'<parent link="base"/><child link="arm"/>'
        f'<origin xyz="0 0 0.2" rpy="0 0 0"/>{axis_fragment}{limit}'
        f"</joint></robot>",
        encoding="utf-8",
    )
    return str(path)


def _write_mjcf(tmp_path, *, jtype: str, axis_attr: str = "", name: str = "axis") -> str:
    """A one-body MJCF whose single joint carries ``axis_attr`` verbatim."""
    path = tmp_path / f"{name}.xml"
    path.write_text(
        f'<mujoco model="{name}"><worldbody>'
        f'<body name="arm" pos="0 0 0.2"><geom type="box" size="0.1 0.02 0.02"/>'
        f'<joint name="j" type="{jtype}"{axis_attr}/></body>'
        f"</worldbody></mujoco>",
        encoding="utf-8",
    )
    return str(path)


def _compiler_axis(path: str) -> tuple[float, float, float]:
    """The joint's axis as MuJoCo compiled the file -- the oracle.

    Looked up by name so a model that compiled the joint under a different name
    fails loudly instead of silently grading joint 0 of something else.
    """
    model = mujoco.MjModel.from_xml_path(path)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "j")
    assert jid >= 0, f"{path} compiled without a joint named 'j'; the oracle would be degenerate"
    axis = model.jnt_axis[jid]
    return (float(axis[0]), float(axis[1]), float(axis[2]))


def _only_joint_axis(path: str, *, mjcf: bool = False) -> tuple[float, float, float]:
    """The single joint's reported axis, asserting the read produced exactly one."""
    robot = load_mjcf(path) if mjcf else load_urdf(path)
    assert len(robot.joints) == 1, f"fixture {path} produced {len(robot.joints)} joints, expected 1"
    return robot.joints[0].axis


class TestTheTwoFormatsDefaultsDiffer:
    """The premise: nothing below can distinguish anything if they agree."""

    def test_the_declared_defaults_are_different_vectors(self) -> None:
        assert _URDF_DEFAULT_JOINT_AXIS != _MJCF_DEFAULT_JOINT_AXIS, (
            "premise: this file grades one format's default against the other's, "
            "which measures nothing unless they differ"
        )

    @pytest.mark.parametrize("jtype", MOVABLE_URDF_TYPES)
    def test_the_compiler_reads_a_urdf_default_as_the_other_formats_is_not(self, tmp_path, jtype) -> None:
        """The oracle itself distinguishes them, so the fixtures are not vacuous."""
        truth = _compiler_axis(_write_urdf(tmp_path, jtype=jtype, axis_fragment=""))
        assert truth == pytest.approx(_URDF_DEFAULT_JOINT_AXIS)
        assert truth != pytest.approx(_MJCF_DEFAULT_JOINT_AXIS)


class TestAnAbsentUrdfAxisIsTheUrdfDefault:
    """``<axis>`` is optional in URDF; omitting it declares +X, not +Z."""

    @pytest.mark.parametrize("jtype", MOVABLE_URDF_TYPES)
    def test_the_reported_axis_matches_the_compiler(self, tmp_path, jtype) -> None:
        path = _write_urdf(tmp_path, jtype=jtype, axis_fragment="")
        assert _only_joint_axis(path) == pytest.approx(_compiler_axis(path))

    @pytest.mark.parametrize("jtype", MOVABLE_URDF_TYPES)
    def test_the_reported_axis_is_not_the_mjcf_default(self, tmp_path, jtype) -> None:
        """Stated separately: the failure this pins is one specific wrong vector."""
        path = _write_urdf(tmp_path, jtype=jtype, axis_fragment="")
        assert _only_joint_axis(path) != pytest.approx(_MJCF_DEFAULT_JOINT_AXIS)


class TestADeclaredUrdfAxisIsUnchanged:
    """Control: a joint that states its axis is still read from the element."""

    @pytest.mark.parametrize(
        "xyz",
        ("1 0 0", "0 1 0", "0 0 1", "0 0 -1", "0.36 -0.48 0.8"),
    )
    def test_a_declared_axis_matches_the_compiler(self, tmp_path, xyz) -> None:
        path = _write_urdf(tmp_path, jtype="revolute", axis_fragment=f'<axis xyz="{xyz}"/>')
        assert _only_joint_axis(path) == pytest.approx(_compiler_axis(path), abs=1e-9)

    def test_a_declared_z_axis_is_not_read_as_the_urdf_default(self, tmp_path) -> None:
        """A joint that does declare +Z keeps +Z: the default never overrides it."""
        path = _write_urdf(tmp_path, jtype="revolute", axis_fragment='<axis xyz="0 0 1"/>')
        assert _only_joint_axis(path) == pytest.approx(_MJCF_DEFAULT_JOINT_AXIS)


class TestTheMjcfDefaultIsUnchanged:
    """The scope boundary: MJCF's default was already the compiler's answer.

    Graded by the same oracle as the URDF cases, which is what establishes that
    the fix belongs at the URDF call site rather than in the shared parser.
    """

    @pytest.mark.parametrize("jtype", MOVABLE_MJCF_TYPES)
    def test_an_absent_mjcf_axis_matches_the_compiler(self, tmp_path, jtype) -> None:
        path = _write_mjcf(tmp_path, jtype=jtype)
        assert _only_joint_axis(path, mjcf=True) == pytest.approx(_compiler_axis(path))

    @pytest.mark.parametrize("jtype", MOVABLE_MJCF_TYPES)
    def test_an_absent_mjcf_axis_is_not_read_as_the_urdf_default(self, tmp_path, jtype) -> None:
        path = _write_mjcf(tmp_path, jtype=jtype)
        assert _only_joint_axis(path, mjcf=True) != pytest.approx(_URDF_DEFAULT_JOINT_AXIS)

    def test_a_declared_mjcf_axis_matches_the_compiler(self, tmp_path) -> None:
        path = _write_mjcf(tmp_path, jtype="hinge", axis_attr=' axis="0 1 0"')
        assert _only_joint_axis(path, mjcf=True) == pytest.approx(_compiler_axis(path))


class TestAnUnreadableUrdfAxisIsTheUrdfDefault:
    """An axis this parser cannot read lands on URDF's default, not MJCF's.

    The parser reaches its default for an absent element and for a malformed
    one alike, so both spellings carried the borrowed +Z.
    """

    @pytest.mark.parametrize("fragment", UNREADABLE_AXES)
    def test_an_unreadable_urdf_axis_reads_as_the_urdf_default(self, tmp_path, fragment) -> None:
        path = _write_urdf(tmp_path, jtype="revolute", axis_fragment=fragment)
        assert _only_joint_axis(path) == pytest.approx(_URDF_DEFAULT_JOINT_AXIS)


class TestTheToleranceBoundaryIsUnchanged:
    """A malformed axis still degrades rather than raising -- unchanged here.

    MuJoCo refuses every malformed fixture below, so tolerating one is this
    package's own choice rather than something the format allows. *Which*
    default the tolerance lands on is what this change moves; that the
    tolerance exists at all is deliberately left alone, and pinning it here
    means moving it later is an explicit decision.
    """

    @pytest.mark.parametrize("fragment", UNREADABLE_AXES)
    def test_an_unreadable_urdf_axis_is_tolerated_rather_than_refused(self, tmp_path, fragment) -> None:
        """The load succeeds and still reports the joint, whatever the axis."""
        path = _write_urdf(tmp_path, jtype="revolute", axis_fragment=fragment)
        robot = load_urdf(path)
        assert [j.name for j in robot.joints] == ["j"]

    @pytest.mark.parametrize("fragment", UNREADABLE_AXES)
    def test_the_compiler_refuses_every_unreadable_fixture(self, tmp_path, fragment) -> None:
        """Premise: except for the absent one, which is the format's default.

        An absent ``<axis>`` is a declaration; the other four are malformed, and
        the compiler's refusal of them is what makes tolerating them a decision.
        """
        path = _write_urdf(tmp_path, jtype="revolute", axis_fragment=fragment)
        if not fragment:
            assert _compiler_axis(path) == pytest.approx(_URDF_DEFAULT_JOINT_AXIS)
            return
        with pytest.raises(ValueError):
            mujoco.MjModel.from_xml_path(path)

    def test_an_unreadable_mjcf_axis_reads_as_the_mjcf_default(self, tmp_path) -> None:
        path = _write_mjcf(tmp_path, jtype="hinge", axis_attr=' axis="0 1"')
        assert _only_joint_axis(path, mjcf=True) == pytest.approx(_MJCF_DEFAULT_JOINT_AXIS)


class TestNoCallSiteCanInheritAFormatsDefault:
    """The parser names no default of its own, so omitting one is not possible.

    A signature default is what let one format's axis reach the other's reader.
    Requiring the argument means a call site added later cannot repeat that by
    saying nothing -- the omission is a ``TypeError``, not a wrong axis.
    """

    def test_parse_axis_declares_no_default_of_its_own(self) -> None:
        parameter = inspect.signature(loaders._parse_axis).parameters["default"]
        assert parameter.default is inspect.Parameter.empty, (
            "a signature default would let a call site inherit one format's axis"
        )

    def test_omitting_the_default_is_refused(self) -> None:
        """The refusal names the argument that was not passed.

        The argument list is built rather than spelled at the call. A literal
        one-argument call *is* a statically wrong call -- that is the property
        under test -- so writing one here would make this assertion's own
        subject a finding for any caller-arity checker reading the file. Built
        this way the refusal stays a runtime fact about the signature, and the
        match pins it to the missing ``default`` rather than to any
        ``TypeError`` the parser body might raise for an unreadable axis.
        """
        one_argument_short: list[Any] = ["not a vector"]
        with pytest.raises(TypeError, match="missing 1 required positional argument: 'default'"):
            loaders._parse_axis(*one_argument_short)

    def test_every_call_site_names_its_default(self) -> None:
        """Derived, so a call site added later is graded on arrival."""
        source = pathlib.Path(loaders.__file__).read_text(encoding="utf-8")
        calls = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_parse_axis"
        ]
        assert calls, "no _parse_axis call sites found; this rule would grade nothing"
        unnamed = [
            node.lineno for node in calls if len(node.args) < 2 and not any(kw.arg == "default" for kw in node.keywords)
        ]
        assert not unnamed, f"_parse_axis called without naming a default at line(s) {unnamed}"
