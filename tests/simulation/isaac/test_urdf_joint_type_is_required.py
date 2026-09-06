"""A URDF joint that states no type is refused, not read as a fixed joint.

``type`` is a required attribute of a URDF ``<joint>``. An absent one is
therefore missing information, not a declaration of a default - which is the
distinction :mod:`strands_robots.simulation.isaac.loaders` already draws for
``<axis>``, an attribute URDF *does* give a default (``+X``), and for MJCF's
``type``, which MuJoCo documents as defaulting to ``hinge``.

Both of this package's URDF readers defaulted it to ``"fixed"``:

* :func:`~strands_robots.simulation.isaac.loaders.load_urdf` emitted a
  ``JointDef`` identical to the one a deliberate ``type="fixed"`` produces, so
  the robot came back with fewer actuated DOFs than the file names and the load
  reported success. That is the silent ``joint_count`` the loader's documented
  failure semantics exist to convert into a message.
* :func:`~strands_robots.simulation.isaac.joint_names.urdf_joint_names` dropped
  the joint from the movable set, so its name never entered the candidate pool
  :func:`~strands_robots.simulation.isaac.joint_names.demangle_usd_joint_names`
  translates a mangled DOF name through.

The empty spelling of the same omission was already refused by name in the
loader (``type=""`` -> "unknown joint type"), so one file was refused and the
other welded for the same missing declaration, decided only by whether the
attribute was written empty or left out. MuJoCo refuses both, and this module
already cites MuJoCo as its reference for the URDF ``<axis>`` default; the
cross-parser class here checks that against the real parser rather than
restating it.

Neither reader needs an Isaac Kit install or ``pxr``: both are stdlib XML
parsers.
"""

from __future__ import annotations

import pathlib

import pytest

from strands_robots.simulation.isaac import joint_names as jn
from strands_robots.simulation.isaac import loaders

# URDF's declared joint types, stated here rather than read from the code so
# these tests are an independent oracle rather than a restatement of it.
URDF_DECLARED_TYPES = ("revolute", "continuous", "prismatic", "fixed", "floating", "planar")

# The declared types the loader reads as moving - the DOFs a file loses when a
# joint is welded by a default instead of being refused.
URDF_MOVING_TYPES = ("revolute", "continuous", "prismatic")

#: The two ways a URDF ``<joint>`` can fail to state a type, and the substring
#: each refusal must carry. Both are one missing declaration; before the fix
#: only the second was refused.
UNSTATED_TYPE_SPELLINGS = {
    "omitted": ("", "type"),
    "empty": ('type=""', "type"),
}


def _arm_urdf(tmp_path: pathlib.Path, shoulder_attrs: str, name: str = "arm.urdf") -> str:
    """Write a three-link arm whose shoulder joint carries ``shoulder_attrs``.

    The elbow is always a plain ``revolute``, so every file here is a robot the
    readers can load apart from the one attribute under test - which keeps the
    refusals below attributable to that attribute and lets the accepted cases
    show what the shoulder contributes.
    """
    path = tmp_path / name
    path.write_text(
        '<robot name="r">'
        '<link name="base"/><link name="mid"/><link name="tip"/>'
        f'<joint name="shoulder" {shoulder_attrs}>'
        '<parent link="base"/><child link="mid"/><axis xyz="0 0 1"/>'
        "</joint>"
        '<joint name="elbow" type="revolute">'
        '<parent link="mid"/><child link="tip"/><axis xyz="0 0 1"/>'
        "</joint>"
        "</robot>"
    )
    return str(path)


def _moving_joint_names(robot: object) -> list[str]:
    """The names ``load_urdf`` reported as something other than a weld."""
    return [j.name for j in robot.joints if j.joint_type != "fixed"]  # type: ignore[attr-defined]


class TestBothReadersRefuseAJointThatStatesNoType:
    """The headline: neither reader answers for an attribute the file omits."""

    @pytest.mark.parametrize("spelling", sorted(UNSTATED_TYPE_SPELLINGS))
    def test_the_loader_refuses_it(self, tmp_path: pathlib.Path, spelling: str) -> None:
        attrs, expected = UNSTATED_TYPE_SPELLINGS[spelling]
        path = _arm_urdf(tmp_path, attrs)
        with pytest.raises(ValueError) as excinfo:
            loaders.load_urdf(path)
        message = str(excinfo.value)
        assert expected in message, message
        assert "shoulder" in message, f"the refusal does not name the joint: {message}"

    def test_the_name_reader_refuses_the_omitted_spelling(self, tmp_path: pathlib.Path) -> None:
        """``urdf_joint_names`` refuses the omission it used to skip.

        Only the omitted spelling: ``type=""`` is a STATED type this function
        has no named DOF for, which it skips like any other non-movable type -
        the loader is the reader that grades the vocabulary.
        """
        path = _arm_urdf(tmp_path, "")
        with pytest.raises(ValueError, match="type"):
            jn.urdf_joint_names(path)

    def test_the_refusal_is_raised_before_the_joint_becomes_a_weld(self, tmp_path: pathlib.Path) -> None:
        """No partial robot is returned alongside the refusal.

        The loader collects joints into a list and validates the kinematic tree
        at the end, so a refusal raised mid-loop must not have produced a
        ``ProceduralRobot`` at all - the caller gets an exception or a complete
        robot, never a robot missing a joint it asked about.
        """
        with pytest.raises(ValueError):
            loaders.load_urdf(_arm_urdf(tmp_path, ""))


class TestWhatTheDefaultUsedToCost:
    """The same file, with the shoulder type stated, keeps its DOF.

    These cells are the measurement the refusal replaces: an omitted ``type``
    used to produce exactly the ``fixed`` reading below, so the robot lost the
    shoulder DOF with the load reporting success.
    """

    @pytest.mark.parametrize("joint_type", URDF_MOVING_TYPES)
    def test_a_stated_moving_type_keeps_the_shoulder(self, tmp_path: pathlib.Path, joint_type: str) -> None:
        robot = loaders.load_urdf(_arm_urdf(tmp_path, f'type="{joint_type}"'))
        assert _moving_joint_names(robot) == ["shoulder", "elbow"]
        assert jn.urdf_joint_names(_arm_urdf(tmp_path, f'type="{joint_type}"')) == ["shoulder", "elbow"]

    def test_a_stated_fixed_type_welds_it_and_is_still_accepted(self, tmp_path: pathlib.Path) -> None:
        """The reading an omission used to be indistinguishable from.

        A deliberate ``type="fixed"`` is a legitimate request and stays
        accepted, which is what made the default silent: the two files produced
        the same answer, so no caller could tell a malformed URDF from a welded
        joint.
        """
        path = _arm_urdf(tmp_path, 'type="fixed"')
        robot = loaders.load_urdf(path)
        assert [(j.name, j.joint_type) for j in robot.joints] == [("shoulder", "fixed"), ("elbow", "revolute")]
        assert jn.urdf_joint_names(path) == ["elbow"]


class TestEveryDeclaredTypeStillLoads:
    """Unchanged behaviour: the refusal is scoped to an absent attribute.

    Every cell here held before the fix too. A type URDF declares is still
    accepted by the loader, and the two readers still agree about which of them
    move.
    """

    @pytest.mark.parametrize("joint_type", URDF_DECLARED_TYPES)
    def test_the_loader_accepts_it(self, tmp_path: pathlib.Path, joint_type: str) -> None:
        robot = loaders.load_urdf(_arm_urdf(tmp_path, f'type="{joint_type}"'))
        assert [j.name for j in robot.joints] == ["shoulder", "elbow"]

    @pytest.mark.parametrize("joint_type", URDF_DECLARED_TYPES)
    def test_the_two_readers_agree_about_whether_it_moves(self, tmp_path: pathlib.Path, joint_type: str) -> None:
        path = _arm_urdf(tmp_path, f'type="{joint_type}"')
        robot = loaders.load_urdf(path)
        assert ("shoulder" in _moving_joint_names(robot)) == ("shoulder" in jn.urdf_joint_names(path))

    def test_a_declared_type_outside_the_movable_set_is_skipped_not_refused(self, tmp_path: pathlib.Path) -> None:
        """``urdf_joint_names`` still answers for a stated non-movable type.

        The scoping sentence of the fix: an omitted attribute is a malformed
        file, a stated ``planar`` is a joint this function has no named DOF for.
        """
        assert jn.urdf_joint_names(_arm_urdf(tmp_path, 'type="planar"')) == ["elbow"]


class TestMjcfKeepsItsOwnDefault:
    """MJCF is deliberately unchanged, because MJCF declares the default.

    MuJoCo documents ``hinge`` as the type of a ``<joint>`` that states none, so
    an absent ``type`` there IS a declaration and reading it as a hinge is
    correct. This is the contrast that makes the URDF refusal a statement about
    the format rather than a blanket rule about missing attributes.
    """

    def test_an_mjcf_joint_with_no_type_is_a_hinge(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "arm.xml"
        path.write_text(
            '<mujoco model="r"><worldbody>'
            '<body name="base"><body name="mid">'
            '<joint name="shoulder" axis="0 0 1"/>'
            "</body></body>"
            "</worldbody></mujoco>"
        )
        robot = loaders.load_mjcf(str(path))
        assert [(j.name, j.joint_type) for j in robot.joints] == [("shoulder", "revolute")]

    def test_the_mjcf_default_is_the_one_the_format_declares(self) -> None:
        """Stated as a fact about MJCF, so the two answers stay traceable."""
        assert loaders._MJCF_JOINT_TYPE_MAP["hinge"] == "revolute"


class TestTheReferenceParserRefusesTheSameFile:
    """MuJoCo, cited by this loader for the ``<axis>`` default, agrees.

    An oracle rather than a restatement: the requiredness of ``type`` is a
    property of URDF, and the parser the module already defers to on the
    format's defaults is the cheapest independent witness to it. Skipped where
    ``mujoco`` is not installed - the readers under test never need it.
    """

    @staticmethod
    def _mujoco_urdf(tmp_path: pathlib.Path, shoulder_attrs: str) -> str:
        """A two-link URDF with the mass/limit data MuJoCo's compiler requires."""
        inertial = "<inertial><mass value='1'/><inertia ixx='1' iyy='1' izz='1' ixy='0' ixz='0' iyz='0'/></inertial>"
        path = tmp_path / "mj.urdf"
        path.write_text(
            '<robot name="r">'
            f'<link name="base">{inertial}</link><link name="tip">{inertial}</link>'
            f'<joint name="shoulder" {shoulder_attrs}>'
            '<parent link="base"/><child link="tip"/><axis xyz="0 0 1"/>'
            "<limit lower='-1' upper='1' effort='1' velocity='1'/>"
            "</joint>"
            "</robot>"
        )
        return str(path)

    def test_it_loads_a_stated_type(self, tmp_path: pathlib.Path) -> None:
        mujoco = pytest.importorskip("mujoco")
        model = mujoco.MjModel.from_xml_path(self._mujoco_urdf(tmp_path, 'type="revolute"'))
        assert model.njnt == 1

    @pytest.mark.parametrize("spelling", sorted(UNSTATED_TYPE_SPELLINGS))
    def test_it_refuses_a_joint_that_states_no_type(self, tmp_path: pathlib.Path, spelling: str) -> None:
        mujoco = pytest.importorskip("mujoco")
        attrs, _ = UNSTATED_TYPE_SPELLINGS[spelling]
        with pytest.raises(ValueError, match="type"):
            mujoco.MjModel.from_xml_path(self._mujoco_urdf(tmp_path, attrs))
