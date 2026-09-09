"""The Isaac backend's two URDF readers agree about what a joint type is.

Two functions in this package read a joint's ``type`` attribute out of a URDF
and reach opposite conclusions about the same file:

* :func:`~strands_robots.simulation.isaac.joint_names.urdf_joint_names` keeps
  the joints whose type produces a named articulation DOF, and hands that list
  to :func:`~strands_robots.simulation.isaac.joint_names.demangle_usd_joint_names`
  as the pool of URDF names a mangled DOF name may decode to.
* :func:`~strands_robots.simulation.isaac.loaders.load_urdf` maps the type onto
  a ``JointDef`` kind, and *refuses* a type it does not recognise - naming the
  set it accepts in the refusal, which makes it this package's record of what
  URDF declares.

The movable set used to carry ``spherical``, which URDF does not declare. So
``urdf_joint_names`` reported such a joint as movable while ``load_urdf``,
handed the very same file, refused it as an unknown joint type - and the name
of a joint no URDF can declare entered the candidate pool, where a spurious
candidate can only make a decode ambiguous. An ambiguous decode is the one
outcome ``demangle_usd_joint_names`` declines to make, so it keeps the mangled
USD name: exactly the #1900 leak the module exists to close.

Nothing graded it. ``test_urdf_joint_name_demangle.py`` builds its fixtures
from ``revolute``, ``continuous``, ``prismatic`` and ``fixed`` only - the four
types an author writing a URDF would reach for - so the vocabulary's fourth
member, and both of URDF's excluded real types, were never exercised.

These tests need no Isaac Kit install and no ``pxr``: both readers are stdlib
XML parsers.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from strands_robots.simulation.isaac import joint_names as jn
from strands_robots.simulation.isaac import loaders

# URDF's declared joint types, stated here rather than read from the code, so
# these tests are an independent oracle for the vocabulary rather than a
# restatement of it. See the URDF joint specification.
URDF_DECLARED_TYPES = ("revolute", "continuous", "prismatic", "fixed", "floating", "planar")

# The three that carry a single degree of freedom, and so can be named by an
# articulation. Also stated locally, for the same reason.
URDF_SINGLE_DOF_TYPES = ("revolute", "continuous", "prismatic")


def _one_joint_urdf(tmp_path: pathlib.Path, joint_type: str) -> str:
    """Write a two-link URDF whose single joint carries ``joint_type``."""
    path = tmp_path / f"{joint_type}.urdf"
    path.write_text(
        '<robot name="r">'
        '<link name="base"/><link name="tip"/>'
        f'<joint name="j1" type="{joint_type}">'
        '<parent link="base"/><child link="tip"/>'
        "</joint>"
        "</robot>"
    )
    return str(path)


def _loader_reads_as_moving(joint_type: str) -> bool:
    """Whether ``load_urdf`` maps ``joint_type`` onto a moving ``JointDef``.

    Indexed rather than defaulted: the only caller iterates the map's own keys,
    and a ``"fixed"`` fallback here was a third copy of the one the loader used
    to apply to an absent ``type`` attribute - now refused, so a fallback
    standing in for a missing declaration should not survive in the tests that
    grade the vocabulary either.
    """
    return loaders._URDF_JOINT_TYPE_MAP[joint_type] != "fixed"


class TestTheLoaderRecordsTheFormat:
    """Premise: the loader's accepted set is this package's record of URDF.

    Both cells would hold before the fix too - they establish that the type
    map is a faithful record of the format, which is what makes it usable as
    the authority the movable set is checked against.
    """

    def test_the_type_map_keys_are_exactly_the_declared_types(self) -> None:
        assert set(loaders._URDF_JOINT_TYPE_MAP) == set(URDF_DECLARED_TYPES)

    def test_the_refusal_names_the_accepted_set(self, tmp_path: pathlib.Path) -> None:
        path = _one_joint_urdf(tmp_path, "screw")  # not a URDF joint type
        with pytest.raises(ValueError, match="unknown joint type") as excinfo:
            loaders.load_urdf(path)
        message = str(excinfo.value)
        for declared in URDF_DECLARED_TYPES:
            assert declared in message, f"the refusal does not name {declared!r}: {message}"


class TestTheMovableSetIsDrawnFromTheFormat:
    """The movable set sits between the loader's two answers.

    ``{types the loader reads as moving} <= movable <= {types the loader
    recognises}``. The upper bound refuses a type URDF does not declare; the
    lower bound refuses dropping one that does move. The bounds are deliberately
    not collapsed into an equality: were the importer found to surface a named
    DOF for a type the loader reads as ``fixed``, that belongs in the movable
    set without the loader's own mapping changing.
    """

    def test_every_movable_type_is_a_type_urdf_declares(self) -> None:
        invented = jn._MOVABLE_URDF_JOINT_TYPES - set(URDF_DECLARED_TYPES)
        assert invented == set(), f"not URDF joint types: {sorted(invented)}"

    def test_every_movable_type_is_one_the_loader_recognises(self) -> None:
        unknown = jn._MOVABLE_URDF_JOINT_TYPES - set(loaders._URDF_JOINT_TYPE_MAP)
        assert unknown == set(), f"movable here and refused by load_urdf as an unknown joint type: {sorted(unknown)}"

    def test_no_type_the_loader_reads_as_moving_is_left_out(self) -> None:
        moving = {t for t in loaders._URDF_JOINT_TYPE_MAP if _loader_reads_as_moving(t)}
        assert moving <= jn._MOVABLE_URDF_JOINT_TYPES, (
            f"the loader reads these as moving but they are not movable here: "
            f"{sorted(moving - jn._MOVABLE_URDF_JOINT_TYPES)}"
        )

    def test_the_set_is_the_single_dof_types(self) -> None:
        assert jn._MOVABLE_URDF_JOINT_TYPES == set(URDF_SINGLE_DOF_TYPES)

    def test_the_public_docstring_names_the_set_it_returns(self) -> None:
        """The documented accepted set is the one the code applies.

        ``urdf_joint_names`` is exported, and its docstring is where a caller
        reads which joint types it keeps. Deriving the expectation from the
        constant means the prose cannot drift away from the behaviour.
        """
        doc = " ".join((jn.urdf_joint_names.__doc__ or "").split())
        match = re.search(r"are returned \(([^)]*)\)", doc)
        assert match is not None, f"no 'are returned (...)' clause in the docstring: {doc!r}"
        documented = set(re.findall(r"``([a-z]+)``", match.group(1)))
        assert documented == jn._MOVABLE_URDF_JOINT_TYPES, (
            f"docstring says {sorted(documented)}, code keeps {sorted(jn._MOVABLE_URDF_JOINT_TYPES)}"
        )


class TestTheTwoReadersAgreeOnEveryDeclaredType:
    """For each type URDF declares, both readers reach the same verdict."""

    @pytest.mark.parametrize("joint_type", URDF_DECLARED_TYPES)
    def test_reported_as_movable_exactly_when_the_loader_reads_it_as_moving(
        self, tmp_path: pathlib.Path, joint_type: str
    ) -> None:
        path = _one_joint_urdf(tmp_path, joint_type)
        reported = jn.urdf_joint_names(path)
        robot = loaders.load_urdf(path)
        moving = [j.name for j in robot.joints if j.joint_type != "fixed"]
        assert (reported == ["j1"]) == (moving == ["j1"]), (
            f"type={joint_type!r}: urdf_joint_names -> {reported}, load_urdf moving joints -> {moving}"
        )


class TestATypeTheLoaderRefusesIsNotReportedAsMovable:
    """A type URDF does not declare is not a movable joint on either reader.

    ``spherical`` is the value that shipped in the movable set; ``screw`` and
    ``ball`` are two more names a reader might reasonably expect a joint format
    to have. None is a URDF joint type, so the two readers must not disagree
    about any of them.
    """

    @pytest.mark.parametrize("joint_type", ["spherical", "screw", "ball"])
    def test_the_loader_refuses_it(self, tmp_path: pathlib.Path, joint_type: str) -> None:
        with pytest.raises(ValueError, match="unknown joint type"):
            loaders.load_urdf(_one_joint_urdf(tmp_path, joint_type))

    @pytest.mark.parametrize("joint_type", ["spherical", "screw", "ball"])
    def test_it_is_not_reported_as_a_movable_joint(self, tmp_path: pathlib.Path, joint_type: str) -> None:
        assert jn.urdf_joint_names(_one_joint_urdf(tmp_path, joint_type)) == []


class TestASpuriousCandidateDamagesTheDecode:
    """What a joint URDF cannot declare does to the name translation.

    ``urdf_joint_names`` feeds ``demangle_usd_joint_names`` the pool of URDF
    names a mangled DOF name may decode to. A name in that pool that no URDF
    can declare has two ways to do harm, and both are reachable from a file.
    """

    def test_a_dof_is_not_translated_to_a_joint_urdf_cannot_declare(self, tmp_path: pathlib.Path) -> None:
        """The importer's own name is kept rather than a name from nowhere."""
        path = tmp_path / "mistranslate.urdf"
        path.write_text(
            '<robot name="r">'
            '<link name="base"/><link name="tip"/>'
            '<joint name="1" type="spherical"><parent link="base"/><child link="tip"/></joint>'
            "</robot>"
        )
        declared = jn.urdf_joint_names(str(path))
        public, usd_to_urdf = jn.demangle_usd_joint_names(["tn__1_"], declared)
        assert public == ["tn__1_"]
        assert usd_to_urdf == {}

    def test_a_real_joint_keeps_its_urdf_name(self, tmp_path: pathlib.Path) -> None:
        """A declarable joint is not robbed of its translation.

        ``a-b`` and ``a.b`` both substitute to ``a_b`` under the legacy
        mangle, so both are candidates for that DOF name. With the second one
        in the pool the decode is ambiguous, and an ambiguous decode keeps the
        USD name - so the revolute joint the URDF really declares surfaces
        under Isaac's mangled name on every public surface. That is the #1900
        leak, reintroduced by a candidate URDF cannot declare.
        """
        path = tmp_path / "ambiguous.urdf"
        path.write_text(
            '<robot name="r">'
            '<link name="base"/><link name="mid"/><link name="tip"/>'
            '<joint name="a-b" type="revolute"><parent link="base"/><child link="mid"/></joint>'
            '<joint name="a.b" type="spherical"><parent link="mid"/><child link="tip"/></joint>'
            "</robot>"
        )
        declared = jn.urdf_joint_names(str(path))
        public, usd_to_urdf = jn.demangle_usd_joint_names(["a_b"], declared)
        assert public == ["a-b"]
        assert usd_to_urdf == {"a_b": "a-b"}


class TestWhatTheRealTypesStillDo:
    """Unchanged behaviour: every expectation here held before the fix too."""

    def test_the_single_dof_types_are_reported_in_file_order(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "arm.urdf"
        path.write_text(
            '<robot name="r">'
            '<link name="l0"/><link name="l1"/><link name="l2"/><link name="l3"/><link name="l4"/>'
            '<joint name="slide" type="prismatic"><parent link="l0"/><child link="l1"/></joint>'
            '<joint name="weld" type="fixed"><parent link="l1"/><child link="l2"/></joint>'
            '<joint name="spin" type="continuous"><parent link="l2"/><child link="l3"/></joint>'
            '<joint name="hinge" type="revolute"><parent link="l3"/><child link="l4"/></joint>'
            "</robot>"
        )
        assert jn.urdf_joint_names(str(path)) == ["slide", "spin", "hinge"]

    @pytest.mark.parametrize("joint_type", ["fixed", "floating", "planar"])
    def test_a_multi_dof_or_fixed_type_is_not_reported(self, tmp_path: pathlib.Path, joint_type: str) -> None:
        assert jn.urdf_joint_names(_one_joint_urdf(tmp_path, joint_type)) == []

    @pytest.mark.parametrize("joint_type", ["fixed", "floating", "planar"])
    def test_the_loader_still_accepts_it_as_a_fixed_joint(self, tmp_path: pathlib.Path, joint_type: str) -> None:
        robot = loaders.load_urdf(_one_joint_urdf(tmp_path, joint_type))
        assert [(j.name, j.joint_type) for j in robot.joints] == [("j1", "fixed")]
