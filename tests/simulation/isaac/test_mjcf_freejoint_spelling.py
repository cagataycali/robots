"""MJCF's two spellings of a free joint report one model.

MJCF states a floating base two ways and MuJoCo compiles both to the same
``mjJNT_FREE``: ``<joint type="free">`` and the dedicated ``<freejoint>``
element. ``load_mjcf`` walked ``body_el.findall("joint")``, so only the first
spelling was read and a ``<freejoint>`` produced no :class:`JointDef` at all --
the base's six degrees of freedom were absent from the report rather than
reported as a joint the caller could see and skip.

That is the spelling the shipped asset corpus uses. Of the 46 loadable robot
MJCFs under ``robot_descriptions`` that declare a free joint, every one spells
it ``<freejoint>`` and none spells ``type="free"``: the quadrupeds
(``unitree_go2``, ``anymal_c``, ``spot``), the humanoids (``unitree_h1``,
``talos``, ``berkeley_humanoid``) and, most visibly, the aircraft --
``skydio_x2`` and ``bitcraze_crazyflie_2`` have exactly one joint each, the free
base, so the loader reported a robot with no joints whatsoever.

Every expectation here is derived from ``mujoco.MjModel`` rather than restated
by hand, and MuJoCo also fixes the two rules the implementation rests on:

* a free joint may not share a body with any other joint (``more than 6 dofs in
  body``), so a ``<freejoint>`` is the only joint on its body and the document
  order of the two tags is never ambiguous; and
* ``<freejoint>`` resolves no default class -- MJCF has no
  ``<default><freejoint>`` block, and MuJoCo gives such a joint the built-in
  damping and armature even where a ``<default><joint>`` class is in force,
  while the ``type="free"`` spelling does inherit that class.

Deliberately unchanged: what ``free`` MAPS to. ``_MJCF_JOINT_TYPE_MAP`` reports
it as ``"fixed"`` because :class:`JointDef` has no 6-DOF spelling, so a floating
base is still excluded from ``num_joints``. Whether the dataclass should gain
one is a contract question about every producer of a ``ProceduralRobot``; this
change is only that the two spellings of one model agree, and the mapping is
pinned below so a later answer has to move it explicitly.
"""

from __future__ import annotations

import pytest

from strands_robots.simulation.isaac.loaders import _MJCF_JOINT_TYPE_MAP, load_mjcf

mujoco = pytest.importorskip("mujoco")

#: The two ways MJCF spells a free joint, with no attributes on either.
SPELLINGS = ("<freejoint/>", '<joint type="free"/>')

#: A ``<default><joint>`` class supplying the two scalars a joint declaration
#: can inherit. MuJoCo applies it to the ``type="free"`` spelling only.
CLASS_DEFAULTS = '<default><joint damping="5" armature="0.3"/></default>'

#: Attributes ``<freejoint>`` accepts beside ``name``, neither of which changes
#: the fact that a joint is declared.
ACCEPTED_EXTRAS = ('group="2"', 'align="true"')

#: Attributes ``<freejoint>`` does not accept, so no reader has to resolve them.
REFUSED_EXTRAS = ('type="free"', 'axis="0 0 1"', 'class="c"')

#: The fields of a reported joint, compared as a whole so no difference between
#: two reports of one model can hide in a field this suite forgot to name.
FIELDS = (
    "name",
    "joint_type",
    "parent_body",
    "child_body",
    "axis",
    "limit_lower",
    "limit_upper",
    "damping",
    "armature",
)


def _xml(joints: str, *, defaults: str = "", body: str = "base") -> str:
    """An MJCF whose single body carries ``joints``."""
    return (
        f"<mujoco>{defaults}<worldbody>"
        f'<body name="{body}" pos="0 0 0.5">'
        f'<geom name="g" type="box" size="0.1 0.1 0.05"/>'
        f"{joints}"
        f"</body></worldbody></mujoco>"
    )


def _write(tmp_path, joints: str, *, defaults: str = "", name: str = "robot") -> str:
    path = tmp_path / f"{name}.xml"
    path.write_text(_xml(joints, defaults=defaults), encoding="utf-8")
    return str(path)


def _rows(path: str) -> list[tuple]:
    """Every joint ``load_mjcf`` reports for the file, field by field."""
    return [tuple(getattr(j, f) for f in FIELDS) for j in load_mjcf(path).joints]


def _compiled(path: str) -> tuple[int, int, int, tuple[int, ...]]:
    """What MuJoCo's compiler makes of the same file."""
    model = mujoco.MjModel.from_xml_path(path)
    return model.njnt, model.nq, model.nv, tuple(int(t) for t in model.jnt_type)


def _free_dof(path: str) -> list[tuple[float, float]]:
    """MuJoCo's own damping/armature on the first dof of each free joint."""
    model = mujoco.MjModel.from_xml_path(path)
    out = []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            dof = int(model.jnt_dofadr[j])
            out.append((float(model.dof_damping[dof]), float(model.dof_armature[dof])))
    return out


class TestThePremisesMujocoFixes:
    """The rules the implementation rests on, each read from MuJoCo itself."""

    def test_the_two_spellings_compile_to_the_same_model(self, tmp_path):
        compiled = {s: _compiled(_write(tmp_path, s, name=f"m{i}")) for i, s in enumerate(SPELLINGS)}
        assert len(set(compiled.values())) == 1, (
            f"premise: the spellings describe one model, but MuJoCo compiled {compiled}"
        )
        njnt, nq, nv, types = compiled[SPELLINGS[0]]
        assert (njnt, nq, nv) == (1, 7, 6)
        assert types == (int(mujoco.mjtJoint.mjJNT_FREE),)

    @pytest.mark.parametrize("spelling", SPELLINGS)
    def test_a_free_joint_may_not_share_a_body_with_another_joint(self, tmp_path, spelling):
        """So a body carrying one carries nothing else, and order cannot differ."""
        path = _write(tmp_path, f'{spelling}<joint type="hinge" axis="0 0 1"/>', name="both")
        with pytest.raises(ValueError, match="more than 6 dofs"):
            mujoco.MjModel.from_xml_path(path)

    @pytest.mark.parametrize("extra", REFUSED_EXTRAS)
    def test_freejoint_refuses_the_attributes_it_does_not_own(self, tmp_path, extra):
        path = _write(tmp_path, f"<freejoint {extra}/>", name="extra")
        with pytest.raises(ValueError):
            mujoco.MjModel.from_xml_path(path)

    @pytest.mark.parametrize("extra", ACCEPTED_EXTRAS)
    def test_freejoint_accepts_its_own_extras(self, tmp_path, extra):
        path = _write(tmp_path, f"<freejoint {extra}/>", name="ok")
        assert _compiled(path)[0] == 1

    def test_mjcf_has_no_default_freejoint_block(self, tmp_path):
        path = _write(tmp_path, "<freejoint/>", defaults="<default><freejoint/></default>", name="dflt")
        with pytest.raises(ValueError):
            mujoco.MjModel.from_xml_path(path)

    def test_a_joint_class_reaches_one_spelling_and_not_the_other(self, tmp_path):
        """MuJoCo's own answer, which is the answer the loader has to mirror."""
        free = _write(tmp_path, "<freejoint/>", defaults=CLASS_DEFAULTS, name="cf")
        typed = _write(tmp_path, '<joint type="free"/>', defaults=CLASS_DEFAULTS, name="ct")
        assert _free_dof(free) == [(0.0, 0.0)], "premise: a <freejoint> inherits no class"
        assert _free_dof(typed) == [(5.0, 0.3)], "premise: the type='free' spelling does"


class TestAFreejointIsReported:
    """The regression: a ``<freejoint>`` produced no joint at all."""

    def test_a_body_whose_only_joint_is_a_freejoint_reports_one(self, tmp_path):
        path = _write(tmp_path, "<freejoint/>")
        assert _compiled(path)[0] == 1, "premise: MuJoCo compiles exactly one joint"
        rows = _rows(path)
        assert len(rows) == 1, f"MuJoCo compiles 1 joint for this model; the loader reported {len(rows)}"

    def test_a_named_freejoint_is_reported_under_its_own_name(self, tmp_path):
        path = _write(tmp_path, '<freejoint name="floating_base"/>')
        assert [r[0] for r in _rows(path)] == ["floating_base"]

    def test_an_unnamed_freejoint_takes_the_bodys_fallback_name(self, tmp_path):
        """The same fallback every unnamed ``<joint>`` gets, so nothing is special."""
        path = _write(tmp_path, "<freejoint/>", name="fallback")
        assert [r[0] for r in _rows(path)] == ["base_joint_0"]

    @pytest.mark.parametrize("extra", ACCEPTED_EXTRAS)
    def test_the_extras_freejoint_accepts_do_not_hide_it(self, tmp_path, extra):
        path = _write(tmp_path, f"<freejoint {extra}/>", name="extras")
        assert len(_rows(path)) == 1

    def test_a_floating_base_beside_actuated_joints_is_reported_too(self, tmp_path):
        """The shipped humanoid shape: a free root body over an articulated chain."""
        path = tmp_path / "chain.xml"
        path.write_text(
            "<mujoco><worldbody>"
            '<body name="pelvis" pos="0 0 1"><freejoint name="floating_base"/>'
            '<geom type="box" size="0.1 0.1 0.1"/>'
            '<body name="thigh" pos="0 0 -0.2">'
            '<joint name="hip" type="hinge" axis="0 1 0"/>'
            '<geom type="box" size="0.05 0.05 0.1"/>'
            "</body></body></worldbody></mujoco>",
            encoding="utf-8",
        )
        njnt = _compiled(str(path))[0]
        names = [r[0] for r in _rows(str(path))]
        assert names == ["floating_base", "hip"], f"MuJoCo compiles {njnt} joints; the loader reported {names}"


class TestBothSpellingsReportOneModel:
    """Two spellings of one model, one report."""

    def test_the_reports_are_identical_field_for_field(self, tmp_path):
        reports = {s: _rows(_write(tmp_path, s, name=f"s{i}")) for i, s in enumerate(SPELLINGS)}
        free, typed = reports["<freejoint/>"], reports['<joint type="free"/>']
        assert free == typed, f"<freejoint/> reported {free}; type='free' reported {typed}"

    def test_the_reported_joint_count_matches_the_compiler(self, tmp_path):
        for i, spelling in enumerate(SPELLINGS):
            path = _write(tmp_path, spelling, name=f"c{i}")
            assert len(_rows(path)) == _compiled(path)[0], f"{spelling} under-reports the compiled joints"


class TestTheDefaultClassBoundaryFollowsMjcf:
    """Where the spellings legitimately differ, they differ MuJoCo's way."""

    def test_the_class_reaches_the_typed_spelling_only(self, tmp_path):
        free = _write(tmp_path, "<freejoint/>", defaults=CLASS_DEFAULTS, name="lf")
        typed = _write(tmp_path, '<joint type="free"/>', defaults=CLASS_DEFAULTS, name="lt")
        assert _free_dof(free) != _free_dof(typed), "premise: MuJoCo itself diverges here"
        free_row, typed_row = _rows(free)[0], _rows(typed)[0]
        assert (typed_row[FIELDS.index("damping")], typed_row[FIELDS.index("armature")]) == (5.0, 0.3)
        assert (free_row[FIELDS.index("damping")], free_row[FIELDS.index("armature")]) != (5.0, 0.3), (
            "a <freejoint> inherits no default class in MJCF, so the loader must not apply one"
        )

    def test_a_class_declaring_a_type_does_not_retype_a_freejoint(self, tmp_path):
        """A ``<default><joint type="slide">`` class names a kind of ``<joint>``."""
        path = _write(
            tmp_path,
            "<freejoint/>",
            defaults='<default><joint type="slide" axis="1 0 0"/></default>',
            name="retype",
        )
        assert _compiled(path)[3] == (int(mujoco.mjtJoint.mjJNT_FREE),), "premise: still a free joint"
        assert [r[FIELDS.index("joint_type")] for r in _rows(path)] == ["fixed"]


class TestTheOtherJointsAreUnchanged:
    """Controls: reading a second tag must not disturb the first."""

    @pytest.mark.parametrize(
        ("mjcf_type", "reported"),
        [("hinge", "revolute"), ("slide", "prismatic"), ("ball", "fixed"), ("free", "fixed")],
    )
    def test_every_declared_type_still_maps_as_before(self, tmp_path, mjcf_type, reported):
        path = _write(tmp_path, f'<joint name="j" type="{mjcf_type}" axis="0 1 0"/>', name=mjcf_type)
        assert [r[FIELDS.index("joint_type")] for r in _rows(path)] == [reported]

    def test_a_joints_own_class_is_still_resolved(self, tmp_path):
        path = _write(
            tmp_path,
            '<joint name="j" class="soft"/>',
            defaults='<default class="soft"><joint type="hinge" axis="0 1 0" damping="7"/></default>',
            name="cls",
        )
        row = _rows(path)[0]
        assert row[FIELDS.index("joint_type")] == "revolute"
        assert row[FIELDS.index("damping")] == 7.0

    def test_an_unknown_joint_type_is_still_refused(self, tmp_path):
        path = _write(tmp_path, '<joint name="j" type="screw"/>', name="unknown")
        with pytest.raises(ValueError, match="unknown joint type"):
            load_mjcf(path)

    def test_a_chains_joints_are_still_reported_in_tree_order(self, tmp_path):
        """One joint per body: the loader's topology guard refuses two on one edge."""
        path = tmp_path / "order.xml"
        path.write_text(
            '<mujoco><worldbody><body name="base">'
            '<joint name="first" type="hinge" axis="1 0 0"/>'
            '<geom type="box" size="0.1 0.1 0.1"/>'
            '<body name="arm" pos="0 0 0.2">'
            '<joint name="second" type="slide" axis="0 0 1"/>'
            '<geom type="box" size="0.05 0.05 0.1"/>'
            "</body></body></worldbody></mujoco>",
            encoding="utf-8",
        )
        assert [r[0] for r in _rows(str(path))] == ["first", "second"]

    def test_two_joints_on_one_body_are_still_refused(self, tmp_path):
        """The compound-joint guard, which a free joint can never reach: MuJoCo
        refuses a free joint beside another joint before the loader is asked."""
        path = tmp_path / "compound.xml"
        path.write_text(
            '<mujoco><worldbody><body name="base">'
            '<geom type="box" size="0.1 0.1 0.1"/>'
            '<body name="arm" pos="0 0 0.2">'
            '<joint name="first" type="hinge" axis="1 0 0"/>'
            '<joint name="second" type="slide" axis="0 0 1"/>'
            '<geom type="box" size="0.05 0.05 0.1"/>'
            "</body></body></worldbody></mujoco>",
            encoding="utf-8",
        )
        assert _compiled(str(path))[0] == 2, "premise: MuJoCo compiles the 2-DOF pair"
        with pytest.raises(ValueError, match="duplicate parent->child body edges"):
            load_mjcf(str(path))


class TestTheJointTypeVocabularyIsUnchanged:
    """The boundary: what ``free`` maps to is a separate contract question."""

    def test_free_is_still_reported_as_fixed(self):
        assert _MJCF_JOINT_TYPE_MAP["free"] == "fixed"

    def test_a_floating_base_is_still_excluded_from_the_actuated_count(self, tmp_path):
        path = _write(tmp_path, "<freejoint/>", name="count")
        robot = load_mjcf(path)
        assert len(robot.joints) == 1, "the joint is reported"
        assert robot.num_joints == 0, "and is still not counted as an actuated DOF"
