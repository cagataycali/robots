"""A model's several top-level ``<default>`` elements are one root class, merged.

``<default>`` is a top-level MJCF element, and MuJoCo merges every one a model
carries into the single root class - the same model-global treatment it gives
``<compiler>``, ``<asset>`` and ``<worldbody>``. The merge is per attribute and
in document order, so a later element overriding ``size`` does not discard a
``type`` an earlier one declared.

Read independently, each element restarting from nothing, the last one REPLACES
the others. The dropped attribute then fails the familiar silent way: a geom
resolving against the root class finds no ``type``, so ``load_mjcf`` reports the
0.05 m fallback box for a capsule and reports success doing it. ``group`` is read
through the same resolver, so collision/visual filtering is wrong for the same
geoms.

The shape is reachable without a file that writes two ``<default>`` elements
itself, which is what makes it more than a curiosity: ``<include>`` is a textual
splice, so a scene including a robot contributes one element each. Measured on
Menagerie, 7 models carry two top-level ``<default>`` elements once spliced, and
``pal_tiago`` and ``pal_tiago_dual`` lose their whole root class to the replace -
15 of 35 and 11 of 43 geoms respectively had no resolvable ``group``, and one
fewer had no resolvable ``type``.

MuJoCo is the oracle throughout: every expected shape is read back off a compiled
``MjModel`` rather than restated. Its scoping rule for nested classes is pinned
too, because it is not the obvious one - a nested class is snapshotted where it
appears, so it does NOT see a top-level element that follows the one enclosing
it, and threading the root through the loop has to reproduce that rather than
merge everything into everything.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from strands_robots.simulation.isaac.loaders import _mjcf_geom_defaults, load_mjcf

mujoco = pytest.importorskip("mujoco")

_GEOM_TYPE_NAMES = {
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
    int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
}


def _write(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def _compiled_geom(path, geom_name):
    """MuJoCo's own ``(type, size)`` for a named geom - the oracle."""
    model = mujoco.MjModel.from_xml_path(str(path))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    assert gid >= 0, f"premise: MuJoCo compiled no geom named {geom_name!r}"
    # int(): a pybind11 enum does not match a numpy integer as a dict key.
    return _GEOM_TYPE_NAMES[int(model.geom_type[gid])], tuple(float(x) for x in model.geom_size[gid])


def _body(path, name):
    return next(b for b in load_mjcf(str(path)).bodies if b.name == name)


def _resolved_classes(path):
    return _mjcf_geom_defaults(ET.parse(path).getroot(), str(path.parent))


def _assert_matches_mujoco(path, body_name, geom_name, why):
    """The loader's shape for a body equals MuJoCo's for that body's geom."""
    true_type, true_size = _compiled_geom(path, geom_name)
    body = _body(path, body_name)
    assert body.shape == true_type, (
        f"{why}: MuJoCo reads {true_type!r}, and the loader reported {body.shape!r} with size {body.shape_size}"
    )
    # A capsule's or cylinder's MJCF size is (radius, half-length); MuJoCo pads
    # geom_size to three components, so compare the ones it populates.
    assert body.shape_size == pytest.approx(true_size[: len(body.shape_size)], abs=1e-9)
    return true_type, true_size


# The issue's minimal reproduction: the first element declares the ``type``, the
# second overrides only ``size``. MuJoCo reads capsule (0.05, 0.3) - type from
# the first, size from the second.
_TYPE_THEN_SIZE = """<mujoco model="m">
  <default><geom type="capsule" size="0.25 0.4"/></default>
  <default><geom size="0.05 0.3"/></default>
  <worldbody>
    <body name="seg"><geom name="seg_g"/></body>
  </worldbody>
</mujoco>
"""


class TestSeveralTopLevelDefaultsMergeIntoOneRootClass:
    """The regression: every top-level element contributes, and none replaces."""

    def test_a_type_the_first_element_declares_survives_a_later_one(self, tmp_path):
        """The issue's reproduction: ``type`` from the first, ``size`` from the second."""
        path = _write(tmp_path, "two.xml", _TYPE_THEN_SIZE)
        true_type, true_size = _compiled_geom(path, "seg_g")
        assert (true_type, true_size[0], true_size[1]) == ("capsule", 0.05, 0.3), (
            "premise: MuJoCo merges the two top-level <default> elements per attribute"
        )
        _assert_matches_mujoco(
            path,
            "seg",
            "seg_g",
            "a second top-level <default> overriding only size dropped the first one's type",
        )

    def test_three_elements_merge_in_document_order(self, tmp_path):
        """Not a two-element special case, and the last declaration of an attribute wins."""
        path = _write(
            tmp_path,
            "three.xml",
            """<mujoco model="m">
  <default><geom type="capsule"/></default>
  <default><geom size="0.1 0.2"/></default>
  <default><geom size="0.05 0.3"/></default>
  <worldbody>
    <body name="seg"><geom name="seg_g"/></body>
  </worldbody>
</mujoco>
""",
        )
        _, true_size = _assert_matches_mujoco(
            path, "seg", "seg_g", "three top-level <default> elements did not merge in document order"
        )
        assert (true_size[0], true_size[1]) == (0.05, 0.3), "premise: the last size declaration wins"

    @pytest.mark.parametrize(
        "first_open,second_open",
        [
            ("<default>", '<default class="main">'),
            ('<default class="main">', "<default>"),
        ],
        ids=["unnamed_then_named", "named_then_unnamed"],
    )
    def test_the_two_root_spellings_merge_with_each_other(self, tmp_path, first_open, second_open):
        """A file may spell the root either way in either element - one class regardless."""
        path = _write(
            tmp_path,
            "spellings.xml",
            f"""<mujoco model="m">
  {first_open}<geom type="capsule" size="0.25 0.4"/></default>
  {second_open}<geom size="0.05 0.3"/></default>
  <worldbody>
    <body name="seg"><geom name="seg_g"/></body>
  </worldbody>
</mujoco>
""",
        )
        _assert_matches_mujoco(
            path,
            "seg",
            "seg_g",
            "the root's two spellings across two top-level elements did not merge into one class",
        )

    @pytest.mark.parametrize(
        "worldbody",
        [
            '<body name="seg"><geom name="seg_g" class="main"/></body>',
            '<body name="p" childclass="main"><body name="seg"><geom name="seg_g"/></body></body>',
        ],
        ids=["geom_names_main", "body_childclass_main"],
    )
    def test_every_arrival_path_sees_the_merge(self, tmp_path, worldbody):
        """The fix is in the resolver, so all three ways to reach the root are covered."""
        path = _write(
            tmp_path,
            "arrival.xml",
            f"""<mujoco model="m">
  <default><geom type="capsule" size="0.25 0.4"/></default>
  <default><geom size="0.05 0.3"/></default>
  <worldbody>
    {worldbody}
  </worldbody>
</mujoco>
""",
        )
        _assert_matches_mujoco(
            path, "seg", "seg_g", "a geom reaching the merged root by an explicit name saw only one element"
        )

    def test_an_included_fragment_contributing_a_second_element_merges(self, tmp_path):
        """The reachable shape: the two elements are in two files, spliced by <include>.

        This is what makes the bug more than a curiosity - a scene including a
        robot contributes one top-level ``<default>`` each, and no file has to
        write two for the resolver to see two.
        """
        _write(
            tmp_path,
            "robot.xml",
            '<mujoco model="r"><default><geom size="0.05 0.3"/></default></mujoco>\n',
        )
        path = _write(
            tmp_path,
            "scene.xml",
            """<mujoco model="m">
  <default><geom type="capsule" size="0.25 0.4"/></default>
  <include file="robot.xml"/>
  <worldbody>
    <body name="seg"><geom name="seg_g"/></body>
  </worldbody>
</mujoco>
""",
        )
        assert len(_resolved_classes(path)[""]) == 2, (
            "premise: the splice really did contribute a second top-level <default>"
        )
        _assert_matches_mujoco(path, "seg", "seg_g", "an <include>d fragment's <default> replaced the including file's")


class TestTheResolverMergesRatherThanReplaces:
    """The merge at the point it is made, not only through the shape it produces."""

    def test_the_root_class_carries_attributes_from_every_element(self, tmp_path):
        path = _write(tmp_path, "two.xml", _TYPE_THEN_SIZE)
        root_class = _resolved_classes(path)[""]
        assert root_class == {"type": "capsule", "size": "0.05 0.3"}, (
            f"the root class should carry the first element's type and the second's size, got {root_class}"
        )

    def test_both_spellings_report_the_merged_class(self, tmp_path):
        """The root is published under ``""`` and ``"main"``, and the merge reaches both."""
        path = _write(tmp_path, "two.xml", _TYPE_THEN_SIZE)
        classes = _resolved_classes(path)
        assert classes[""] == classes["main"]
        assert "type" in classes["main"], "the alias must publish the merged class, not one element's"


class TestNestedClassesFollowDocumentOrder:
    """MuJoCo snapshots a nested class where it appears; the resolver must too.

    This is the boundary of the fix. Threading the root through the loop must not
    become "merge every element into every class": a nested class inherits the
    root as accumulated *up to its own position*, so an element that follows the
    one enclosing it does not reach it.
    """

    def test_a_nested_class_inherits_the_root_accumulated_before_it(self, tmp_path):
        """The enclosing element's own attributes reach it, and its own still win."""
        path = _write(
            tmp_path,
            "nested.xml",
            """<mujoco model="m">
  <default>
    <geom type="capsule" size="0.07 0.31"/>
    <default class="leg"><geom size="0.06 0.2"/></default>
  </default>
  <default><geom size="0.9 0.9"/></default>
  <worldbody>
    <body name="leg_b"><geom name="leg_g" class="leg"/></body>
    <body name="root_b"><geom name="root_g"/></body>
  </worldbody>
</mujoco>
""",
        )
        # The nested class keeps its own size and takes type from the element
        # enclosing it; the root takes the later element's size.
        _assert_matches_mujoco(path, "leg_b", "leg_g", "a nested class lost the type of the element enclosing it")
        _assert_matches_mujoco(path, "root_b", "root_g", "the root class did not take the later element's size")

        classes = _resolved_classes(path)
        assert classes["leg"] == {"type": "capsule", "size": "0.06 0.2"}
        assert classes[""] == {"type": "capsule", "size": "0.9 0.9"}

    def test_a_nested_class_does_not_inherit_a_later_top_level_element(self, tmp_path):
        """MuJoCo's scoping rule, asserted where the resolver decides it.

        Asserted on the resolver rather than on the reported shape: the geom here
        resolves to no ``type`` at all, and what the loader reports for a typeless
        geom (its ``box`` fallback) diverges from MuJoCo's own ``sphere`` default
        for reasons that predate this fix and are not its business. The scoping
        claim is checked against MuJoCo directly instead - the compiled model must
        not read the later element's ``type``.
        """
        path = _write(
            tmp_path,
            "later.xml",
            """<mujoco model="m">
  <default>
    <default class="leg"><geom size="0.06 0.2"/></default>
  </default>
  <default><geom type="capsule" size="0.5 0.6"/></default>
  <worldbody>
    <body name="leg_b"><geom name="leg_g" class="leg"/></body>
    <body name="root_b"><geom name="root_g"/></body>
  </worldbody>
</mujoco>
""",
        )
        assert _compiled_geom(path, "leg_g")[0] != "capsule", (
            "premise: MuJoCo does not give a nested class the type of a top-level "
            "element that follows the one enclosing it"
        )

        classes = _resolved_classes(path)
        assert "type" in classes[""], "premise: the later element's type reaches the root class"
        assert classes["leg"] == {"size": "0.06 0.2"}, (
            f"the nested class took an attribute from a top-level element declared after the one "
            f"enclosing it, so threading the root over-merged: {classes['leg']}"
        )
        # The root itself still merges, so the two rules coexist.
        _assert_matches_mujoco(path, "root_b", "root_g", "the root class did not see the second element")

    def test_mujoco_refuses_one_nested_class_name_in_two_elements(self):
        """Why threading cannot make a nested class ambiguous across elements.

        A premise, expected to hold before and after the fix. If a future MuJoCo
        permits this, the reasoning behind writing each nested class to a single
        key fails loudly here rather than silently merging two distinct classes.
        """
        with pytest.raises(ValueError, match="repeated default class name"):
            mujoco.MjModel.from_xml_string(
                """<mujoco model="m">
  <default><default class="leg"><geom type="capsule"/></default></default>
  <default><default class="leg"><geom size="0.05 0.3"/></default></default>
  <worldbody><body name="b"><geom name="g" class="leg"/></body></worldbody>
</mujoco>
"""
            )


class TestWhatTheMergeMustNotChange:
    """Controls: the single-element case, precedence, and non-vacuity."""

    def test_a_single_top_level_default_reads_as_before(self, tmp_path):
        """The overwhelmingly common shape, and the one the fix must leave alone."""
        path = _write(
            tmp_path,
            "one.xml",
            """<mujoco model="m">
  <default><geom type="capsule" size="0.06 0.21"/></default>
  <worldbody>
    <body name="seg"><geom name="seg_g"/></body>
  </worldbody>
</mujoco>
""",
        )
        _assert_matches_mujoco(path, "seg", "seg_g", "a model with one top-level <default> changed")
        assert _resolved_classes(path)[""] == {"type": "capsule", "size": "0.06 0.21"}

    def test_the_geoms_own_attribute_still_beats_the_merged_root(self, tmp_path):
        path = _write(
            tmp_path,
            "own.xml",
            """<mujoco model="m">
  <default><geom type="capsule" size="0.25 0.4"/></default>
  <default><geom size="0.05 0.3"/></default>
  <worldbody>
    <body name="seg"><geom name="seg_g" type="box" size="0.11 0.12 0.13"/></body>
  </worldbody>
</mujoco>
""",
        )
        true_type, _ = _assert_matches_mujoco(
            path, "seg", "seg_g", "the merged root class overrode the geom's own attributes"
        )
        assert true_type == "box", "premise: MuJoCo lets the geom's own type win over its class"

    def test_two_elements_declaring_no_geom_contribute_nothing(self, tmp_path):
        """Non-vacuity: two top-level elements alone do not populate the root class.

        Menagerie's ``aloha/scene.xml`` and ``pal_talos`` are this shape - two
        spliced top-level ``<default>`` elements carrying only nested classes -
        and their root class is empty on both trees. So the four Menagerie
        verdicts this fix changes are changed by the merge, not by the count.
        """
        path = _write(
            tmp_path,
            "empty_roots.xml",
            """<mujoco model="m">
  <default><default class="leg"><geom type="capsule" size="0.06 0.2"/></default></default>
  <default><default class="arm"><geom type="cylinder" size="0.04 0.1"/></default></default>
  <worldbody>
    <body name="seg"><geom name="seg_g" class="leg"/></body>
  </worldbody>
</mujoco>
""",
        )
        classes = _resolved_classes(path)
        assert classes[""] == {}, f"neither element declares a <geom>, so the root class is empty: {classes['']}"
        assert classes["leg"] == {"type": "capsule", "size": "0.06 0.2"}
        assert classes["arm"] == {"type": "cylinder", "size": "0.04 0.1"}
        _assert_matches_mujoco(path, "seg", "seg_g", "a nested class under one of two bare elements broke")

    def test_a_class_the_model_never_declares_still_contributes_nothing(self, tmp_path):
        """The merge did not turn the mapping into one that answers for every name."""
        path = _write(tmp_path, "two.xml", _TYPE_THEN_SIZE)
        assert _resolved_classes(path).get("nosuchclass") is None
