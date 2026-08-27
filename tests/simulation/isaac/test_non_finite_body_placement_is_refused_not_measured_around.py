"""A body whose own placement is not finite is refused, not measured around.

``load_mjcf_scene_objects`` composes a
:class:`~strands_robots.simulation.isaac.loaders.SceneObject` from three
placements: the top-level body's ``pos`` becomes the object's ``position``, its
orientation becomes ``quat``, and each nested body's ``pos`` becomes the running
offset every geom below it is measured from. ``_geom_aabb`` refuses a non-finite
value on all four of a *geom*'s parse paths. The *body* placements the same
bound is composed from were read straight through ``_parse_xyz`` /
``_parse_orientation``, whose contract is to fall back to a default on an
unreadable attribute and to return whatever a readable one parsed to.

The nested case is the one nothing can detect. ``_recursive_collision_aabb``
unions its subtree with a running ``min``/``max``, and Python orders a NaN as
neither smaller nor larger than anything, so a comparison against one keeps the
accumulator it started with. A non-finite offset makes every bound below it
non-finite, so the whole subtree disappears from the union::

    table, leg body pos="0 0 -0.37"  -> size (0.8, 0.8, 0.76)
    table, leg body pos="0 0 nan"    -> size (0.8, 0.8, 0.04)
    table with the leg deleted       -> size (0.8, 0.8, 0.04)

-- a 4 cm slab where the file declares a 76 cm table, byte-identical to the same
fixture with the leg deleted, under ``status`` success, and with every field the
loader reports finite. That last part is what separates it from the geom
spellings: a consumer cannot screen for it. The other placements fail more
loudly but no more correctly - a non-finite top-level ``pos`` or orientation is
reported verbatim as the object's own, and ``pos="0 0 inf"`` on a nested body
reports an infinite extent, because ``inf - inf`` is a NaN the outer accumulator
drops in turn.

Reachability is wider here than for the geom spellings, and the difference is
worth being precise about. The geom finding is scoped to fixtures: a non-finite
geom makes the inertia MuJoCo derives non-finite, so it refuses a body with a
free joint and compiles the same geom on one without. A body's own ``pos`` is
not an input to that derivation, so MuJoCo compiles a non-finite body placement
with or without a free joint -- movable task objects as well as the tables and
cabinets whose footprint a manipulation scene is planned against. Every scene
carrying one reaches the reader, which is why it cannot defer the question to a
compile step.

The disposition follows the sibling rather than inventing one: refuse, naming
the body and the attribute, because ``None`` from these readers means "this
attribute was unreadable, use the documented default" and a caller cannot tell
that apart from "this attribute was read and is not a number".
"""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.simulation.isaac.loaders import (  # noqa: E402
    _ORIENTATION_SPELLINGS,
    _refuse_non_finite_body,
    _refuse_non_finite_placement,
    load_mjcf_scene_objects,
)

_TOP_GEOM = '<geom name="t_top" type="box" size="0.40 0.40 0.02"/>'
_LEG_GEOM = '<geom name="t_leg" type="box" size="0.03 0.03 0.37"/>'

# Every spelling of a non-finite body placement, as (label, attribute, scene).
# The leg hangs BELOW the tabletop, so the nested body's own pos is what extends
# the bound downward: a fixture whose child sits inside the parent's span cannot
# show a dropped contribution at all.
_NESTED_NON_FINITE = [
    ("nested-pos-nan", "pos", '<body name="t_leg_body" pos="0 0 nan">'),
    ("nested-pos-inf", "pos", '<body name="t_leg_body" pos="0 0 inf">'),
    ("nested-pos-negative-inf", "pos", '<body name="t_leg_body" pos="0 0 -inf">'),
]
_TOP_NON_FINITE = [
    ("top-pos-nan", "pos", 'pos="nan 0 0.75"'),
    ("top-pos-inf", "pos", 'pos="inf 0 0.75"'),
    ("top-quat-nan", "quat", 'pos="0 0 0.75" quat="nan 0 0 1"'),
    ("top-euler-nan", "euler", 'pos="0 0 0.75" euler="nan 0 0"'),
    ("top-axisangle-nan", "axisangle", 'pos="0 0 0.75" axisangle="0 0 1 nan"'),
    ("top-zaxis-nan", "zaxis", 'pos="0 0 0.75" zaxis="nan 0 1"'),
]


def _ids(rows: list[tuple[str, str, str]]) -> list[str]:
    return [row[0] for row in rows]


def _scene(*, top: str = 'pos="0 0 0.75"', leg_open: str = '<body name="t_leg_body" pos="0 0 -0.37">') -> str:
    """A static table fixture: a tabletop plus a leg on a nested body.

    Args:
        top: The top-level body's own attributes.
        leg_open: The opening tag of the nested body carrying the leg.

    Returns:
        A complete MJCF document.
    """
    return (
        f'<mujoco model="kitchen"><worldbody>'
        f'<body name="kitchen_table" {top}>{_TOP_GEOM}'
        f"{leg_open}{_LEG_GEOM}</body>"
        f"</body></worldbody></mujoco>"
    )


def _write(tmp_path: Path, scene: str) -> str:
    path = tmp_path / "kitchen.xml"
    path.write_text(scene)
    return str(path)


def _size(tmp_path: Path, scene: str) -> tuple[float, float, float]:
    objects = load_mjcf_scene_objects(_write(tmp_path, scene))
    assert objects, "the fixture declares one object; the reader found none"
    return tuple(round(v, 4) for v in objects[0].size)  # type: ignore[return-value]


class TestThePremisesTheFindingRestsOn:
    """These hold before and after the fix; the report is unfounded without them."""

    @pytest.mark.parametrize(("_label", "_attribute", "leg_open"), _NESTED_NON_FINITE, ids=_ids(_NESTED_NON_FINITE))
    def test_mujoco_compiles_a_non_finite_nested_body(
        self, tmp_path: Path, _label: str, _attribute: str, leg_open: str
    ) -> None:
        # A non-finite body placement on a fixture is a model MuJoCo accepts, so
        # the reader really does meet these scenes.
        model = mujoco.MjModel.from_xml_path(_write(tmp_path, _scene(leg_open=leg_open)))
        assert model.ngeom == 2

    @pytest.mark.parametrize(("_label", "_attribute", "top"), _TOP_NON_FINITE, ids=_ids(_TOP_NON_FINITE))
    def test_mujoco_compiles_a_non_finite_top_level_body(
        self, tmp_path: Path, _label: str, _attribute: str, top: str
    ) -> None:
        model = mujoco.MjModel.from_xml_path(_write(tmp_path, _scene(top=top)))
        assert model.ngeom == 2

    def test_a_free_joint_does_not_narrow_the_reachability_the_way_it_does_for_a_geom(self, tmp_path: Path) -> None:
        # The geom finding is scoped to fixtures: a non-finite geom makes the
        # inertia MuJoCo derives non-finite, so it refuses a moving body. A
        # body's own pos is not an input to that derivation, so MuJoCo compiles
        # it either way and this finding is NOT scoped to fixtures. A free joint
        # is only legal on a top-level body, so that is where it is declared.
        moving = (
            '<mujoco model="kitchen"><worldbody>'
            '<body name="kitchen_table" pos="0 0 0.75"><freejoint/>'
            f'{_TOP_GEOM}<body name="t_leg_body" pos="0 0 nan">{_LEG_GEOM}</body>'
            "</body></worldbody></mujoco>"
        )
        assert mujoco.MjModel.from_xml_path(_write(tmp_path, moving)).ngeom == 2
        assert mujoco.MjModel.from_xml_path(_write(tmp_path, _scene(leg_open='<body pos="0 0 nan">'))).ngeom == 2

    def test_a_running_min_max_drops_a_nan_rather_than_propagating_it(self) -> None:
        # The mechanism itself, so a reader need not take it on trust.
        assert min(float("inf"), float("nan")) == float("inf")
        assert max(float("-inf"), float("nan")) == float("-inf")
        # And why an infinite offset lands as an infinite extent rather than
        # being dropped: the subtraction that leaves is a NaN one level up.
        assert math.isnan(float("inf") - float("inf"))

    def test_the_geom_spellings_were_already_refused(self, tmp_path: Path) -> None:
        # The in-file control for the whole finding: the sibling guard covers a
        # geom's own placement, which is why only the body placements remained.
        scene = (
            '<mujoco model="kitchen"><worldbody><body name="kitchen_table" pos="0 0 0.75">'
            f'{_TOP_GEOM}<geom name="t_side" type="box" size="0.1 0.1 0.1" pos="nan 0 0"/>'
            "</body></worldbody></mujoco>"
        )
        with pytest.raises(ValueError, match=r"geom 't_side': pos has a component that is not finite"):
            load_mjcf_scene_objects(_write(tmp_path, scene))

    def test_the_orientation_spellings_are_the_modules_own_vocabulary(self) -> None:
        # The refusal names the attribute the file used, taken from the list the
        # reader already resolves orientation through rather than a second copy.
        named = {attribute for _, attribute, _ in _TOP_NON_FINITE}
        assert named & set(_ORIENTATION_SPELLINGS) == named - {"pos"}


class TestANonFiniteBodyPlacementIsRefusedByName:
    """The regression: every spelling refuses, naming the body and the attribute."""

    @pytest.mark.parametrize(("_label", "attribute", "leg_open"), _NESTED_NON_FINITE, ids=_ids(_NESTED_NON_FINITE))
    def test_a_nested_body_is_refused(self, tmp_path: Path, _label: str, attribute: str, leg_open: str) -> None:
        with pytest.raises(ValueError, match="is not finite") as excinfo:
            load_mjcf_scene_objects(_write(tmp_path, _scene(leg_open=leg_open)))
        assert f"{attribute} has a component that is not finite" in str(excinfo.value)

    @pytest.mark.parametrize(("_label", "attribute", "top"), _TOP_NON_FINITE, ids=_ids(_TOP_NON_FINITE))
    def test_a_top_level_body_is_refused(self, tmp_path: Path, _label: str, attribute: str, top: str) -> None:
        with pytest.raises(ValueError, match="is not finite") as excinfo:
            load_mjcf_scene_objects(_write(tmp_path, _scene(top=top)))
        assert f"{attribute} has a component that is not finite" in str(excinfo.value)

    def test_the_refusal_names_the_body_rather_than_a_geom(self, tmp_path: Path) -> None:
        # Bodies and geoms are located differently; reporting one as the other
        # sends a reader to an element that is not at fault.
        with pytest.raises(ValueError, match=r"^body 't_leg_body': ") as excinfo:
            load_mjcf_scene_objects(_write(tmp_path, _scene(leg_open='<body name="t_leg_body" pos="0 0 nan">')))
        assert "geom" not in str(excinfo.value).split(":")[0]

    def test_an_unnamed_nested_body_is_located_as_a_body(self, tmp_path: Path) -> None:
        # A body carries no ``type``, so the geom locator's resolved-type
        # fallback has nothing to report; ``<body>`` is what is left.
        with pytest.raises(ValueError, match=r"^unnamed <body>: pos has a component"):
            load_mjcf_scene_objects(_write(tmp_path, _scene(leg_open='<body pos="0 0 nan">')))

    def test_the_fixture_is_no_longer_measured_as_its_tabletop_alone(self, tmp_path: Path) -> None:
        # The headline: the reported size was byte-identical to the same fixture
        # with the leg deleted, and every field it reported was finite.
        healthy = _size(tmp_path, _scene())
        deleted = _size(
            tmp_path,
            '<mujoco model="kitchen"><worldbody><body name="kitchen_table" pos="0 0 0.75">'
            f"{_TOP_GEOM}</body></worldbody></mujoco>",
        )
        assert healthy != deleted, "the fixture cannot show a dropped subtree"
        assert healthy[2] == pytest.approx(0.76)
        assert deleted[2] == pytest.approx(0.04)
        with pytest.raises(ValueError, match="is not finite"):
            load_mjcf_scene_objects(_write(tmp_path, _scene(leg_open='<body name="t_leg_body" pos="0 0 nan">')))

    def test_an_infinite_offset_is_not_reported_as_an_infinite_extent(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="is not finite"):
            load_mjcf_scene_objects(_write(tmp_path, _scene(leg_open='<body name="t_leg_body" pos="0 0 inf">')))

    def test_a_grandchild_under_a_non_finite_parent_is_refused(self, tmp_path: Path) -> None:
        # The offset is composed as the walk descends, so a non-finite component
        # anywhere above a geom drops it. The refusal names the body at fault
        # rather than the deepest one carrying geometry.
        scene = (
            '<mujoco model="kitchen"><worldbody><body name="kitchen_table" pos="0 0 0.75">'
            f'{_TOP_GEOM}<body name="t_frame" pos="0 0 nan">'
            f'<body name="t_leg_body" pos="0 0 -0.37">{_LEG_GEOM}</body></body>'
            "</body></worldbody></mujoco>"
        )
        with pytest.raises(ValueError, match=r"^body 't_frame': pos has a component"):
            load_mjcf_scene_objects(_write(tmp_path, scene))

    def test_the_refusal_says_what_the_alternative_would_have_been(self, tmp_path: Path) -> None:
        # The message earns the refusal by naming the reading it declined to
        # emit, which is what tells a reader why a load stopped.
        with pytest.raises(ValueError, match="collision bound for geometry the scene does not declare"):
            load_mjcf_scene_objects(_write(tmp_path, _scene(leg_open='<body name="t_leg_body" pos="0 0 nan">')))


class TestWhatIsDeliberatelyUnchanged:
    """A finite fixture, and an unreadable attribute, behave exactly as before."""

    def test_a_healthy_fixture_reports_exactly_what_it_did(self, tmp_path: Path) -> None:
        assert _size(tmp_path, _scene()) == pytest.approx((0.8, 0.8, 0.76))

    def test_an_unparseable_placement_still_falls_back_to_its_default(self, tmp_path: Path) -> None:
        # ``_parse_xyz`` returns its documented default for an attribute it
        # cannot read. Only a value that PARSED is graded, so this is unchanged:
        # the leg sits at the body origin rather than the scene being refused.
        size = _size(tmp_path, _scene(leg_open='<body name="t_leg_body" pos="not a vector">'))
        assert all(map(math.isfinite, size))

    @pytest.mark.parametrize("value", ["1e308", "-1e308", "0", "1e-300"])
    def test_an_extreme_but_finite_placement_is_accepted(self, tmp_path: Path, value: str) -> None:
        size = _size(tmp_path, _scene(leg_open=f'<body name="t_leg_body" pos="0 0 {value}">'))
        assert all(map(math.isfinite, size))

    def test_the_helper_is_silent_on_finite_input(self) -> None:
        element = ast_element('<body name="t_leg_body"/>')
        _refuse_non_finite_body(element, "pos", (0.0, -1.5, 2.0))
        _refuse_non_finite_body(element, "pos", ())
        _refuse_non_finite_placement("body 't'", "pos", (1.0, 2.0, 3.0))


class TestTheAccumulatorsGuardCoversTheMeshWalksBodyOffsets:
    """Two walks compose the same *body* offsets, so one guard answers for both.

    ``_find_body_mesh`` composes a nested body's ``pos`` into the mesh offset
    exactly as ``_recursive_collision_aabb`` composes it into the bound. Both
    descend the same ``findall("body")`` traversal from the same root, so every
    body the mesh walk reads is a body the guarded accumulator also reads - a
    second guard for *that* quantity would be unreachable, and it would mask
    the one whose harm this file measures.

    The reason reaches the body offsets and stops there. ``_find_body_mesh``
    also reads each mesh geom's *own* ``pos`` and orientation, which the
    accumulator never parses, so those carry a guard of their own -
    ``tests/simulation/isaac/test_non_finite_visual_mesh_placement_is_refused.py``
    measures why the geom guard in ``_geom_aabb`` does not reach them.
    """

    def test_both_walks_descend_the_same_traversal(self) -> None:
        from strands_robots.simulation.isaac import loaders

        for name in ("_recursive_collision_aabb", "_find_body_mesh"):
            source = inspect.getsource(getattr(loaders, name)).strip()
            loops = [ast.unparse(node.iter) for node in ast.walk(ast.parse(source)) if isinstance(node, ast.For)]
            # ``ast.unparse`` normalizes string quotes, so match its spelling.
            assert "body_el.findall('body')" in loops, (name, loops)

    @staticmethod
    def _guards_called_by(function: object) -> set[str]:
        """The finiteness guards a function *calls* - the docstring names them too."""
        tree = ast.parse(inspect.getsource(function).strip())  # type: ignore[arg-type]
        return {
            node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

    def test_the_mesh_walk_does_not_re_guard_the_body_offset(self) -> None:
        """The quantity this file guards is tested once, in the accumulator."""
        from strands_robots.simulation.isaac import loaders

        assert "_refuse_non_finite_body" not in self._guards_called_by(loaders._find_body_mesh)
        assert "_refuse_non_finite_body" in self._guards_called_by(loaders._recursive_collision_aabb)

    def test_the_mesh_walk_does_guard_the_geom_placement_the_accumulator_never_sees(self) -> None:
        """A quantity only this reader parses is refused by this reader."""
        from strands_robots.simulation.isaac import loaders

        assert "_refuse_non_finite_geom" in self._guards_called_by(loaders._find_body_mesh)
        accumulator = inspect.getsource(loaders._recursive_collision_aabb)
        assert "_parse_orientation" not in accumulator


def ast_element(xml: str):
    """Parse a single XML element, for driving the helper directly.

    Args:
        xml: A one-element document.

    Returns:
        The parsed element.
    """
    import xml.etree.ElementTree as ET

    return ET.fromstring(xml)
