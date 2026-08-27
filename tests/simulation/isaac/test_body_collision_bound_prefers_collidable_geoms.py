"""A body's collision bound spans the geoms MuJoCo can actually collide.

``_body_collision_aabb`` computes the axis-aligned box that stands in for a
body's physics footprint, and it is what ``load_mjcf_scene_objects`` reports as
a :class:`~strands_robots.simulation.isaac.loaders.SceneObject`'s ``size`` and
``position``. It selected geoms by ``group == "0"``, falling back to every geom
when that matched nothing, which is wrong in two independent ways.

First, ``group`` carries no contact meaning in MuJoCo -- it is a visualiser
toggle -- and the conventions built on it disagree: MuJoCo Menagerie marks
*collision* geoms with ``group="3"`` while robosuite marks *visual* ones with
``group="1"``. So the attribute cannot say which geoms the solver will touch,
which is the only question this function asks.

Second, the comparison was against the literal string. A geom that omits
``group`` *is* group 0 -- MuJoCo resolves it to ``geom_group == 0`` -- but the
attribute reads ``None``, so the ``"0"`` pass skipped it. In a Menagerie-shaped
body every geom falls through to the "every geom" fallback and the bound becomes
the union of the decorative shell and the collision primitive.

Both directions are measurable. A body carrying a 1.0 m visual shell around a
0.2 m collision box reported a ``(1.0, 1.0, 1.2)`` proxy centred 0.5 m away from
the primitive it stands for; the shipped ``franka_emika_panda`` ``mjx_hand``
fingers reported ``(0.022, 0.026, 0.0532)`` where MuJoCo's collision geometry is
``(0.0175, 0.0152, 0.0165)`` -- 3.2x too long along the finger. And where the
``"0"`` pass *did* match, it narrowed instead: a body whose two collidable geoms
spell ``group="0"`` and ``group="3"`` dropped the second one entirely, reporting
a bound that excludes geometry the solver collides.

The signal now read is the format's own, through :func:`_geom_cannot_collide` --
already the tie-break :func:`_mesh_geom_visual_rank` ranks first for the mirror
question of which mesh is a body's *visual* asset. MuJoCo lets two geoms touch
only when ``contype1 & conaffinity2`` or ``contype2 & conaffinity1`` is
non-zero, so a geom declaring both as ``0`` can never take part in a contact.

MuJoCo is the oracle throughout: every expected bound below is read back off a
compiled ``MjModel`` rather than restated, including the ``geom_group``
resolution that establishes what the old comparison missed.

Deliberately unchanged: a body whose every analytic geom is contact-free. There
is nothing collidable to prefer, so all of them are bounded exactly as before --
an approximate proxy is still better than none. That is pinned below, as is the
single-geom case, so a body the new signal says nothing new about keeps the
answer it has always had.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from strands_robots.simulation.isaac.loaders import (
    _body_collision_aabb,
    _class_attrs,
    _geom_cannot_collide,
    _mjcf_class_defaults,
    load_mjcf_scene_objects,
)

mujoco = pytest.importorskip("mujoco")

#: The Menagerie spelling in shape: the visual class marks itself only by being
#: unable to collide and names no ``group`` at all, and the collision class names
#: ``group="3"``. Under the old rule neither geom matched ``group == "0"``.
MENAGERIE_DEFAULTS = """<default>
    <default class="visual"><geom contype="0" conaffinity="0"/></default>
    <default class="collision"><geom group="3"/></default>
  </default>"""

#: A 1.0 m decorative shell around a 0.2 m collision box held 0.6 m up the z
#: axis, so an over-bound differs from the truth in both size and centre.
SHELL_AROUND_PRIMITIVE = """<body name="link">
    <geom class="visual" type="box" size="0.5 0.5 0.5"/>
    <geom class="collision" type="box" size="0.1 0.1 0.1" pos="0 0 0.6"/>
  </body>"""

#: Two geoms MuJoCo can both collide, distinguished only by ``group``. The old
#: ``"0"`` pass matched the first and dropped the second.
TWO_COLLIDABLE_GROUPS = """<body name="link">
    <geom name="near" type="box" size="0.1 0.1 0.1" group="0"/>
    <geom name="far" type="box" size="0.1 0.1 0.1" group="3" pos="0 0 0.4"/>
  </body>"""

#: ``contype="0"`` alone does not make a geom decorative: MuJoCo still lets it
#: touch anything declaring ``contype`` non-zero, because ``contype_other &
#: conaffinity_self`` is then non-zero. So the wide geom here is collidable and
#: the bound must span it -- a rule reading either half of the pair on its own
#: would drop it.
HALF_DECLARED_CONTACT = """<body name="link">
    <geom name="wide" type="box" size="0.5 0.5 0.5" contype="0" conaffinity="1"/>
    <geom name="small" type="box" size="0.1 0.1 0.1"/>
  </body>"""

#: Nothing here can collide, so there is no preference to express.
ONLY_CONTACT_FREE = """<body name="link">
    <geom class="visual" type="box" size="0.5 0.5 0.5"/>
    <geom class="visual" type="sphere" size="0.2" pos="0 0 0.8"/>
  </body>"""

#: One collidable geom and nothing else: both rules bound the same thing.
SINGLE_GEOM = """<body name="link">
    <geom type="box" size="0.3 0.2 0.1" pos="0.05 0 0"/>
  </body>"""

#: A nested collision child, the ``living_room_table`` -> ``..._col`` shape the
#: recursive walk exists for, so the preference is exercised through it too.
NESTED_COLLISION_CHILD = """<body name="link">
    <geom class="visual" type="box" size="0.5 0.5 0.5"/>
    <body name="link_col" pos="0 0 0.6">
      <geom class="visual" type="box" size="0.4 0.4 0.4"/>
      <geom class="collision" type="box" size="0.1 0.1 0.1"/>
    </body>
  </body>"""

#: Half-extent of each analytic geom type, in MuJoCo's ``geom_size`` terms.
_HALF_EXTENT = {
    int(mujoco.mjtGeom.mjGEOM_BOX): lambda s: (s[0], s[1], s[2]),
    int(mujoco.mjtGeom.mjGEOM_SPHERE): lambda s: (s[0], s[0], s[0]),
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): lambda s: (s[0], s[0], s[1]),
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): lambda s: (s[0], s[0], s[1] + s[0]),
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): lambda s: (s[0], s[1], s[2]),
}


def _model(body: str, defaults: str = MENAGERIE_DEFAULTS) -> str:
    return f"<mujoco>{defaults}<worldbody>{body}</worldbody></mujoco>"


def _compile(body: str, defaults: str = MENAGERIE_DEFAULTS):
    return mujoco.MjModel.from_xml_string(_model(body, defaults))


def _bound(body: str, defaults: str = MENAGERIE_DEFAULTS):
    """The loader's ``(centre, size)`` for the model's single named body."""
    root = ET.fromstring(_model(body, defaults))
    geom_defaults = _mjcf_class_defaults(root, ".", "geom")
    body_el = root.find(".//body")
    assert body_el is not None
    return _body_collision_aabb(body_el, geom_defaults, body_el.get("childclass") or "")


def _mujoco_bound(body: str, defaults: str = MENAGERIE_DEFAULTS, name: str = "link"):
    """MuJoCo's own ``(centre, size)`` over a body's collidable analytic geoms.

    Read off a compiled model, so ``<default>`` inheritance and the contact
    declaration are resolved by MuJoCo rather than by the code under test.
    """
    model = _compile(body, defaults)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert bid != -1, name
    mins = [float("inf")] * 3
    maxs = [float("-inf")] * 3
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != bid:
            continue
        half_of = _HALF_EXTENT.get(int(model.geom_type[gid]))
        if half_of is None:
            continue
        if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
            continue
        half = half_of(model.geom_size[gid])
        pos = model.geom_pos[gid]
        for axis in range(3):
            mins[axis] = min(mins[axis], pos[axis] - half[axis])
            maxs[axis] = max(maxs[axis], pos[axis] + half[axis])
    assert mins[0] != float("inf"), "no collidable analytic geom to compare against"
    return (
        tuple((mins[a] + maxs[a]) / 2.0 for a in range(3)),
        tuple(maxs[a] - mins[a] for a in range(3)),
    )


def _geom_signals(body: str, defaults: str = MENAGERIE_DEFAULTS) -> dict[str, dict[str, object]]:
    """Per-geom ``group``/contact facts as MuJoCo and as the loader see them."""
    model = _compile(body, defaults)
    root = ET.fromstring(_model(body, defaults))
    geom_defaults = _mjcf_class_defaults(root, ".", "geom")
    body_el = root.find(".//body")
    assert body_el is not None
    childclass = body_el.get("childclass") or ""
    out: dict[str, dict[str, object]] = {}
    for index, geom_el in enumerate(body_el.findall("geom")):
        attrs = _class_attrs(geom_el, geom_defaults, childclass)
        out[geom_el.get("name") or geom_el.get("class") or str(index)] = {
            "mujoco_group": int(model.geom_group[index]),
            "loader_group_attr": attrs.get("group"),
            "mujoco_contact_free": bool(model.geom_contype[index] == 0 and model.geom_conaffinity[index] == 0),
            "loader_cannot_collide": _geom_cannot_collide(attrs),
        }
    return out


class TestMuJoCoFixesWhichGeomsCollide:
    """The premises: what the format says, read off a compiled model."""

    def test_a_geom_that_omits_group_is_group_zero_but_reads_as_absent(self) -> None:
        """The blind spot in the old comparison, stated as MuJoCo resolves it."""
        signals = _geom_signals(SHELL_AROUND_PRIMITIVE)["visual"]
        assert signals["mujoco_group"] == 0
        assert signals["loader_group_attr"] is None

    def test_the_visual_class_makes_its_geom_unable_to_collide(self) -> None:
        signals = _geom_signals(SHELL_AROUND_PRIMITIVE)["visual"]
        assert signals["mujoco_contact_free"] is True
        assert signals["loader_cannot_collide"] is True

    def test_the_collision_class_geom_can_collide(self) -> None:
        signals = _geom_signals(SHELL_AROUND_PRIMITIVE)["collision"]
        assert signals["mujoco_contact_free"] is False
        assert signals["loader_cannot_collide"] is False

    def test_two_geoms_differing_only_by_group_can_both_collide(self) -> None:
        """So the bound must span both -- ``group`` does not exclude either."""
        signals = _geom_signals(TWO_COLLIDABLE_GROUPS)
        assert signals["near"]["mujoco_contact_free"] is False
        assert signals["far"]["mujoco_contact_free"] is False
        assert (signals["near"]["mujoco_group"], signals["far"]["mujoco_group"]) == (0, 3)


class TestABodyBoundsTheGeomsThatCanCollide:
    """The bound is over the collidable geoms, and it is MuJoCo's."""

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param(SHELL_AROUND_PRIMITIVE, id="shell-around-primitive"),
            pytest.param(TWO_COLLIDABLE_GROUPS, id="two-collidable-groups"),
            pytest.param(HALF_DECLARED_CONTACT, id="half-declared-contact"),
            pytest.param(SINGLE_GEOM, id="single-geom"),
        ],
    )
    def test_the_bound_is_the_one_mujoco_computes(self, body: str) -> None:
        bound = _bound(body)
        assert bound is not None
        centre, size = bound
        expected_centre, expected_size = _mujoco_bound(body)
        assert centre == pytest.approx(expected_centre)
        assert size == pytest.approx(expected_size)

    def test_a_decorative_shell_does_not_widen_the_bound(self) -> None:
        """The over-bound direction: the shell is 5x the primitive per axis."""
        bound = _bound(SHELL_AROUND_PRIMITIVE)
        assert bound is not None
        _, size = bound
        assert size == pytest.approx((0.2, 0.2, 0.2))

    def test_a_decorative_shell_does_not_move_the_bound(self) -> None:
        """A proxy centred on the union sits 0.5 m off the geometry it stands for."""
        bound = _bound(SHELL_AROUND_PRIMITIVE)
        assert bound is not None
        centre, _ = bound
        assert centre == pytest.approx((0.0, 0.0, 0.6))

    def test_group_does_not_narrow_the_bound(self) -> None:
        """The under-bound direction: both collidable geoms are inside it."""
        bound = _bound(TWO_COLLIDABLE_GROUPS)
        assert bound is not None
        centre, size = bound
        assert (centre, size) == (pytest.approx((0.0, 0.0, 0.2)), pytest.approx((0.2, 0.2, 0.6)))

    def test_a_nested_collision_child_is_preferred_too(self) -> None:
        """The recursive walk asks the same question of each nested body."""
        root = ET.fromstring(_model(NESTED_COLLISION_CHILD))
        geom_defaults = _mjcf_class_defaults(root, ".", "geom")
        child = root.find(".//body/body")
        assert child is not None
        bound = _body_collision_aabb(child, geom_defaults, "")
        assert bound is not None
        assert bound[1] == pytest.approx((0.2, 0.2, 0.2))


class TestASceneObjectCarriesTheCollisionBound:
    """The public reader reports the proxy, so the fix is observable there."""

    def test_the_proxy_size_and_position_are_the_collision_geometry_s(self, tmp_path) -> None:
        scene = tmp_path / "scene.xml"
        scene.write_text(
            _model(
                """<body name="mug" pos="1 2 0">
    <freejoint/>
    <geom class="visual" type="box" size="0.5 0.5 0.5"/>
    <geom class="collision" type="box" size="0.1 0.1 0.1" pos="0 0 0.6"/>
  </body>"""
            ),
            encoding="utf-8",
        )
        objects = {obj.name: obj for obj in load_mjcf_scene_objects(str(scene))}
        assert "mug" in objects
        mug = objects["mug"]
        assert mug.size == pytest.approx((0.2, 0.2, 0.2))
        assert mug.position == pytest.approx((1.0, 2.0, 0.6))


class TestOnlyAGeomThatCannotCollideIsExcluded:
    """The exclusion must not widen: both halves of the pair have to be zero."""

    def test_one_zero_half_of_the_contact_pair_is_not_decorative(self) -> None:
        """MuJoCo still collides such a geom, so it stays inside the bound."""
        bound = _bound(HALF_DECLARED_CONTACT)
        assert bound is not None
        assert bound[1] == pytest.approx((1.0, 1.0, 1.0))

    def test_mujoco_agrees_the_half_declared_geom_can_collide(self) -> None:
        signals = _geom_signals(HALF_DECLARED_CONTACT)["wide"]
        assert signals["mujoco_contact_free"] is False
        assert signals["loader_cannot_collide"] is False


class TestABodyWithNothingCollidableKeepsItsBound:
    """No collidable geom means no preference to express: bound them all."""

    def test_every_contact_free_geom_still_contributes(self) -> None:
        bound = _bound(ONLY_CONTACT_FREE)
        assert bound is not None
        centre, size = bound
        assert centre == pytest.approx((0.0, 0.0, 0.25))
        assert size == pytest.approx((1.0, 1.0, 1.5))

    def test_a_mesh_only_body_still_reports_nothing(self) -> None:
        """No analytic geom at all keeps the ``None`` the caller falls back on."""
        assert _bound("""<body name="link"><geom type="mesh" mesh="m"/></body>""") is None
