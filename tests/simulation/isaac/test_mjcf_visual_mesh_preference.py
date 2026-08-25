"""A body's reported mesh is its visual asset, not its collision hull.

MJCF bodies routinely carry two mesh geoms: the visual asset a renderer should
show, and a convex hull the solver should collide. ``_find_body_mesh`` picks the
one :class:`~strands_robots.simulation.isaac.loaders.SceneObject` carries, and it
picked "the first geom whose ``group`` is not ``"0"``" - reading a non-default
group as the visual marking.

That is backwards for the dominant convention. MuJoCo Menagerie marks a *visual*
geom with ``contype="0" conaffinity="0"`` and a *collision* geom with
``group="3"``, and the collision geom is routinely declared first, so the rule
returned the hull. In the shipped ``shadow_dexee`` hand the visual class carries
no ``group`` at all -- ``<default class="finger/visual"><geom contype="0"
conaffinity="0"/>`` -- so each finger skipped its eight visual mesh geoms for the
ninth, ``r3_finger_base_col``. ``load_mjcf_scene_objects`` reported those fingers
as ``r3_finger_base_col.stl`` under a successful load, while ``hand_base`` in the
same file reported its visual STL: one reader, one file, two answers.

The signal it now ranks first is the format's own. MuJoCo lets two geoms touch
only when ``contype1 & conaffinity2`` or ``contype2 & conaffinity1`` is
non-zero, so a geom declaring both as ``0`` cannot collide with anything and
exists purely to be looked at. ``group`` carries no meaning in MuJoCo itself --
it is a visualiser toggle -- and the conventions built on it disagree, robosuite
emitting visual geoms as ``group="1"`` where Menagerie spells collision as
``group="3"``. So a non-default group is kept only as the weaker hint, taken
when the contact declaration says nothing.

Every premise below is derived from ``mujoco.MjModel`` rather than restated: it
is MuJoCo that fixes which geoms can touch, and MuJoCo that resolves a nested
``<default>`` class onto the geom before either signal is read.

Deliberately unchanged: a subtree where no geom is contact-free. There the
weaker ``group`` hint still decides, and where nothing carries either signal the
first mesh in document order is still reported. Both are pinned below, so a body
the new signal says nothing about keeps the answer it has always had.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from strands_robots.simulation.isaac.loaders import (
    _MESH_VISUAL_RANK_NON_COLLIDING,
    _MESH_VISUAL_RANK_NON_DEFAULT_GROUP,
    _MESH_VISUAL_RANK_UNMARKED,
    _class_attrs,
    _find_body_mesh,
    _geom_cannot_collide,
    _mesh_geom_visual_rank,
    _mjcf_class_defaults,
    _parse_mjcf_mesh_assets,
)

mujoco = pytest.importorskip("mujoco")

#: The Menagerie spelling, verbatim in shape: the visual class marks itself only
#: by being unable to collide and names no ``group``, and the collision class
#: names ``group="3"`` through a nested class the geom actually cites. A reader
#: that took the group as the visual signal preferred ``hull``.
MENAGERIE_DEFAULTS = (
    "<default>"
    '<default class="visual"><geom type="mesh" contype="0" conaffinity="0"/></default>'
    '<default class="collision"><geom type="mesh" group="3"/>'
    '<default class="collision_hard"><geom condim="4"/></default>'
    "</default>"
    "</default>"
)

#: The two geoms of such a body, by the class each cites.
VISUAL_GEOM = '<geom class="visual" mesh="skin" pos="0.1 0 0"/>'
COLLISION_GEOM = '<geom class="collision_hard" mesh="hull" pos="0.5 0 0"/>'

#: Both document orders. Which asset a body describes cannot depend on the order
#: its two geoms happen to be written in, and the collision-first order is the
#: one the shipped assets use.
DOCUMENT_ORDERS = (
    pytest.param(COLLISION_GEOM + VISUAL_GEOM, id="collision-declared-first"),
    pytest.param(VISUAL_GEOM + COLLISION_GEOM, id="visual-declared-first"),
)


def _model(body: str, defaults: str = MENAGERIE_DEFAULTS) -> str:
    """An MJCF whose worldbody is ``body``, with mesh assets MuJoCo can compile."""
    return (
        "<mujoco>"
        f"{defaults}"
        "<asset>"
        '<mesh name="skin" vertex="0 0 0  0.4 0 0  0 0.3 0  0 0 0.2"/>'
        '<mesh name="hull" vertex="0 0 0  0.9 0 0  0 0.8 0  0 0 0.7"/>'
        "</asset>"
        f"<worldbody>{body}</worldbody>"
        "</mujoco>"
    )


def _read(body: str, defaults: str = MENAGERIE_DEFAULTS) -> tuple[str, tuple[float, ...], tuple[float, ...], int]:
    """What the reader reports for the first body of ``_model(body)``."""
    root = ET.fromstring(_model(body, defaults))
    found = _find_body_mesh(next(iter(root.iter("body"))), _mjcf_class_defaults(root, ".", "geom"), "")
    assert found is not None, "premise: the fixture body declares a mesh geom"
    return found


def _mujoco_contact_pairs(body: str, defaults: str = MENAGERIE_DEFAULTS) -> dict[str, tuple[int, int]]:
    """Each named geom's compiled ``(contype, conaffinity)``, as MuJoCo resolves them."""
    model = mujoco.MjModel.from_xml_string(_model(body, defaults))
    return {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i): (
            int(model.geom_contype[i]),
            int(model.geom_conaffinity[i]),
        )
        for i in range(model.ngeom)
    }


class TestMuJoCoFixesWhichGeomsCanTouch:
    """The premises the ranking rests on, taken from the compiler."""

    def test_the_visual_class_compiles_to_a_geom_that_cannot_collide(self):
        # Both halves matter: contact-free for the visual geom is the signal, and
        # the collision geom being able to touch is what makes it a real hull
        # rather than a second decoration.
        pairs = _mujoco_contact_pairs(
            f'<body name="b">{COLLISION_GEOM.replace("mesh=", "name='hull_g' mesh=")}'
            f"{VISUAL_GEOM.replace('mesh=', "name='skin_g' mesh=")}</body>"
        )
        assert pairs["skin_g"] == (0, 0)
        assert pairs["hull_g"] != (0, 0)

    def test_a_non_default_group_says_nothing_about_whether_a_geom_collides(self):
        # This is why the group cannot be the visual signal: MuJoCo gives a
        # group="3" geom the ordinary contact defaults.
        pairs = _mujoco_contact_pairs('<body name="b"><geom name="g" type="mesh" mesh="hull" group="3"/></body>')
        assert pairs["g"] != (0, 0)

    def test_the_group_is_resolved_through_the_nested_default_class(self):
        # The collision geom cites "collision_hard" and inherits group="3" from
        # its parent class, so the fixture exercises class resolution rather
        # than a group spelled on the geom.
        root = ET.fromstring(_model(f'<body name="b">{COLLISION_GEOM}</body>'))
        # The body's own geom, not the one inside the <default> block.
        geom = next(iter(root.iter("body"))).findall("geom")[0]
        assert _class_attrs(geom, _mjcf_class_defaults(root, ".", "geom"), "").get("group") == "3"


class TestABodyReportsItsVisualMesh:
    """The regression: the reported asset is the one a renderer should show."""

    @pytest.mark.parametrize("body_geoms", DOCUMENT_ORDERS)
    def test_the_visual_mesh_is_reported_whichever_geom_is_declared_first(self, body_geoms):
        name, pos, _quat, rank = _read(f'<body name="b">{body_geoms}</body>')
        assert name == "skin"
        assert rank == _MESH_VISUAL_RANK_NON_COLLIDING
        # The pose travels with the pick, so reporting the right name at the
        # wrong geom's offset would still be the wrong asset.
        assert pos == pytest.approx((0.1, 0.0, 0.0))

    def test_a_visual_mesh_in_a_nested_body_beats_a_collision_mesh_on_the_body(self):
        # shadow_dexee's shape: the addressed body carries the hull and the
        # visual asset sits one body down.
        name, pos, _quat, rank = _read(
            f'<body name="b">{COLLISION_GEOM}<body name="inner" pos="0 0.25 0">{VISUAL_GEOM}</body></body>'
        )
        assert name == "skin"
        assert rank == _MESH_VISUAL_RANK_NON_COLLIDING
        # The nested body's own offset is folded into the reported position.
        assert pos == pytest.approx((0.1, 0.25, 0.0))

    def test_a_deeper_visual_mesh_beats_a_nearer_collision_mesh(self):
        # The ranking is over the whole subtree, not one level of it, and the
        # offsets of every body on the way accumulate.
        name, pos, _quat, _rank = _read(
            '<body name="b">'
            f'<body name="mid" pos="0 0 0.1">{COLLISION_GEOM}'
            f'<body name="deep" pos="0 0 0.2">{VISUAL_GEOM}</body></body></body>'
        )
        assert name == "skin"
        assert pos == pytest.approx((0.1, 0.0, 0.3))


class TestTheWeakerHintStillDecidesWhereItIsTheOnlySignal:
    """Over-reach controls: bodies the contact declaration says nothing about.

    Every expectation here is a mesh *name*, and every one of them is the name
    the old rule reported too. Preferring the contact-free geom must not change
    a single body the contact declaration is silent about.
    """

    def test_the_collision_hull_is_reported_when_it_is_the_only_mesh(self):
        # A body with no visual asset still describes itself: refusing to report
        # the hull would lose the only geometry there is.
        name, _pos, _quat, _rank = _read(f'<body name="b">{COLLISION_GEOM}</body>')
        assert name == "hull"

    def test_a_non_default_group_still_wins_when_no_geom_is_contact_free(self):
        # google_barkour_v0's shape: one set of meshes serves both roles, so the
        # group is the only marking in the file. Preferring the contact-free geom
        # must not cost these bodies the answer they have.
        name, _pos, _quat, rank = _read(
            '<body name="b"><geom type="mesh" mesh="hull"/><geom type="mesh" mesh="skin" group="1"/></body>',
            defaults="",
        )
        assert name == "skin"

    def test_an_unmarked_subtree_still_reports_its_first_mesh_in_document_order(self):
        name, _pos, _quat, _rank = _read(
            '<body name="b"><geom type="mesh" mesh="hull"/><geom type="mesh" mesh="skin"/></body>',
            defaults="",
        )
        assert name == "hull"

    def test_a_nearer_group_marked_mesh_beats_an_unmarked_one_in_a_nested_body(self):
        name, _pos, _quat, _rank = _read(
            '<body name="b"><geom type="mesh" mesh="skin" group="1"/>'
            '<body name="inner"><geom type="mesh" mesh="hull"/></body></body>',
            defaults="",
        )
        assert name == "skin"

    def test_an_unmarked_mesh_on_the_body_loses_to_a_group_marked_one_below_it(self):
        # Rank, not depth: a stronger candidate anywhere in the subtree wins.
        name, _pos, _quat, _rank = _read(
            '<body name="b"><geom type="mesh" mesh="hull"/>'
            '<body name="inner"><geom type="mesh" mesh="skin" group="1"/></body></body>',
            defaults="",
        )
        assert name == "skin"

    def test_a_subtree_with_no_mesh_geom_reports_nothing(self):
        root = ET.fromstring(_model('<body name="b"><geom type="sphere" size="0.1"/></body>'))
        assert _find_body_mesh(next(iter(root.iter("body"))), _mjcf_class_defaults(root, ".", "geom"), "") is None


class TestWhenAGeomCannotCollide:
    """``_geom_cannot_collide``'s domain: both halves of the pair, or neither."""

    @pytest.mark.parametrize(
        ("attrs", "expected"),
        [
            pytest.param({"contype": "0", "conaffinity": "0"}, True, id="both-zero"),
            pytest.param({"contype": "0"}, False, id="contype-only-conaffinity-defaults-to-1"),
            pytest.param({"conaffinity": "0"}, False, id="conaffinity-only"),
            pytest.param({}, False, id="neither-declared"),
            pytest.param({"contype": "1", "conaffinity": "0"}, False, id="one-nonzero"),
            pytest.param({"contype": "0", "conaffinity": "nope"}, False, id="unparseable"),
            pytest.param({"contype": "0", "conaffinity": "0.0"}, False, id="not-an-integer"),
        ],
    )
    def test_the_pair_decides(self, attrs, expected):
        assert _geom_cannot_collide(attrs) is expected

    def test_mujoco_agrees_a_single_zero_still_leaves_a_geom_able_to_touch(self):
        # The asymmetry above is MuJoCo's, not this reader's: contype=0 alone
        # leaves conaffinity at its default, so the geom can still be collided
        # with by a geom whose contype overlaps it.
        pairs = _mujoco_contact_pairs('<body name="b"><geom name="g" type="mesh" mesh="hull" contype="0"/></body>')
        assert pairs["g"] != (0, 0)

    def test_the_three_ranks_are_all_reachable(self):
        # Without this a ranking that collapsed to one value would still satisfy
        # every ordering expectation above by accident.
        assert {
            _mesh_geom_visual_rank({"contype": "0", "conaffinity": "0"}),
            _mesh_geom_visual_rank({"group": "3"}),
            _mesh_geom_visual_rank({}),
        } == {
            _MESH_VISUAL_RANK_NON_COLLIDING,
            _MESH_VISUAL_RANK_NON_DEFAULT_GROUP,
            _MESH_VISUAL_RANK_UNMARKED,
        }

    @pytest.mark.parametrize(
        ("attrs", "expected"),
        [
            pytest.param({"contype": "0", "conaffinity": "0"}, _MESH_VISUAL_RANK_NON_COLLIDING, id="contact-free"),
            pytest.param({"group": "3"}, _MESH_VISUAL_RANK_NON_DEFAULT_GROUP, id="menagerie-collision-group"),
            pytest.param({"group": "1"}, _MESH_VISUAL_RANK_NON_DEFAULT_GROUP, id="robosuite-visual-group"),
            pytest.param({}, _MESH_VISUAL_RANK_UNMARKED, id="unmarked"),
            pytest.param({"group": "0"}, _MESH_VISUAL_RANK_UNMARKED, id="default-group-spelled-out"),
        ],
    )
    def test_a_declaration_ranks_where_it_says_it_does(self, attrs, expected):
        assert _mesh_geom_visual_rank(attrs) == expected

    def test_a_contact_free_geom_outranks_a_group_marked_one(self):
        assert _mesh_geom_visual_rank({"contype": "0", "conaffinity": "0", "group": "3"}) < _mesh_geom_visual_rank(
            {"group": "1"}
        )


class TestAMeshAssetThatNamesNoFile:
    """A ``<mesh>`` with no ``file`` contributes no path to resolve."""

    def test_it_is_skipped_rather_than_registered_under_an_empty_path(self):
        # MJCF lets a mesh carry inline vertices instead of a file, which every
        # fixture above relies on; such an asset has no path for the scene
        # loader to hand a renderer.
        root = ET.fromstring(_model('<body name="b"><geom type="mesh" mesh="skin"/></body>'))
        assert _parse_mjcf_mesh_assets(root, ".") == {}

    def test_a_file_backed_asset_alongside_it_is_still_registered(self):
        # Non-vacuity: the empty map above is the skip, not a reader that
        # registers nothing at all.
        root = ET.fromstring(
            "<mujoco><asset>"
            '<mesh name="inline" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>'
            '<mesh name="ondisk" file="widget.obj"/>'
            "</asset><worldbody/></mujoco>"
        )
        assert set(_parse_mjcf_mesh_assets(root, "/assets")) == {"ondisk"}
