"""A geom's own rotation is part of the collision bound it contributes.

``_geom_aabb`` reports the axis-aligned box one ``<geom>`` occupies in its
owning body's frame. ``_body_collision_aabb`` unions those, and
``load_mjcf_scene_objects`` publishes the union as a
:class:`~strands_robots.simulation.isaac.loaders.SceneObject`'s ``size`` and
``position`` -- the physics footprint that stands in for the object. It read
``pos`` and ``size`` and never asked how the geom is turned.

MJCF gives a geom five mutually exclusive ways to state a rotation (``quat``,
``euler``, ``axisangle``, ``xyaxes``, ``zaxis``), and dropping it does not make
the bound approximate, it reports the extents on the wrong axes. A 0.60 m bar
turned a quarter turn about z came back as ``(0.6, 0.04, 0.04)`` from
``load_mjcf_scene_objects`` while MuJoCo places it along y at
``(0.04, 0.6, 0.04)`` -- 15x too small on the axis the bar occupies and 15x too
large on an axis it does not.

The same function already read a rotation on its other channel. A capsule or
cylinder may spell its placement with ``fromto``, whose endpoints carry the
axis, and ``_segment_aabb`` bounds that exactly. So one geom got two different
bounds depending on which spelling was used for it, though MuJoCo compiles both
to the same shape -- measured below, because it is what makes this a defect
rather than a missing feature.

Over the 336 loadable MJCFs in the shipped asset cache, 421 bodies declare a
rotated collidable analytic geom (574 with ``quat``, 53 with ``euler``), and 27
of the 72 registered robots own at least one -- ``unitree_g1``, ``unitree_go2``,
``unitree_h1``, ``ur5e``, ``ur10e``, ``stretch``, ``shadow_hand`` among them.
Against MuJoCo's own geometry the old rule under-bounded 279 of those bodies and
over-bounded 247; the worst was ``stretch``'s gripper finger, reported 0.0085 m
on the axis it spans 0.1601 m. Afterwards none of the 421 is under- or
over-bounded beyond 0.1%.

MuJoCo is the oracle throughout. Expected extents are read off a compiled
``MjModel`` -- for a box from its eight corners rotated by MuJoCo's own
``mju_rotVecQuat``, which is exact, and for the curved primitives from a
deterministic surface grid, which is why those carry a tolerance. Nothing here
restates the formula under test.

Deliberately unchanged, and pinned below: an unrotated geom of every analytic
type, a sphere (which no rotation moves), the ``fromto`` channel -- MuJoCo
derives that geom's orientation from the endpoints and discards any the element
declares, so applying one there would be wrong -- and the preference for the
geoms MuJoCo can actually collide.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from strands_robots.simulation.isaac.loaders import (
    _body_collision_aabb,
    _geom_aabb,
    _mjcf_angle_units,
    _mjcf_class_defaults,
    _mjcf_model_toplevel,
    load_mjcf_scene_objects,
)

mujoco = pytest.importorskip("mujoco")

#: A 0.60 x 0.04 x 0.04 m bar. Long enough that reporting its extent on the
#: wrong axis is a 15x error rather than a rounding difference.
BAR_HALF = (0.30, 0.02, 0.02)

#: Every MJCF spelling of the same quarter turn about z, so the bound cannot
#: depend on which one a model happens to use.
QUARTER_TURN_ABOUT_Z = {
    "quat": 'quat="0.7071067811865476 0 0 0.7071067811865476"',
    "euler": 'euler="0 0 90"',
    "axisangle": 'axisangle="0 0 1 90"',
    "xyaxes": 'xyaxes="0 1 0 -1 0 0"',
    # ``zaxis`` cannot express a turn about z, so it gets its own tilt below.
}

#: A tilt ``zaxis`` can express: local +z onto a direction with all three
#: components non-zero, so no axis of the answer is zero or shared.
SKEW_ZAXIS = 'zaxis="0.3 0.5 0.81"'

#: A rotation with no special structure: ``zaxis`` builds the *minimal* rotation
#: onto an axis, whose matrix is symmetric enough that a row and the matching
#: column agree, so it cannot see an index slip. Three composed Euler angles can.
GENERAL_ROTATION = 'euler="30 40 50"'

#: Half-extents of MuJoCo's analytic primitives in ``geom_size`` terms, used to
#: decide which compiled geoms the oracle has a surface for.
_ANALYTIC = {
    int(mujoco.mjtGeom.mjGEOM_BOX),
    int(mujoco.mjtGeom.mjGEOM_SPHERE),
    int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    int(mujoco.mjtGeom.mjGEOM_CAPSULE),
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
}

#: Azimuth/elevation resolution of the surface grid for the curved primitives.
#: 180 samples put the worst discretisation gap on a 0.3 m radius near 1e-4 m.
_GRID = 180


def _model(body: str, compiler: str = "", defaults: str = "") -> str:
    return f"<mujoco>{compiler}{defaults}<worldbody>{body}</worldbody></mujoco>"


def _bound(body: str, compiler: str = "", defaults: str = "", name: str = "link"):
    """The loader's ``(centre, size)`` for one named body, units resolved as production does."""
    xml = _model(body, compiler, defaults)
    root = ET.fromstring(xml)
    geom_defaults = _mjcf_class_defaults(root, ".", "geom")
    angle_scale, eulerseq = _mjcf_angle_units(_mjcf_model_toplevel(root, "."))
    body_el = root.find(f".//body[@name='{name}']")
    assert body_el is not None
    return _body_collision_aabb(
        body_el,
        geom_defaults,
        body_el.get("childclass") or "",
        angle_scale=angle_scale,
        eulerseq=eulerseq,
    )


def _surface(gtype: int, size) -> list[tuple[float, float, float]]:
    """Points on one geom's surface in its own local frame.

    A box's eight corners are exact -- its farthest point along any direction is
    always a corner. The curved primitives get a deterministic grid instead, so
    the bounds derived from them are tight rather than exact.
    """
    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        return [(sx * size[0], sy * size[1], sz * size[2]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    steps = [(-0.5 + (i + 0.5) / _GRID) * math.pi for i in range(_GRID)]
    turns = [2.0 * math.pi * i / _GRID for i in range(_GRID)]
    if gtype in (int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)):
        semi = (size[0], size[0], size[0]) if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE) else tuple(size[:3])
        return [
            (semi[0] * math.cos(el) * math.cos(az), semi[1] * math.cos(el) * math.sin(az), semi[2] * math.sin(el))
            for el in steps
            for az in turns
        ]
    radius, half_length = size[0], size[1]
    points = [(radius * math.cos(az), radius * math.sin(az), sz * half_length) for az in turns for sz in (-1, 1)]
    if gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        points += [
            (
                radius * math.cos(el) * math.cos(az),
                radius * math.cos(el) * math.sin(az),
                radius * math.sin(el) + (half_length if el > 0 else -half_length),
            )
            for el in steps
            for az in turns
        ]
    return points


def _mujoco_bound(body: str, compiler: str = "", defaults: str = "", name: str = "link"):
    """MuJoCo's own ``(centre, size)`` over a body's collidable analytic geoms.

    Every geom's placement, rotation and ``<default>`` inheritance is read off
    the compiled model, and each surface point is rotated with MuJoCo's
    ``mju_rotVecQuat`` -- so no arithmetic from the module under test takes part.
    """
    model = mujoco.MjModel.from_xml_string(_model(body, compiler, defaults))
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert bid != -1, name
    mins = [float("inf")] * 3
    maxs = [float("-inf")] * 3
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != bid or int(model.geom_type[gid]) not in _ANALYTIC:
            continue
        if model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0:
            continue
        quat = model.geom_quat[gid]
        pos = model.geom_pos[gid]
        spun = np.zeros(3)
        for point in _surface(int(model.geom_type[gid]), model.geom_size[gid]):
            mujoco.mju_rotVecQuat(spun, np.asarray(point, dtype=float), quat)
            for axis in range(3):
                world = float(spun[axis]) + float(pos[axis])
                mins[axis] = min(mins[axis], world)
                maxs[axis] = max(maxs[axis], world)
    assert maxs[0] > mins[0], "premise: the oracle found no collidable analytic geom"
    return (
        tuple((mins[i] + maxs[i]) / 2.0 for i in range(3)),
        tuple(maxs[i] - mins[i] for i in range(3)),
    )


def _geom_of(body: str, defaults: str = "", index: int = 0):
    """``(element, defaults map)`` for one geom, for the direct-``_geom_aabb`` cells."""
    root = ET.fromstring(_model(body, "", defaults))
    return root.findall(".//geom")[index], _mjcf_class_defaults(root, ".", "geom")


class TestTheOracleAndTheRespellAreSound:
    """Premises. Each of these must hold or the cells below prove nothing."""

    def test_one_capsule_spelled_two_ways_is_one_shape_to_mujoco(self):
        # The respell is what makes the two channels comparable: if MuJoCo saw
        # two different shapes, two different bounds would be correct.
        p1, p2, radius = (0.0, 0.0, -0.1), (0.1, 0.0, 0.1), 0.02
        length = math.dist(p1, p2)
        mid = tuple((p1[i] + p2[i]) / 2.0 for i in range(3))
        unit = tuple((p2[i] - p1[i]) / length for i in range(3))
        segment = mujoco.MjModel.from_xml_string(
            _model(
                f'<body name="link"><geom type="capsule" fromto="{p1[0]} {p1[1]} {p1[2]} '
                f'{p2[0]} {p2[1]} {p2[2]}" size="{radius}"/></body>'
            )
        )
        turned = mujoco.MjModel.from_xml_string(
            _model(
                f'<body name="link"><geom type="capsule" pos="{mid[0]} {mid[1]} {mid[2]}" '
                f'zaxis="{unit[0]} {unit[1]} {unit[2]}" size="{radius} {length / 2}"/></body>'
            )
        )
        assert int(segment.geom_type[0]) == int(turned.geom_type[0])
        assert segment.geom_size[0][:2] == pytest.approx(turned.geom_size[0][:2], abs=1e-12)
        assert list(segment.geom_pos[0]) == pytest.approx(list(turned.geom_pos[0]), abs=1e-12)
        # A capsule is symmetric about its axis, so the two quaternions may
        # differ by a half turn; what must agree is the axis, up to sign.
        axes = []
        for model in (segment, turned):
            spun = np.zeros(3)
            mujoco.mju_rotVecQuat(spun, np.array([0.0, 0.0, 1.0]), model.geom_quat[0])
            axes.append(spun.tolist())
        assert abs(sum(axes[0][i] * axes[1][i] for i in range(3))) == pytest.approx(1.0, abs=1e-12)

    def test_mujoco_discards_an_orientation_declared_beside_fromto(self):
        # This is why the ``fromto`` branch must stay rotation-free: honouring a
        # declared rotation there would disagree with the compiler.
        declared = 'quat="0.7071067811865476 0 0.7071067811865476 0"'
        with_rotation = mujoco.MjModel.from_xml_string(
            _model(f'<body name="link"><geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.02" {declared}/></body>')
        )
        without = mujoco.MjModel.from_xml_string(
            _model('<body name="link"><geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.02"/></body>')
        )
        assert list(with_rotation.geom_quat[0]) == pytest.approx(list(without.geom_quat[0]), abs=1e-12)

    def test_mujoco_refuses_pos_and_fromto_on_one_geom(self):
        # So the two channels are genuinely exclusive and neither can be reached
        # through the other's inputs.
        with pytest.raises(ValueError, match="both pos and fromto"):
            mujoco.MjModel.from_xml_string(
                _model('<body name="link"><geom type="capsule" pos="1 0 0" fromto="0 0 0 0 0 0.5" size="0.02"/></body>')
            )

    def test_the_bar_is_long_enough_for_a_wrong_axis_to_be_obvious(self):
        # A cube would hide every error this file is about.
        assert BAR_HALF[0] > 10 * max(BAR_HALF[1], BAR_HALF[2])


class TestARotatedGeomIsBoundedOnTheAxesItOccupies:
    """The regression. Every expected value comes from MuJoCo, not from the reader."""

    def test_a_quarter_turn_moves_the_long_axis(self):
        body = (
            f'<body name="link"><geom type="box" size="{BAR_HALF[0]} {BAR_HALF[1]} {BAR_HALF[2]}" '
            f"{QUARTER_TURN_ABOUT_Z['euler']}/></body>"
        )
        centre, size = _bound(body)
        assert size == pytest.approx(_mujoco_bound(body)[1], abs=1e-12)
        assert size == pytest.approx((0.04, 0.60, 0.04), abs=1e-12)
        assert centre == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)

    @pytest.mark.parametrize("spelling", sorted(QUARTER_TURN_ABOUT_Z))
    def test_every_spelling_of_one_turn_gives_one_bound(self, spelling):
        body = (
            f'<body name="link"><geom type="box" size="{BAR_HALF[0]} {BAR_HALF[1]} {BAR_HALF[2]}" '
            f"{QUARTER_TURN_ABOUT_Z[spelling]}/></body>"
        )
        assert _bound(body)[1] == pytest.approx(_mujoco_bound(body)[1], abs=1e-12)

    def test_a_tilt_no_axis_of_which_is_zero(self):
        # A turn about a single axis leaves one extent untouched, which a wrong
        # formula can still get right. This tilt moves all three.
        body = f'<body name="link"><geom type="box" size="0.3 0.2 0.1" {SKEW_ZAXIS}/></body>'
        expected = _mujoco_bound(body)[1]
        assert _bound(body)[1] == pytest.approx(expected, abs=1e-12)
        assert min(expected) > 0.0 and len(set(round(v, 9) for v in expected)) == 3

    def test_a_rotation_inherited_from_a_default_class_is_honoured(self):
        # MJCF lets a ``<default>`` supply the orientation, so reading the
        # element alone sees an unrotated geom where MuJoCo sees a turned one.
        defaults = '<default><default class="turned"><geom euler="0 0 90"/></default></default>'
        body = (
            f'<body name="link"><geom class="turned" type="box" '
            f'size="{BAR_HALF[0]} {BAR_HALF[1]} {BAR_HALF[2]}"/></body>'
        )
        assert _bound(body, defaults=defaults)[1] == pytest.approx(_mujoco_bound(body, defaults=defaults)[1], abs=1e-12)

    def test_an_element_spelling_beats_the_classs_quat(self):
        # MJCF keeps the four alternative spellings in a slot separate from
        # ``quat`` and prefers that slot, so the element's ``euler`` wins over a
        # class ``quat``. Resolving that needs the element's own attributes, not
        # only the merged view.
        defaults = (
            '<default><default class="turned">'
            '<geom quat="0.7071067811865476 0 0 0.7071067811865476"/></default></default>'
        )
        body = (
            f'<body name="link"><geom class="turned" type="box" '
            f'size="{BAR_HALF[0]} {BAR_HALF[1]} {BAR_HALF[2]}" euler="0 90 0"/></body>'
        )
        assert _bound(body, defaults=defaults)[1] == pytest.approx(_mujoco_bound(body, defaults=defaults)[1], abs=1e-12)
        # The class quat would have put the bar on y; the element's euler puts it on z.
        assert _bound(body, defaults=defaults)[1] == pytest.approx((0.04, 0.04, 0.60), abs=1e-12)

    def test_the_compiler_angle_unit_is_honoured(self):
        # ``<compiler angle="radian">`` makes ``euler="0 0 1.5707963"`` the same
        # quarter turn that reads as 1.57 degrees under the default.
        compiler = '<compiler angle="radian"/>'
        body = (
            f'<body name="link"><geom type="box" size="{BAR_HALF[0]} {BAR_HALF[1]} {BAR_HALF[2]}" '
            f'euler="0 0 {math.pi / 2}"/></body>'
        )
        assert _bound(body, compiler=compiler)[1] == pytest.approx(_mujoco_bound(body, compiler=compiler)[1], abs=1e-12)
        assert _bound(body, compiler=compiler)[1] == pytest.approx((0.04, 0.60, 0.04), abs=1e-9)

    def test_the_compiler_eulerseq_is_honoured(self):
        # Under ``zyx`` the first angle turns about z; under the default ``xyz``
        # it turns about x, and the two bounds differ.
        compiler = '<compiler eulerseq="zyx"/>'
        body = (
            f'<body name="link"><geom type="box" size="{BAR_HALF[0]} {BAR_HALF[1]} {BAR_HALF[2]}" '
            f'euler="90 0 0"/></body>'
        )
        assert _bound(body, compiler=compiler)[1] == pytest.approx(_mujoco_bound(body, compiler=compiler)[1], abs=1e-12)
        assert _bound(body, compiler=compiler)[1] != pytest.approx(_bound(body)[1], abs=1e-9)

    @pytest.mark.parametrize("rotation", [SKEW_ZAXIS, GENERAL_ROTATION], ids=["zaxis-tilt", "euler-general"])
    @pytest.mark.parametrize(
        ("gtype", "size"),
        [
            ("box", "0.3 0.2 0.1"),
            ("capsule", "0.05 0.25"),
            ("cylinder", "0.05 0.25"),
            ("ellipsoid", "0.3 0.2 0.1"),
        ],
    )
    def test_every_rotatable_primitive_matches_mujoco(self, gtype, size, rotation):
        body = f'<body name="link"><geom type="{gtype}" size="{size}" {rotation}/></body>'
        expected = _mujoco_bound(body)[1]
        # The curved primitives are graded against a surface grid, which can only
        # under-report an extent, so the reader may exceed it by the grid's gap.
        assert _bound(body)[1] == pytest.approx(expected, rel=2e-3, abs=1e-12)
        assert all(got >= want - 1e-12 for got, want in zip(_bound(body)[1], expected, strict=True))

    def test_one_capsule_spelled_two_ways_gets_one_bound(self):
        # The defect in one line: MuJoCo compiles these to the same shape (pinned
        # in the premises above), so the reader owes them the same bound.
        p1, p2, radius = (0.0, 0.0, -0.1), (0.1, 0.0, 0.1), 0.02
        length = math.dist(p1, p2)
        mid = tuple((p1[i] + p2[i]) / 2.0 for i in range(3))
        unit = tuple((p2[i] - p1[i]) / length for i in range(3))
        segment = (
            f'<body name="link"><geom type="capsule" fromto="{p1[0]} {p1[1]} {p1[2]} '
            f'{p2[0]} {p2[1]} {p2[2]}" size="{radius}"/></body>'
        )
        turned = (
            f'<body name="link"><geom type="capsule" pos="{mid[0]} {mid[1]} {mid[2]}" '
            f'zaxis="{unit[0]} {unit[1]} {unit[2]}" size="{radius} {length / 2}"/></body>'
        )
        by_endpoints, by_rotation = _bound(segment), _bound(turned)
        assert by_rotation[0] == pytest.approx(by_endpoints[0], abs=1e-12)
        assert by_rotation[1] == pytest.approx(by_endpoints[1], abs=1e-12)

    def test_the_public_loader_reports_the_size_mujoco_places(self, tmp_path):
        # Through ``load_mjcf_scene_objects``, which is what publishes the proxy.
        scene = (
            '<mujoco model="turned"><worldbody>'
            '<geom name="floor" type="plane" size="2 2 0.1"/>'
            '<body name="ruler" pos="0.4 0 0.1"><freejoint/>'
            f'<geom type="box" size="{BAR_HALF[0]} {BAR_HALF[1]} {BAR_HALF[2]}" euler="0 0 90"/>'
            "</body></worldbody></mujoco>"
        )
        path = tmp_path / "turned.xml"
        path.write_text(scene)
        objects = {obj.name: obj for obj in load_mjcf_scene_objects(str(path))}
        assert set(objects) == {"ruler"}
        assert tuple(objects["ruler"].size) == pytest.approx((0.04, 0.60, 0.04), abs=1e-12)

    def test_the_public_loader_honours_a_radian_model(self, tmp_path):
        # The units live on the model's ``<compiler>`` and the reader defaults to
        # MJCF's own, so a chain that forgets to forward them reads a radian
        # model in degrees and still returns a plausible box.
        scene = (
            '<mujoco model="radians"><compiler angle="radian"/><worldbody>'
            '<geom name="floor" type="plane" size="2 2 0.1"/>'
            '<body name="ruler" pos="0.4 0 0.1"><freejoint/>'
            f'<geom type="box" size="{BAR_HALF[0]} {BAR_HALF[1]} {BAR_HALF[2]}" euler="0 0 {math.pi / 2}"/>'
            "</body></worldbody></mujoco>"
        )
        path = tmp_path / "radians.xml"
        path.write_text(scene)
        objects = {obj.name: obj for obj in load_mjcf_scene_objects(str(path))}
        assert tuple(objects["ruler"].size) == pytest.approx((0.04, 0.60, 0.04), abs=1e-9)

    def test_no_rotation_reports_a_bound_smaller_than_the_geom(self):
        # The unsafe direction, swept: a proxy narrower than the object it stands
        # for lets a policy be evaluated against geometry that is not there.
        size = f"{BAR_HALF[0]} {BAR_HALF[1]} {BAR_HALF[2]}"
        for degrees in range(0, 190, 10):
            for axis in ("1 0 0", "0 1 0", "0 0 1", "1 1 0", "1 2 3"):
                body = f'<body name="link"><geom type="box" size="{size}" axisangle="{axis} {degrees}"/></body>'
                got = _bound(body)[1]
                want = _mujoco_bound(body)[1]
                assert all(g >= w - 1e-12 for g, w in zip(got, want, strict=True)), (degrees, axis, got, want)

    def test_a_rotated_geom_widens_the_union_of_a_multi_geom_body(self):
        # The bound is a union, so a turned geom has to reach the union too - not
        # only be right when it is a body's only geom.
        body = (
            '<body name="link">'
            '<geom name="hub" type="box" size="0.05 0.05 0.05"/>'
            f'<geom name="arm" type="box" size="{BAR_HALF[0]} {BAR_HALF[1]} {BAR_HALF[2]}" euler="0 0 90"/>'
            "</body>"
        )
        assert _bound(body)[1] == pytest.approx(_mujoco_bound(body)[1], abs=1e-12)
        assert _bound(body)[1] == pytest.approx((0.10, 0.60, 0.10), abs=1e-12)


class TestWhatARotationDoesNotChange:
    """Controls. Every expectation here is one the reader already met."""

    @pytest.mark.parametrize(
        ("gtype", "size", "expected"),
        [
            ("box", "0.3 0.2 0.1", (0.6, 0.4, 0.2)),
            ("sphere", "0.25", (0.5, 0.5, 0.5)),
            ("capsule", "0.05 0.25", (0.1, 0.1, 0.6)),
            ("cylinder", "0.05 0.25", (0.1, 0.1, 0.5)),
            ("ellipsoid", "0.3 0.2 0.1", (0.6, 0.4, 0.2)),
        ],
    )
    def test_an_unrotated_primitive_keeps_its_bound(self, gtype, size, expected):
        body = f'<body name="link"><geom type="{gtype}" size="{size}"/></body>'
        assert _bound(body)[1] == pytest.approx(expected, abs=1e-12)

    def test_a_sphere_is_rotation_invariant(self):
        plain = '<body name="link"><geom type="sphere" size="0.25"/></body>'
        spun = f'<body name="link"><geom type="sphere" size="0.25" {SKEW_ZAXIS}/></body>'
        assert _bound(spun)[0] == pytest.approx(_bound(plain)[0], abs=1e-12)
        assert _bound(spun)[1] == pytest.approx(_bound(plain)[1], abs=1e-12)

    def test_the_fromto_channel_ignores_a_declared_rotation(self):
        # MuJoCo does too (pinned in the premises), so honouring one here would
        # bound a shape the compiler does not build.
        plain = '<body name="link"><geom type="capsule" fromto="0 0 -0.2 0 0 0.2" size="0.03"/></body>'
        declared = (
            '<body name="link"><geom type="capsule" fromto="0 0 -0.2 0 0 0.2" size="0.03" euler="0 90 0"/></body>'
        )
        assert _bound(declared)[0] == pytest.approx(_bound(plain)[0], abs=1e-12)
        assert _bound(declared)[1] == pytest.approx(_bound(plain)[1], abs=1e-12)
        assert _bound(plain)[1] == pytest.approx((0.06, 0.06, 0.46), abs=1e-12)

    @pytest.mark.parametrize("gtype", ["mesh", "plane", "hfield", "sdf"])
    def test_a_geom_with_no_analytic_bound_still_has_none(self, gtype):
        extra = ' mesh="m"' if gtype == "mesh" else ""
        geom, defaults = _geom_of(f'<body name="link"><geom type="{gtype}" size="1 1 1"{extra}/></body>')
        assert _geom_aabb(geom, defaults, "") is None

    def test_the_collidable_preference_still_decides(self):
        # A turned decorative shell must not widen the proxy: the rotation is
        # read after the contact declaration has already excluded the geom.
        body = (
            '<body name="link">'
            '<geom type="box" size="0.5 0.1 0.1" contype="0" conaffinity="0" euler="0 0 90"/>'
            '<geom type="box" size="0.1 0.1 0.1"/>'
            "</body>"
        )
        assert _bound(body)[1] == pytest.approx((0.2, 0.2, 0.2), abs=1e-12)


class TestTheCompilerUnitsReachTheReader:
    """A default that is never overridden is a rule nothing applies."""

    def test_the_scene_loader_forwards_the_compiler_units(self):
        # ``_geom_aabb`` defaults to MJCF's own defaults, so a chain that forgets
        # to forward them reads every ``<compiler angle="radian">`` model in
        # degrees and still returns a plausible number.
        import ast
        import inspect

        from strands_robots.simulation.isaac import loaders as mod

        tree = ast.parse(inspect.getsource(mod))
        forwarding = {
            "load_mjcf_scene_objects": "_recursive_collision_aabb",
            "_recursive_collision_aabb": "_body_collision_aabb",
            "_body_collision_aabb": "_geom_aabb",
        }
        seen = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name not in forwarding:
                continue
            callee = forwarding[node.name]
            calls = [
                call
                for call in ast.walk(node)
                if isinstance(call, ast.Call) and getattr(call.func, "id", None) == callee
            ]
            assert calls, f"premise: {node.name} no longer calls {callee}"
            for call in calls:
                keywords = {kw.arg for kw in call.keywords}
                assert {"angle_scale", "eulerseq"} <= keywords, (
                    f"{node.name} calls {callee} without forwarding the compiler units"
                )
            seen[node.name] = len(calls)
        assert set(seen) == set(forwarding), f"premise: only reached {sorted(seen)}"

    def test_the_rotated_capsule_shares_the_exact_segment_bound(self):
        # Both spellings of a capsule end at ``_segment_aabb``, so the cylinder
        # cap term cannot drift between them.
        import inspect

        from strands_robots.simulation.isaac import loaders as mod

        source = inspect.getsource(mod._geom_aabb)
        assert source.count("_segment_aabb(") == 2, "the two capsule spellings no longer share one bound"
        assert "math.sqrt" not in source, "_geom_aabb re-derives a segment extent instead of sharing it"
