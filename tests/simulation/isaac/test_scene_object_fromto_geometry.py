"""A ``fromto`` collision geom carries its length and its offset.

MJCF spells a capsule / cylinder two ways: ``pos`` + ``size="radius
half-length"`` along the geom's local z, or ``fromto`` + ``size="radius"``,
where the two endpoints carry the placement *and* the axis extent. MuJoCo
refuses a geom declaring ``pos`` and ``fromto`` together, and refuses a
capsule / cylinder that carries one ``size`` component and no ``fromto``, so
the one-component shape is reachable only through ``fromto`` - the two
premises this file pins against MuJoCo directly.

Read as ``pos`` + ``size`` alone, such a geom collapses to a ball of that
radius at the body origin: a 0.60 m bar becomes a 0.04 m sphere and the
segment midpoint is lost. Both halves are load-bearing and both are reported
under ``status="success"`` - ``SceneObject.size`` is the collision proxy
``IsaacSimulation.load_scene`` realizes for the object, and
``SceneObject.offset`` is what the LIBERO pose applier adds to a body pose to
place the prim (``prim_pos = xpos + xmat @ offset``).

MuJoCo is the oracle. For an axis-aligned ``fromto`` its compiler resolves
``geom_pos`` to the segment midpoint and ``geom_size`` to ``(radius,
half-length)``, and the body-frame bound derived from those matches this
loader's exactly.

``fromto`` on a box or ellipsoid squares the cross-section by copying the
first ``size`` component and needs the rotated-box bound, which this loader
does not compute; those keep returning no analytic AABB so the caller falls
back, and that boundary is pinned here so a fix aimed at the segment shapes
does not quietly reach past them.
"""

from __future__ import annotations

import pytest

from strands_robots.simulation.isaac.loaders import load_mjcf_scene_objects

# (geom type, geom attributes, expected body-frame offset, expected full size).
# Every extent below is MuJoCo's own resolved geometry, cross-checked by
# ``test_the_reported_box_matches_mujocos_own_resolved_geometry``.
_AXIS_ALIGNED = [
    pytest.param(
        "capsule",
        'fromto="0 0 0  0 0 0.6" size="0.02"',
        (0.0, 0.0, 0.3),
        (0.04, 0.04, 0.64),
        id="capsule-along-z",
    ),
    pytest.param(
        "cylinder",
        'fromto="0 0 0  0 0 0.6" size="0.02"',
        (0.0, 0.0, 0.3),
        (0.04, 0.04, 0.6),
        id="cylinder-along-z",
    ),
    pytest.param(
        "capsule",
        'fromto="-0.4 0 0.1  0.4 0 0.1" size="0.03"',
        (0.0, 0.0, 0.1),
        (0.86, 0.06, 0.06),
        id="capsule-along-x-raised",
    ),
    pytest.param(
        "cylinder",
        'fromto="0 -0.25 0.05  0 0.25 0.05" size="0.04"',
        (0.0, 0.0, 0.05),
        (0.08, 0.5, 0.08),
        id="cylinder-along-y-raised",
    ),
]

#: What ``load_mjcf_scene_objects`` falls back to for a body with no analytic
#: collision geometry and no parseable mesh.
_NO_GEOMETRY_FALLBACK = (0.05, 0.05, 0.05)


def _write_scene(tmp_path, body_xml: str) -> str:
    scene = f"""
    <mujoco model="probe">
      <worldbody>
        <geom name="floor" type="plane" size="2 2 0.1"/>
        {body_xml}
      </worldbody>
    </mujoco>
    """
    path = tmp_path / "scene.xml"
    path.write_text(scene, encoding="utf-8")
    return str(path)


def _one_object(tmp_path, geom_xml: str, body_pos: str = "0 0 0"):
    """Load a scene holding one movable body with ``geom_xml`` inside it."""
    scene = _write_scene(
        tmp_path,
        f'<body name="probe_1_main" pos="{body_pos}"><joint type="free"/>{geom_xml}</body>',
    )
    (obj,) = load_mjcf_scene_objects(scene)
    return obj


def _mujoco_body_frame_aabb(scene_path: str, geom_name: str = "g"):
    """The body-frame AABB MuJoCo's own compiler resolves for ``geom_name``.

    ``model.geom_aabb`` is the geom's bounding box in its LOCAL frame, so
    rotating that box by the compiled ``geom_quat`` and translating by
    ``geom_pos`` gives the body-frame bound. For an axis-aligned ``fromto``
    the rotation is a signed permutation, so the rotated box is the exact
    AABB - which is the only case this oracle is used for.
    """
    mujoco = pytest.importorskip("mujoco")
    import numpy as np

    model = mujoco.MjModel.from_xml_path(scene_path)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    assert gid >= 0, f"premise: geom {geom_name!r} is missing from the compiled model"
    local_center = model.geom_aabb[gid][:3].copy()
    local_half = model.geom_aabb[gid][3:].copy()
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.geom_quat[gid])
    rot = rot.reshape(3, 3)
    center = model.geom_pos[gid] + rot @ local_center
    half = np.abs(rot) @ local_half
    return tuple(float(v) for v in center), tuple(float(2.0 * v) for v in half)


class TestTheSegmentIsTheGeometry:
    """The endpoints give the extent and the placement, not a ball at the origin."""

    @pytest.mark.parametrize(("gtype", "attrs", "offset", "size"), _AXIS_ALIGNED)
    def test_the_reported_box_spans_the_segment(self, tmp_path, gtype, attrs, offset, size):
        obj = _one_object(tmp_path, f'<geom name="g" type="{gtype}" {attrs} group="0"/>')
        assert obj.size == pytest.approx(size), (
            f"a {gtype} declared with {attrs} reports extents {obj.size} instead of {size}: "
            f"read as pos+size its endpoints are discarded and the shape collapses to a ball "
            f"of its radius, which is the collision proxy load_scene realizes"
        )
        assert obj.offset == pytest.approx(offset), (
            f"a {gtype} declared with {attrs} reports a body-frame offset of {obj.offset} "
            f"instead of the segment midpoint {offset}, which is what the LIBERO pose applier "
            f"adds to a body pose to place the prim"
        )

    @pytest.mark.parametrize(("gtype", "attrs", "offset", "size"), _AXIS_ALIGNED)
    def test_the_reported_box_matches_mujocos_own_resolved_geometry(self, tmp_path, gtype, attrs, offset, size):
        scene = _write_scene(
            tmp_path,
            f'<body name="probe_1_main" pos="0 0 0"><joint type="free"/>'
            f'<geom name="g" type="{gtype}" {attrs} group="0"/></body>',
        )
        want_center, want_size = _mujoco_body_frame_aabb(scene)
        (obj,) = load_mjcf_scene_objects(scene)
        assert obj.size == pytest.approx(want_size, abs=1e-9), (
            f"MuJoCo resolves this {gtype} to a body-frame extent of {want_size}; the loader reports {obj.size}"
        )
        assert obj.offset == pytest.approx(want_center, abs=1e-9), (
            f"MuJoCo resolves this {gtype}'s centre to {want_center}; the loader reports {obj.offset}"
        )

    def test_a_cylinders_axis_extent_excludes_the_end_caps(self, tmp_path):
        """The capsule / cylinder distinction survives the endpoint spelling."""
        shared = 'fromto="0 0 0  0 0 0.6" size="0.02"'
        capsule = _one_object(tmp_path, f'<geom name="g" type="capsule" {shared} group="0"/>')
        cylinder = _one_object(tmp_path, f'<geom name="g" type="cylinder" {shared} group="0"/>')
        assert capsule.size[2] == pytest.approx(0.64)
        assert cylinder.size[2] == pytest.approx(0.6)
        assert capsule.size[2] > cylinder.size[2], (
            "a capsule's hemispherical caps add its radius at each end of the segment while a "
            f"cylinder's flat cap does not, yet both report {capsule.size[2]}"
        )

    def test_the_world_position_folds_in_the_body_pos(self, tmp_path):
        obj = _one_object(
            tmp_path,
            '<geom name="g" type="capsule" fromto="0 0 0  0 0 0.6" size="0.02" group="0"/>',
            body_pos="0.1 -0.2 0.8",
        )
        assert obj.position == pytest.approx((0.1, -0.2, 1.1))

    def test_a_segment_geom_unions_with_an_analytic_sibling(self, tmp_path):
        """A body mixing spellings bounds both geoms, not just the pos+size one."""
        obj = _one_object(
            tmp_path,
            '<geom name="g" type="capsule" fromto="0 0 0  0 0 0.6" size="0.02" group="0"/>'
            '<geom name="base" type="box" pos="0 0 -0.05" size="0.1 0.1 0.03" group="0"/>',
        )
        # x / y come from the wider box; z spans the box's floor at -0.08 up to
        # the capsule's far cap at 0.6 + 0.02, so both geoms bound the result.
        assert obj.size == pytest.approx((0.2, 0.2, 0.7))
        assert obj.offset == pytest.approx((0.0, 0.0, 0.27))

    def test_a_diagonal_segment_is_bounded_exactly_not_by_a_rotated_box(self, tmp_path):
        """The exact segment bound, which is tighter than boxing the rotated local box."""
        attrs = 'fromto="0 0 0  0.3 0.4 0" size="0.05"'
        capsule = _one_object(tmp_path, f'<geom name="g" type="capsule" {attrs} group="0"/>')
        cylinder = _one_object(tmp_path, f'<geom name="g" type="cylinder" {attrs} group="0"/>')
        assert capsule.size == pytest.approx((0.4, 0.5, 0.1))
        assert cylinder.size == pytest.approx((0.38, 0.46, 0.1))
        assert capsule.offset == pytest.approx((0.15, 0.2, 0.0))

        scene = _write_scene(
            tmp_path,
            f'<body name="probe_1_main"><joint type="free"/><geom name="g" type="capsule" {attrs} group="0"/></body>',
        )
        _center, boxed = _mujoco_body_frame_aabb(scene)
        for exact, superset in zip(capsule.size, boxed, strict=True):
            assert exact <= superset + 1e-9, (
                "the exact segment bound must never exceed the bound obtained by boxing "
                f"MuJoCo's rotated local box: {capsule.size} vs {boxed}"
            )


class TestAFileMuJoCoRefusesGetsNoConfidentBox:
    """A ``fromto`` MuJoCo would not compile leaves no analytic geometry behind."""

    @pytest.mark.parametrize(
        "attrs",
        [
            pytest.param('fromto="0 0 0  0 0" size="0.02"', id="five-numbers"),
            pytest.param('fromto="0 0 0  0 0 1 9" size="0.02"', id="seven-numbers"),
            pytest.param('fromto="a b c d e f" size="0.02"', id="non-numeric"),
            pytest.param('fromto="0 0 0.2  0 0 0.2" size="0.02"', id="zero-length"),
        ],
    )
    def test_an_uncompilable_segment_falls_back_instead_of_asserting_a_ball(self, tmp_path, attrs):
        obj = _one_object(tmp_path, f'<geom name="g" type="capsule" {attrs} group="0"/>')
        assert obj.size == pytest.approx(_NO_GEOMETRY_FALLBACK), (
            f"a capsule declared with {attrs} is not a shape MuJoCo compiles, yet the loader "
            f"reports {obj.size} - a confident box for geometry it could not read"
        )


class TestNothingElseMoves:
    """The ``pos`` + ``size`` spellings and the out-of-scope shapes are untouched."""

    @pytest.mark.parametrize(
        ("attrs", "offset", "size"),
        [
            pytest.param(
                'type="capsule" pos="0 0 0.1" size="0.02 0.3"', (0.0, 0.0, 0.1), (0.04, 0.04, 0.64), id="capsule"
            ),
            pytest.param(
                'type="cylinder" pos="0 0 0.1" size="0.02 0.3"', (0.0, 0.0, 0.1), (0.04, 0.04, 0.6), id="cylinder"
            ),
            pytest.param(
                'type="box" pos="0 0 0.1" size="0.05 0.06 0.07"', (0.0, 0.0, 0.1), (0.1, 0.12, 0.14), id="box"
            ),
            pytest.param('type="sphere" pos="0 0 0.2" size="0.03"', (0.0, 0.0, 0.2), (0.06, 0.06, 0.06), id="sphere"),
            pytest.param(
                'type="ellipsoid" pos="0 0 0.2" size="0.03 0.04 0.05"',
                (0.0, 0.0, 0.2),
                (0.06, 0.08, 0.1),
                id="ellipsoid",
            ),
        ],
    )
    def test_a_pos_and_size_geom_is_unchanged(self, tmp_path, attrs, offset, size):
        obj = _one_object(tmp_path, f'<geom name="g" {attrs} group="0"/>')
        assert obj.size == pytest.approx(size)
        assert obj.offset == pytest.approx(offset)

    @pytest.mark.parametrize("gtype", ["box", "ellipsoid"])
    def test_a_fromto_box_or_ellipsoid_still_has_no_analytic_aabb(self, tmp_path, gtype):
        """Out of scope: those need the rotated-box bound, so they still fall back.

        MuJoCo accepts ``fromto`` on a box and an ellipsoid too, squaring the
        cross-section by copying the first ``size`` component. Reporting a bound
        for them means boxing a rotated box, which the segment formula does not
        do - so they keep degrading to the caller's fallback rather than
        asserting an approximation this loader has not computed.
        """
        obj = _one_object(tmp_path, f'<geom name="g" type="{gtype}" fromto="0 0 0  0 0 0.5" size="0.06" group="0"/>')
        assert obj.size == pytest.approx(_NO_GEOMETRY_FALLBACK)

    def test_a_segment_geom_with_no_radius_falls_back(self, tmp_path):
        """``size`` still has to carry the radius; MuJoCo refuses a missing one too."""
        obj = _one_object(tmp_path, '<geom name="g" type="capsule" fromto="0 0 0  0 0 0.6" group="0"/>')
        assert obj.size == pytest.approx(_NO_GEOMETRY_FALLBACK)


class TestThePremisesMuJoCoSettles:
    """The two facts that make the one-``size`` shape a ``fromto`` shape."""

    @pytest.mark.parametrize("gtype", ["capsule", "cylinder"])
    def test_one_size_and_no_fromto_is_not_a_shape_mujoco_compiles(self, tmp_path, gtype):
        mujoco = pytest.importorskip("mujoco")
        scene = _write_scene(
            tmp_path,
            f'<body name="probe_1_main"><joint type="free"/>'
            f'<geom name="g" type="{gtype}" size="0.02" group="0"/></body>',
        )
        with pytest.raises(ValueError, match="size 1 must be positive"):
            mujoco.MjModel.from_xml_path(scene)

    def test_pos_and_fromto_together_are_refused(self, tmp_path):
        """So the endpoints are the whole placement - there is nothing to compose."""
        mujoco = pytest.importorskip("mujoco")
        scene = _write_scene(
            tmp_path,
            '<body name="probe_1_main"><joint type="free"/>'
            '<geom name="g" type="capsule" fromto="0 0 0  0 0 0.6" pos="1 2 3" size="0.02" group="0"/></body>',
        )
        with pytest.raises(ValueError, match="both pos and fromto"):
            mujoco.MjModel.from_xml_path(scene)
