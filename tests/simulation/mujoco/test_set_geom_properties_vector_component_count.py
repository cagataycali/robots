"""``set_geom_properties`` honors every vector component or rejects the vector.

``color`` / ``friction`` / ``size`` are vector-valued, and each targets a MuJoCo
model buffer with a fixed component layout: ``geom_rgba`` is RGBA, ``geom_friction``
is (sliding, torsional, rolling), and ``geom_size`` carries as many half-extents as
the geom's compiled type defines. The mutator used to write whatever it was given,
component by component:

* ``size`` wrote ``geom_size[gid, :min(len(size), 3)]``, so a one-element vector
  resized x on a box and left y/z at their compiled value - a shape the caller
  never described, reported as ``status="success"``. A longer vector had its tail
  silently discarded, and an empty one changed nothing while reporting a resize.
* ``friction`` padded with zeros, so ``friction=[1.0]`` also zeroed the torsional
  and rolling coefficients the caller never mentioned (and ``friction=[]`` made the
  geom frictionless).
* ``color`` appended an alpha to ``color[:3]``, so a one- or two-element vector
  produced a 2/3-wide array and crashed with a bare NumPy broadcast ``ValueError``
  past the tool envelope, while ``color=[]`` repainted the geom opaque white.

These pin the contract: a vector whose component count cannot be honored is
rejected with an actionable structured error, the model is left untouched, and the
exact counts each type defines still apply.
"""

import mujoco
import numpy as np
import pytest

from strands_robots.simulation.mujoco import Simulation

# Inline scene with one geom per size-defined primitive plus a mesh geom whose
# extent comes from asset data (vertices are inline, so no asset file is needed).
_SCENE = """
<mujoco>
  <asset>
    <mesh name="tetra" vertex="0 0 0  0.1 0 0  0 0.1 0  0 0 0.1"/>
  </asset>
  <worldbody>
    <geom name="ground" type="plane" size="5 5 0.1"/>
    <body name="b_box" pos="0 0 1"><freejoint/>
      <geom name="box_g" type="box" size="0.1 0.15 0.2"/></body>
    <body name="b_sphere" pos="1 0 1"><freejoint/>
      <geom name="sphere_g" type="sphere" size="0.1"/></body>
    <body name="b_capsule" pos="2 0 1"><freejoint/>
      <geom name="capsule_g" type="capsule" size="0.05 0.2"/></body>
    <body name="b_cylinder" pos="3 0 1"><freejoint/>
      <geom name="cylinder_g" type="cylinder" size="0.05 0.2"/></body>
    <body name="b_ellipsoid" pos="4 0 1"><freejoint/>
      <geom name="ellipsoid_g" type="ellipsoid" size="0.1 0.2 0.3"/></body>
    <body name="b_mesh" pos="5 0 1"><freejoint/>
      <geom name="mesh_g" type="mesh" mesh="tetra"/></body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def sim():
    """A sim whose scene carries one geom of every geom_size layout."""
    s = Simulation(tool_name="test_geom_component_count", mesh=False)
    s.create_world()
    assert s.replace_scene_mjcf(_SCENE)["status"] == "success"
    yield s
    s.cleanup()


def _gid(sim, name):
    return mujoco.mj_name2id(sim._world._model, mujoco.mjtObj.mjOBJ_GEOM, name)


def _snapshot(sim, name):
    model, gid = sim._world._model, _gid(sim, name)
    return {
        "size": model.geom_size[gid].copy(),
        "friction": model.geom_friction[gid].copy(),
        "rgba": model.geom_rgba[gid].copy(),
        "rbound": float(model.geom_rbound[gid]),
    }


def _assert_untouched(sim, name, before):
    after = _snapshot(sim, name)
    for key in ("size", "friction", "rgba"):
        assert after[key] == pytest.approx(before[key]), f"{key} was mutated by a rejected call"
    assert after["rbound"] == pytest.approx(before["rbound"])


@pytest.mark.parametrize(
    ("geom", "bad_size"),
    [
        # Short: the omitted components would keep their compiled value.
        ("box_g", [0.2]),
        ("box_g", [0.2, 0.3]),
        ("ellipsoid_g", [0.2, 0.3]),
        ("capsule_g", [0.05]),
        ("cylinder_g", [0.05]),
        ("ground", [8.0, 8.0]),
        # Long: the tail would be discarded.
        ("box_g", [0.2, 0.3, 0.4, 0.5]),
        ("sphere_g", [0.2, 0.3]),
        ("capsule_g", [0.05, 0.2, 0.3]),
        # Empty: nothing would be written at all.
        ("box_g", []),
        ("sphere_g", []),
    ],
)
def test_size_component_count_mismatch_rejected_and_model_untouched(sim, geom, bad_size):
    """A size vector that does not match the geom type's component count is refused."""
    before = _snapshot(sim, geom)
    result = sim.set_geom_properties(geom_name=geom, size=bad_size)
    assert result["status"] == "error", f"{geom} accepted size={bad_size!r}"
    _assert_untouched(sim, geom, before)


@pytest.mark.parametrize(
    ("geom", "expected", "layout_word"),
    [
        ("sphere_g", 1, "radius"),
        ("capsule_g", 2, "half-length"),
        ("cylinder_g", 2, "half-length"),
        ("box_g", 3, "half-extents"),
        ("ellipsoid_g", 3, "semi-axes"),
        ("ground", 3, "grid spacing"),
    ],
)
def test_size_error_names_the_type_its_count_and_its_layout(sim, geom, expected, layout_word):
    """The error teaches the caller the exact vector the geom's type needs.

    A component-count rejection is only actionable if it says how many components
    this particular geom wants and what they mean - the count is type-dependent,
    so a generic "wrong length" leaves the caller guessing.
    """
    result = sim.set_geom_properties(geom_name=geom, size=[0.1, 0.2, 0.3, 0.4])
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert f"exactly {expected} component(s)" in text
    assert layout_word in text
    assert "'size'" in text


@pytest.mark.parametrize(
    ("geom", "good_size", "expected_prefix"),
    [
        ("sphere_g", [0.3], [0.3]),
        ("capsule_g", [0.06, 0.25], [0.06, 0.25]),
        ("cylinder_g", [0.06, 0.25], [0.06, 0.25]),
        ("box_g", [0.2, 0.3, 0.4], [0.2, 0.3, 0.4]),
        ("ellipsoid_g", [0.15, 0.25, 0.4], [0.15, 0.25, 0.4]),
        ("ground", [8.0, 8.0, 0.2], [8.0, 8.0, 0.2]),
    ],
)
def test_exact_component_count_still_applies(sim, geom, good_size, expected_prefix):
    """Every size-defined primitive still resizes when given its exact vector."""
    result = sim.set_geom_properties(geom_name=geom, size=good_size)
    assert result["status"] == "success", result["content"][0]["text"]
    gid = _gid(sim, geom)
    written = sim._world._model.geom_size[gid][: len(expected_prefix)].tolist()
    assert written == pytest.approx(expected_prefix)


def test_size_refused_for_a_geom_whose_extent_comes_from_its_asset(sim):
    """A mesh geom defines no ``geom_size`` component, so ``size`` is refused.

    ``geom_size`` is ignored by the compiler for mesh / height-field / SDF geoms:
    their extent comes from the asset. Storing the requested value would report a
    resize that never happens, so the call is refused and names the alternatives.
    """
    before = _snapshot(sim, "mesh_g")
    result = sim.set_geom_properties(geom_name="mesh_g", size=[0.2, 0.2, 0.2])
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "mesh" in text
    assert "asset" in text
    assert "box" in text and "sphere" in text
    _assert_untouched(sim, "mesh_g", before)


@pytest.mark.parametrize("bad_friction", [[], [1.0], [1.0, 0.5]])
def test_partial_friction_rejected_instead_of_zero_padded(sim, bad_friction):
    """A friction vector shorter than three coefficients is refused, not padded.

    Zero-padding silently replaced the torsional and rolling coefficients with 0.0,
    which is not MuJoCo's default and lets a resting object spin and roll freely -
    a contact model the caller never asked for under a success result.
    """
    before = _snapshot(sim, "box_g")
    result = sim.set_geom_properties(geom_name="box_g", friction=bad_friction)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "exactly 3 component(s)" in text
    assert "sliding, torsional, rolling" in text
    _assert_untouched(sim, "box_g", before)


def test_full_friction_vector_still_applies(sim):
    """The three-coefficient vector is written verbatim."""
    result = sim.set_geom_properties(geom_name="box_g", friction=[0.8, 0.02, 0.002])
    assert result["status"] == "success"
    gid = _gid(sim, "box_g")
    assert sim._world._model.geom_friction[gid].tolist() == pytest.approx([0.8, 0.02, 0.002])


@pytest.mark.parametrize("bad_color", [[], [0.5], [0.5, 0.5], [1.0, 0.0, 0.0, 1.0, 1.0]])
def test_color_component_count_mismatch_rejected_not_raised(sim, bad_color):
    """A color that is not RGB or RGBA returns a structured error, never raises.

    One- and two-element colors used to reach ``geom_rgba[gid] = color[:3] + [1.0]``
    and crash with a bare NumPy broadcast ``ValueError`` past the tool envelope,
    while an empty color repainted the geom opaque white and a five-element one had
    its tail dropped.
    """
    before = _snapshot(sim, "box_g")
    result = sim.set_geom_properties(geom_name="box_g", color=bad_color)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "exactly 3 or 4 component(s)" in text
    assert "'color'" in text
    _assert_untouched(sim, "box_g", before)


@pytest.mark.parametrize(
    ("color", "expected_rgba"),
    [
        ([0.2, 0.4, 0.6], [0.2, 0.4, 0.6, 1.0]),
        ([0.2, 0.4, 0.6, 0.5], [0.2, 0.4, 0.6, 0.5]),
    ],
)
def test_rgb_and_rgba_colors_still_apply(sim, color, expected_rgba):
    """RGB gets an opaque alpha; RGBA is written verbatim."""
    result = sim.set_geom_properties(geom_name="box_g", color=color)
    assert result["status"] == "success"
    gid = _gid(sim, "box_g")
    assert sim._world._model.geom_rgba[gid].tolist() == pytest.approx(expected_rgba, abs=1e-6)


def test_rejected_vector_does_not_partially_apply_the_other_vectors(sim):
    """Validation precedes every write, so one bad vector aborts the whole call.

    ``color`` / ``friction`` / ``size`` are applied under a single lock; a valid
    color must not land while an unusable size is rejected, or the caller has to
    reason about which half of its request survived.
    """
    before = _snapshot(sim, "box_g")
    result = sim.set_geom_properties(
        geom_name="box_g",
        color=[1.0, 0.0, 0.0, 1.0],
        friction=[0.9, 0.01, 0.001],
        size=[0.2],
    )
    assert result["status"] == "error"
    _assert_untouched(sim, "box_g", before)


def test_grown_box_still_recomputes_collision_bounds(sim):
    """The exact-count path keeps the broadphase/mid-phase bound recompute."""
    model = sim._world._model
    gid = _gid(sim, "box_g")
    result = sim.set_geom_properties(geom_name="box_g", size=[0.25, 0.25, 0.02])
    assert result["status"] == "success"
    assert float(model.geom_rbound[gid]) == pytest.approx(float(np.linalg.norm([0.25, 0.25, 0.02])))
    assert model.geom_aabb[gid][3:6].tolist() == pytest.approx([0.25, 0.25, 0.02])


def test_describe_advertises_the_component_counts(sim):
    """describe() teaches the counts, so a caller need not discover them by trial.

    The component count is type-dependent and cannot be guessed from the signature,
    which is exactly the knowledge gap that produced partial-vector calls.
    """
    advertised = sim.describe()["methods"]["set_geom_properties"]
    assert "RGB" in advertised
    assert "sliding" in advertised
    assert "sphere 1" in advertised
