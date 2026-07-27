"""``add_object`` honors every ``size`` component or rejects the vector.

The MuJoCo backend documents an exact per-shape ``size`` layout (full extents in
meters: ``box``/``ellipsoid`` need all three components, ``cylinder``/``capsule``
need the diameter and the height at index 2, ``sphere``/``plane`` need only their
leading component). A vector shorter than the shape consumes used to be replaced
wholesale by a hardcoded default, so ``add_object("crate", shape="box",
size=[0.5])`` reported success -- echoing the requested ``[0.5]`` -- while
compiling a 10 cm cube, and a vector longer than three components died with a
generic "spec recompile refused" that never named the parameter.

These tests pin the contract at both entry points that normalize a size (the
``add_object`` action and the ``patch_scene_mjcf`` ``add_geom`` op) plus the
shared helpers, and pin that the legitimately shorter layouts
(``sphere=[diameter]``, ``plane=[x]``) keep working.
"""

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402
from strands_robots.simulation.mujoco.spec_builder import (  # noqa: E402
    _normalize_size,
    _validate_size,
)


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_size_count_sim", mesh=False)
    s.create_world()
    yield s
    s.cleanup()


def _geom_id(sim, geom_name):
    return mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_GEOM, geom_name)


def _half_extents(sim, geom_name):
    gid = _geom_id(sim, geom_name)
    assert gid >= 0, f"geom {geom_name!r} missing from the compiled model"
    return list(sim._world._model.geom_size[gid])


# Shapes/lengths that previously returned success while compiling a default-sized
# geom: the caller's extents were dropped entirely, not padded.
PARTIAL_SIZES = [
    ("box", [0.5]),
    ("box", [0.4, 0.3]),
    ("box", []),
    ("ellipsoid", [0.3, 0.3]),
    ("cylinder", [0.2]),
    ("capsule", [0.2]),
    ("sphere", []),
    ("plane", []),
]


@pytest.mark.parametrize(("shape", "size"), PARTIAL_SIZES)
def test_partial_size_is_rejected_and_nothing_is_added(sim, shape, size):
    """A size the shape cannot consume is refused, leaving the scene untouched."""
    result = sim.add_object("part", shape=shape, size=size, is_static=shape == "plane")

    assert result["status"] == "error"
    message = result["content"][0]["text"]
    assert "size" in message
    assert str(len(size)) in message  # names the count it actually got
    # The refusal must not half-apply: no registry entry, no compiled geom.
    assert "part" not in sim._world.objects
    assert _geom_id(sim, "part_geom") < 0


def test_partial_box_size_error_names_the_shape_and_required_layout(sim):
    """The message is self-correcting: shape, required count, layout, convention."""
    result = sim.add_object("crate", shape="box", size=[0.5])

    message = result["content"][0]["text"]
    assert "box" in message
    assert "3" in message
    assert "[x, y, z] full edge lengths" in message
    assert "full extent in meters" in message
    assert "[0.5]" in message  # echoes what was passed


def test_size_longer_than_three_components_is_rejected_before_recompile(sim):
    """A 4-component size names the parameter instead of failing the recompile.

    It previously reached the spec compiler and surfaced as
    "spec recompile refused." -- no mention of ``size`` or of the real reason.
    """
    result = sim.add_object("over", shape="box", size=[0.1, 0.1, 0.1, 0.1])

    assert result["status"] == "error"
    message = result["content"][0]["text"]
    assert "size" in message
    assert "at most 3" in message
    assert "recompile" not in message
    assert "over" not in sim._world.objects


def test_complete_size_vectors_are_honored(sim):
    """The honored path is unchanged: full extents halve into MuJoCo geom sizes."""
    assert sim.add_object("cube", shape="box", size=[0.5, 0.4, 0.3])["status"] == "success"
    assert _half_extents(sim, "cube_geom") == pytest.approx([0.25, 0.2, 0.15])

    assert sim.add_object("can", shape="cylinder", size=[0.08, 0.0, 0.2])["status"] == "success"
    can = _half_extents(sim, "can_geom")
    assert can[0] == pytest.approx(0.04)  # radius from diameter
    assert can[1] == pytest.approx(0.1)  # half-height from full height


def test_shapes_that_document_a_shorter_layout_still_accept_it(sim):
    """The guard must not over-reject: sphere and plane consume fewer components."""
    assert sim.add_object("ball", shape="sphere", size=[0.06])["status"] == "success"
    assert _half_extents(sim, "ball_geom")[0] == pytest.approx(0.03)

    assert sim.add_object("mat", shape="plane", size=[2.0], is_static=True)["status"] == "success"
    mat = _half_extents(sim, "mat_geom")
    assert mat[0] == pytest.approx(2.0)
    assert mat[1] == pytest.approx(2.0)  # y mirrors x when omitted


def test_omitted_size_uses_the_documented_default(sim):
    """``size=None`` still means the documented 5 cm box (2.5 cm half-extents)."""
    assert sim.add_object("default_box")["status"] == "success"
    assert _half_extents(sim, "default_box_geom") == pytest.approx([0.025, 0.025, 0.025])


def test_patch_scene_mjcf_add_geom_rejects_partial_size(sim):
    """The scene-patch entry point normalizes sizes too, and rolls the batch back."""
    result = sim.patch_scene_mjcf(
        [
            {"op": "add_body", "name": "pbody", "pos": [1.0, 0.0, 1.0]},
            {"op": "add_geom", "body": "pbody", "name": "pgeom", "type": "box", "size": [0.4]},
        ]
    )

    assert result["status"] == "error"
    assert "size" in result["content"][0]["text"]
    # Atomic rollback: neither op survives, so no default-sized geom is left behind.
    assert _geom_id(sim, "pgeom") < 0
    assert mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_BODY, "pbody") < 0


class TestValidateSizeComponentCount:
    """Unit-level contract of the shared size validator/normalizer."""

    @pytest.mark.parametrize(("shape", "size"), PARTIAL_SIZES)
    def test_short_vectors_report_an_error(self, shape, size):
        assert _validate_size(shape, size) is not None

    def test_more_than_three_components_reports_an_error(self):
        assert "at most 3" in (_validate_size("sphere", [0.1, 0.1, 0.1, 0.1]) or "")

    @pytest.mark.parametrize(
        ("shape", "size"),
        [
            ("box", [0.2, 0.4, 0.6]),
            ("ellipsoid", [0.1, 0.2, 0.3]),
            ("cylinder", [0.1, 0.0, 0.4]),
            ("capsule", [0.1, 0.0, 0.4]),
            ("sphere", [0.1]),
            ("plane", [2.0]),
            ("mesh", []),
        ],
    )
    def test_documented_layouts_pass(self, shape, size):
        assert _validate_size(shape, size) is None

    def test_normalize_raises_instead_of_padding_a_short_vector(self):
        """Direct builder callers get a loud ValueError, not a defaulted geom."""
        with pytest.raises(ValueError, match=r"needs 3 'size' component"):
            _normalize_size("box", [0.5])
        with pytest.raises(ValueError, match=r"needs 3 'size' component"):
            _normalize_size("ellipsoid", [])
