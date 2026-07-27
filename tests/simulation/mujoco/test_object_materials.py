"""Regression tests for ``add_object(..., material=...)`` -- matte/textured surfaces.

Before the material plumbing landed, ``add_object`` exposed only ``rgba`` so
every object compiled with ``geom_matid == -1`` (flat, plastic-shiny primitive).
These tests pin the new behaviour:

* A ``material`` dict produces a real ``mjMaterial`` (and, when requested, a
  ``mjTexture``) and the geom's ``matid`` points at it.
* ``material=None`` is byte-for-byte the old behaviour (``matid == -1``).
* Invalid material specs fail loudly (``ValueError``) -- no silent fallback to
  the flat-plastic default.
* A material key the builder cannot honor (a typo, another renderer's field
  name, an empty spec, or ``rgb1``/``rgb2``/``texdim`` without ``builtin``) is
  rejected instead of dropped -- dropping one compiles the glossy default while
  reporting success.

They exercise the full path ``Simulation.add_object`` -> ``SimObject`` ->
``SpecBuilder._build_material`` -> compiled ``MjModel`` so the assertions are on
the compiled model (real behaviour), not internal bookkeeping.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

import mujoco  # noqa: E402

from strands_robots.simulation.models import SimObject, SimWorld  # noqa: E402
from strands_robots.simulation.mujoco.spec_builder import SpecBuilder  # noqa: E402


def _compile(*objects: SimObject) -> mujoco.MjModel:
    """Build a fresh spec from a world holding ``objects`` and compile it."""
    world = SimWorld()
    for obj in objects:
        world.objects[obj.name] = obj
    return SpecBuilder.build(world).compile()


def _matid(model: mujoco.MjModel, geom_name: str) -> int:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    assert gid >= 0, f"geom {geom_name!r} not found in compiled model"
    return int(model.geom_matid[gid])


@pytest.fixture
def texture_png(tmp_path):
    """A tiny on-disk RGB PNG to use as an image texture."""
    from PIL import Image

    path = tmp_path / "surface.png"
    Image.new("RGB", (16, 16), (140, 90, 40)).save(path)
    return str(path)


def test_material_none_is_unchanged_flat_behaviour():
    """material=None must compile with no material assigned (matid == -1)."""
    model = _compile(SimObject(name="plain", shape="box"))
    assert _matid(model, "plain_geom") == -1


def test_matte_material_attaches_real_material():
    """A reflectance/specular/shininess dict yields a real material, no texture."""
    obj = SimObject(
        name="apple",
        shape="sphere",
        material={"specular": 0.0, "shininess": 0.0, "reflectance": 0.0},
    )
    model = _compile(obj)

    matid = _matid(model, "apple_geom")
    assert matid >= 0
    mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "apple_mat")
    assert mat_id == matid
    # Matte values round-tripped into the compiled material.
    assert float(model.mat_specular[matid]) == pytest.approx(0.0)
    assert float(model.mat_shininess[matid]) == pytest.approx(0.0)
    assert float(model.mat_reflectance[matid]) == pytest.approx(0.0)
    # No image/procedural texture was requested -> the only texture in the
    # model is the ground grid's (the matte material carries none of its own).
    assert int(model.mat_texid[matid][mujoco.mjtTextureRole.mjTEXROLE_RGB]) == -1


def test_image_texture_material_attaches_texture(texture_png):
    """A ``texture`` path yields a material whose RGB texture role is populated."""
    obj = SimObject(
        name="table",
        shape="box",
        material={"texture": texture_png, "texrepeat": [2, 2], "specular": 0.0},
    )
    model = _compile(obj)

    matid = _matid(model, "table_geom")
    assert matid >= 0
    tex_role = int(model.mat_texid[matid][mujoco.mjtTextureRole.mjTEXROLE_RGB])
    assert tex_role >= 0, "image texture not bound to the material RGB role"
    tex_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TEXTURE, tex_role)
    assert tex_name == "table_tex"


def test_builtin_texture_material_creates_procedural_texture():
    """A ``builtin`` checker yields a procedural texture bound to the material."""
    obj = SimObject(
        name="floor2",
        shape="box",
        material={"builtin": "checker", "rgb1": [1.0, 0.0, 0.0], "rgb2": [0.0, 1.0, 0.0]},
    )
    model = _compile(obj)

    matid = _matid(model, "floor2_geom")
    assert matid >= 0
    assert int(model.mat_texid[matid][mujoco.mjtTextureRole.mjTEXROLE_RGB]) >= 0


def test_missing_texture_file_fails_loudly(tmp_path):
    """A texture path that does not exist must raise, not silently fall back."""
    obj = SimObject(name="x", shape="box", material={"texture": str(tmp_path / "nope.png")})
    with pytest.raises(ValueError, match="texture file not found"):
        _compile(obj)


def test_unknown_builtin_fails_loudly():
    """An unknown builtin name must raise, not silently fall back."""
    obj = SimObject(name="x", shape="box", material={"builtin": "marble"})
    with pytest.raises(ValueError, match="unknown builtin"):
        _compile(obj)


def test_texture_and_builtin_conflict_fails_loudly(texture_png):
    """Specifying both texture and builtin is ambiguous and must raise."""
    obj = SimObject(name="x", shape="box", material={"texture": texture_png, "builtin": "checker"})
    with pytest.raises(ValueError, match="not both"):
        _compile(obj)


def test_simulation_add_object_material_end_to_end(texture_png):
    """The live Simulation.add_object path plumbs material through to the model."""
    import strands_robots as sr

    sim = sr.Simulation()
    try:
        assert sim.create_world(ground_plane=True)["status"] == "success"
        matte = sim.add_object(
            "ball",
            shape="sphere",
            position=[0.3, 0.0, 0.1],
            material={"specular": 0.0, "shininess": 0.0},
        )
        assert matte["status"] == "success", matte
        textured = sim.add_object(
            "tabletop",
            shape="box",
            position=[0.3, 0.0, 0.0],
            is_static=True,
            material={"texture": texture_png, "texrepeat": [2, 2]},
        )
        assert textured["status"] == "success", textured

        model = sim._world._model
        assert _matid(model, "ball_geom") >= 0
        table_matid = _matid(model, "tabletop_geom")
        assert table_matid >= 0
        assert int(model.mat_texid[table_matid][mujoco.mjtTextureRole.mjTEXROLE_RGB]) >= 0
    finally:
        sim.destroy()


def test_simulation_add_object_bad_material_reports_error(tmp_path):
    """Bad material via the live path is rejected (status=error, no crash).

    The material is built before the body is added to the spec, so a rejected
    add leaves no orphan body behind: re-adding the same name with a valid
    material must succeed.
    """
    import strands_robots as sr

    sim = sr.Simulation()
    try:
        sim.create_world(ground_plane=True)
        result = sim.add_object(
            "ball",
            shape="sphere",
            material={"texture": str(tmp_path / "missing.png")},
        )
        assert result["status"] == "error", result
        # Failed add must not leave the object registered.
        assert "ball" not in sim._world.objects
        # No orphan body in the spec: the same name re-adds cleanly.
        retry = sim.add_object("ball", shape="sphere", material={"specular": 0.0})
        assert retry["status"] == "success", retry
        assert _matid(sim._world._model, "ball_geom") >= 0
    finally:
        sim.destroy()


@pytest.mark.parametrize(
    ("material", "expected"),
    [
        # A typo in a procedural colour key silently dropped the caller's
        # colours and compiled the default grey checker.
        ({"builtin": "checker", "rgb_1": [1.0, 0.0, 0.0]}, "did you mean 'rgb1'"),
        # Another renderer's field name: no MuJoCo equivalent, so it did nothing.
        ({"roughness": 0.2}, "'roughness'"),
        ({"tex_repeat": [4, 4]}, "did you mean 'texrepeat'"),
        # Real mjMaterial field that this schema does not plumb through.
        ({"emission": 0.5}, "'emission'"),
        ({"specular": 0.0, "shininess": 0.0, "bogus": 1}, "'bogus'"),
        # rgb1/rgb2/texdim only colour a procedural texture.
        ({"rgb1": [1.0, 0.0, 0.0], "rgb2": [0.0, 0.0, 1.0]}, "only colour/size a procedural"),
        ({"texture": "/tmp/x.png", "texdim": 256}, "only colour/size a procedural"),
        # An empty spec cannot express anything: it compiled MuJoCo's defaults.
        ({}, "has no effect"),
    ],
)
def test_unhonorable_material_keys_fail_loudly(material, expected):
    """A material key the builder cannot honor must raise, not be dropped."""
    obj = SimObject(name="x", shape="box", material=material)
    with pytest.raises(ValueError, match=expected):
        _compile(obj)


def test_non_mapping_material_fails_loudly():
    """A non-dict material must raise a ValueError, not an opaque AttributeError."""
    obj = SimObject(name="x", shape="box", material=["specular", 0.0])
    with pytest.raises(ValueError, match="expected a dict of material options, got list"):
        _compile(obj)


def test_simulation_add_object_rejects_unknown_material_key():
    """The live path reports status=error and leaves no object registered.

    Pre-fix this returned ``status=success`` with the unknown key dropped, so
    the caller believed a red checkerboard had been applied while the compiled
    geom carried the default grey checker.
    """
    import strands_robots as sr

    sim = sr.Simulation()
    try:
        sim.create_world(ground_plane=True)
        result = sim.add_object(
            "cube",
            shape="box",
            material={"builtin": "checker", "rgb_1": [1.0, 0.0, 0.0], "rgb2": [0.0, 0.0, 1.0]},
        )
        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        assert "rgb_1" in text and "rgb1" in text
        # Accepted keys are listed so the caller can self-correct.
        assert "texrepeat" in text
        assert "cube" not in sim._world.objects
        # The corrected spelling is accepted and reaches the compiled model.
        retry = sim.add_object(
            "cube",
            shape="box",
            material={"builtin": "checker", "rgb1": [1.0, 0.0, 0.0], "rgb2": [0.0, 0.0, 1.0]},
        )
        assert retry["status"] == "success", retry
        matid = _matid(sim._world._model, "cube_geom")
        assert matid >= 0
        assert int(sim._world._model.mat_texid[matid][mujoco.mjtTextureRole.mjTEXROLE_RGB]) >= 0
    finally:
        sim.destroy()
