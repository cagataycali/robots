"""An MJCF's mesh registry and search directory are read from the whole model.

MuJoCo treats ``<include file=...>`` as a textual splice, so ``<compiler>`` and
``<asset>`` are model-global: the fragment declaring the mesh search directory
need not be the fragment declaring the mesh, and neither need be the top file.
:func:`strands_robots.assets.download._mjcf_mesh_subdir` states that rule for
the robot-asset path and names the harm of a narrower reader - it "reports a
mesh that is present as absent".

Reading only the top file's direct children broke both directions, and they
failed differently. An ``<asset>`` block in an included fragment left the
registry empty, so the object fell back to the hardcoded 0.05 m box proxy that
#2459 removed - silent, and reported as a successful scene load. A
``<compiler meshdir>`` in an included fragment resolved the mesh against the
model directory instead, so the existence check refused a file that is present
and the whole scene load raised, losing every object rather than one.

Every fixture here is a model real MuJoCo compiles (pinned in
``TestMuJoCoCompilesEveryFixture``), which is what makes a loader disagreement a
defect rather than invalid scaffolding. The two directories hold *different*
meshes, so the reported ``size`` proves which one was searched:
``meshes/`` is a tetrahedron with extent (0.4, 0.3, 0.2) and ``assets/`` a
pyramid with extent (1.0, 1.0, 1.0).

``TestDocumentOrderDecidesTheSearchDirectory`` pins the precedence the fix must
not disturb: within one ``<compiler>`` element ``meshdir`` overrides
``assetdir``, but a later element overrides an earlier one. Both were already
correct, and both are measured against MuJoCo's own choice. Delegating this
question to ``_mjcf_mesh_subdir`` - which prefers ``meshdir`` across fragments
rather than honouring document order - would answer the second case the way
MuJoCo does not.
"""

from __future__ import annotations

import pathlib

import pytest

from strands_robots.simulation.isaac.loaders import load_mjcf_scene_objects

#: Tetrahedron, full extent (0.4, 0.3, 0.2), 4 vertices.
_TETRA_OBJ = "v 0 0 0\nv 0.4 0 0\nv 0 0.3 0\nv 0 0 0.2\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"
#: Square pyramid, full extent (1.0, 1.0, 1.0), 5 vertices.
_PYRAMID_OBJ = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0.5 0.5 1\nf 1 2 3\nf 1 3 4\nf 1 2 5\nf 2 3 5\nf 3 4 5\nf 4 1 5\n"

_MESHDIR_EXTENT = (0.4, 0.3, 0.2)
_ASSETDIR_EXTENT = (1.0, 1.0, 1.0)
#: The proxy a mesh-less body falls back to, and the value #2459 removed.
_BOX_PROXY_EXTENT = (0.05, 0.05, 0.05)

_BODY = (
    "  <worldbody>\n"
    '    <body name="widget" pos="0.1 0.2 0.3"><freejoint/>\n'
    '      <geom type="mesh" mesh="widget_mesh" group="1"/>\n'
    "    </body>\n"
    "  </worldbody>\n"
)
_ASSET = '  <asset><mesh name="widget_mesh" file="widget.obj"/></asset>\n'


def _write_model(root: pathlib.Path, top: str, **fragments: str) -> str:
    """Write a model tree under ``root`` and return the top file's path.

    Both mesh directories are always populated so a wrong search directory
    resolves to a file that exists, and the reported extent - not an absence -
    is what tells the two apart.
    """
    (root / "meshes").mkdir(parents=True, exist_ok=True)
    (root / "assets").mkdir(parents=True, exist_ok=True)
    (root / "meshes" / "widget.obj").write_text(_TETRA_OBJ, encoding="utf-8")
    (root / "assets" / "widget.obj").write_text(_PYRAMID_OBJ, encoding="utf-8")
    for name, text in fragments.items():
        path = root / name.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    top_path = root / "scene.xml"
    top_path.write_text(top, encoding="utf-8")
    return str(top_path)


def _only_object(scene_path: str):
    """The scene's single object, or a failure naming what came back instead."""
    objects = load_mjcf_scene_objects(scene_path)
    if len(objects) != 1:
        raise AssertionError(f"expected exactly one scene object, got {[o.name for o in objects]}")
    return objects[0]


def _resolved_extent(scene_path: str) -> tuple[float, ...]:
    """The extent the loader reports, or an AssertionError naming the degradation."""
    try:
        obj = _only_object(scene_path)
    except ValueError as exc:
        raise AssertionError(
            f"the loader refused a model MuJoCo compiles: {exc}. A present mesh reported as absent "
            f"loses every object in the scene, not just this one."
        ) from exc
    if obj.mesh_path is None:
        raise AssertionError(
            f"the mesh reference did not resolve, so the object fell back to the "
            f"{tuple(round(v, 4) for v in obj.size)} box proxy #2459 removed"
        )
    return tuple(round(v, 4) for v in obj.size)


# Each fixture is (name, top-file text, fragment files, the extent MuJoCo searches).
_MODEL_GLOBAL_FIXTURES: list[tuple[str, str, dict[str, str], tuple[float, float, float]]] = [
    (
        "asset block in an included fragment",
        '<mujoco>\n  <compiler meshdir="meshes"/>\n  <include file="frag.xml"/>\n' + _BODY + "</mujoco>\n",
        {"frag.xml": "<mujoco>\n" + _ASSET + "</mujoco>\n"},
        _MESHDIR_EXTENT,
    ),
    (
        "meshdir in an included fragment",
        '<mujoco>\n  <include file="frag.xml"/>\n' + _ASSET + _BODY + "</mujoco>\n",
        {"frag.xml": '<mujoco>\n  <compiler meshdir="meshes"/>\n</mujoco>\n'},
        _MESHDIR_EXTENT,
    ),
    (
        "assetdir in an included fragment",
        '<mujoco>\n  <include file="frag.xml"/>\n' + _ASSET + _BODY + "</mujoco>\n",
        {"frag.xml": '<mujoco>\n  <compiler assetdir="assets"/>\n</mujoco>\n'},
        _ASSETDIR_EXTENT,
    ),
    (
        "fragment in a subdirectory, resolved against the model directory",
        '<mujoco>\n  <include file="sub/frag.xml"/>\n' + _ASSET + _BODY + "</mujoco>\n",
        {"sub__frag.xml": '<mujoco>\n  <compiler meshdir="meshes"/>\n</mujoco>\n'},
        _MESHDIR_EXTENT,
    ),
    (
        "nested include",
        '<mujoco>\n  <include file="sub/outer.xml"/>\n' + _ASSET + _BODY + "</mujoco>\n",
        {
            "sub__outer.xml": '<mujoco>\n  <include file="inner.xml"/>\n</mujoco>\n',
            "sub__inner.xml": '<mujoco>\n  <compiler meshdir="meshes"/>\n</mujoco>\n',
        },
        _MESHDIR_EXTENT,
    ),
]

_FIXTURE_IDS = [name.replace(" ", "-") for name, _t, _f, _e in _MODEL_GLOBAL_FIXTURES]


class TestTheMeshRegistryIsReadFromTheWholeModel:
    """A declaration in an ``<include>``d fragment reaches the registry."""

    @pytest.mark.parametrize(("_name", "top", "fragments", "extent"), _MODEL_GLOBAL_FIXTURES, ids=_FIXTURE_IDS)
    def test_an_included_declaration_resolves_the_mesh(self, tmp_path, _name, top, fragments, extent):
        scene = _write_model(tmp_path, top, **fragments)
        extent = _resolved_extent(scene)
        assert extent == pytest.approx(extent)

    def test_a_top_level_declaration_is_unchanged(self, tmp_path):
        """Control: the already-working shape keeps resolving, and to the same extent."""
        scene = _write_model(
            tmp_path,
            '<mujoco>\n  <compiler meshdir="meshes"/>\n' + _ASSET + _BODY + "</mujoco>\n",
        )
        extent = _resolved_extent(scene)
        assert extent == pytest.approx(_MESHDIR_EXTENT)

    def test_a_mesh_declared_nowhere_still_degrades_to_the_box_proxy(self, tmp_path):
        """Control: with no ``<asset>`` entry anywhere the historical fallback stands.

        Scaffolding that references a mesh the model never declares must keep
        loading, so the fix must not turn this into a refusal.
        """
        scene = _write_model(tmp_path, '<mujoco>\n  <compiler meshdir="meshes"/>\n' + _BODY + "</mujoco>\n")
        obj = _only_object(scene)
        assert obj.mesh_path is None
        assert tuple(round(v, 4) for v in obj.size) == pytest.approx(_BOX_PROXY_EXTENT)


class TestDocumentOrderDecidesTheSearchDirectory:
    """Precedence the fix must leave alone, measured against MuJoCo's own choice."""

    def test_one_element_carrying_both_attributes_prefers_meshdir(self, tmp_path):
        scene = _write_model(
            tmp_path,
            '<mujoco>\n  <compiler meshdir="meshes" assetdir="assets"/>\n' + _ASSET + _BODY + "</mujoco>\n",
        )
        extent = _resolved_extent(scene)
        assert extent == pytest.approx(_MESHDIR_EXTENT)

    def test_a_later_element_overrides_an_earlier_one(self, tmp_path):
        """``meshdir`` then ``assetdir`` resolves against ``assetdir``.

        This is the case a ``_mjcf_mesh_subdir`` delegation would get wrong: it
        prefers ``meshdir`` wherever any fragment declares it, while MuJoCo
        honours the last declaration in document order.
        """
        scene = _write_model(
            tmp_path,
            '<mujoco>\n  <compiler meshdir="meshes"/>\n  <compiler assetdir="assets"/>\n'
            + _ASSET
            + _BODY
            + "</mujoco>\n",
        )
        extent = _resolved_extent(scene)
        assert extent == pytest.approx(_ASSETDIR_EXTENT)

    def test_an_earlier_assetdir_is_overridden_by_a_later_meshdir(self, tmp_path):
        scene = _write_model(
            tmp_path,
            '<mujoco>\n  <compiler assetdir="assets"/>\n  <compiler meshdir="meshes"/>\n'
            + _ASSET
            + _BODY
            + "</mujoco>\n",
        )
        extent = _resolved_extent(scene)
        assert extent == pytest.approx(_MESHDIR_EXTENT)


class TestAnUnusableIncludeDoesNotFailTheScene:
    """Following includes must not add a way for the scene load to fail."""

    @pytest.mark.parametrize(
        ("label", "fragments"),
        [
            ("missing", {}),
            ("malformed", {"frag.xml": "<mujoco><asset><mesh file=\n"}),
            ("not-xml", {"frag.xml": "this is not xml at all\n"}),
        ],
    )
    def test_the_rest_of_the_model_still_resolves(self, tmp_path, label, fragments):
        scene = _write_model(
            tmp_path,
            '<mujoco>\n  <compiler meshdir="meshes"/>\n  <include file="frag.xml"/>\n' + _ASSET + _BODY + "</mujoco>\n",
            **fragments,
        )
        extent = _resolved_extent(scene)
        assert extent == pytest.approx(_MESHDIR_EXTENT)

    def test_an_include_cycle_terminates(self, tmp_path):
        """A fragment including its own includer resolves once instead of recursing."""
        scene = _write_model(
            tmp_path,
            '<mujoco>\n  <include file="a.xml"/>\n' + _ASSET + _BODY + "</mujoco>\n",
            **{
                "a.xml": '<mujoco>\n  <compiler meshdir="meshes"/>\n  <include file="b.xml"/>\n</mujoco>\n',
                "b.xml": '<mujoco>\n  <include file="a.xml"/>\n</mujoco>\n',
            },
        )
        extent = _resolved_extent(scene)
        assert extent == pytest.approx(_MESHDIR_EXTENT)

    def test_an_include_without_a_file_attribute_is_ignored(self, tmp_path):
        scene = _write_model(
            tmp_path,
            '<mujoco>\n  <compiler meshdir="meshes"/>\n  <include/>\n' + _ASSET + _BODY + "</mujoco>\n",
        )
        extent = _resolved_extent(scene)
        assert extent == pytest.approx(_MESHDIR_EXTENT)


class TestMuJoCoCompilesEveryFixture:
    """Premise: every model above is one MuJoCo accepts, and it agrees on the mesh.

    Without this the loader's answers could be dismissed as invalid
    scaffolding. MuJoCo's own vertex count identifies which directory it
    searched: the tetrahedron under ``meshes/`` has 4 vertices, the pyramid
    under ``assets/`` has 5.
    """

    @pytest.mark.parametrize(("_name", "top", "fragments", "extent"), _MODEL_GLOBAL_FIXTURES, ids=_FIXTURE_IDS)
    def test_mujoco_compiles_the_model_and_finds_the_same_directory(self, tmp_path, _name, top, fragments, extent):
        mujoco = pytest.importorskip("mujoco")
        scene = _write_model(tmp_path, top, **fragments)
        model = mujoco.MjModel.from_xml_path(scene)
        assert model.nmesh == 1
        expected_vertices = 4 if extent == _MESHDIR_EXTENT else 5
        assert int(model.mesh_vertnum[0]) == expected_vertices

    @pytest.mark.parametrize(
        ("compiler", "expected_vertices"),
        [
            ('<compiler meshdir="meshes" assetdir="assets"/>', 4),
            ('<compiler meshdir="meshes"/>\n  <compiler assetdir="assets"/>', 5),
            ('<compiler assetdir="assets"/>\n  <compiler meshdir="meshes"/>', 4),
        ],
        ids=["one-element-prefers-meshdir", "later-assetdir-wins", "later-meshdir-wins"],
    )
    def test_mujoco_resolves_precedence_in_document_order(self, tmp_path, compiler, expected_vertices):
        mujoco = pytest.importorskip("mujoco")
        scene = _write_model(tmp_path, "<mujoco>\n  " + compiler + "\n" + _ASSET + _BODY + "</mujoco>\n")
        model = mujoco.MjModel.from_xml_path(scene)
        assert int(model.mesh_vertnum[0]) == expected_vertices
