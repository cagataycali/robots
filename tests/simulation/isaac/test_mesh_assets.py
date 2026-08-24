"""Unit pins for :mod:`strands_robots.simulation.isaac.mesh_assets` (#2459).

The OBJ/STL parsing + USD conversion is the CPU half of realizing LIBERO
objects as real meshes on the Isaac stage: it must be exercisable with no
Isaac Sim install (parsing is pure stdlib; conversion needs only the
``usd-core`` wheel), and it must fail loud - a missing or empty asset is an
error, never a silent default box, because a box standing in for a bowl is
exactly the eval-integrity defect the feature removes.
"""

from __future__ import annotations

import struct

import pytest

from strands_robots.simulation.isaac.mesh_assets import (
    MESH_EXTENSIONS,
    USD_EXTENSIONS,
    convert_mesh_to_usd,
    load_mesh_geometry,
    mesh_aabb,
)

_TETRA_OBJ = "v 0 0 0\nv 1 0 0\nv 0 2 0\nv 0 0 3\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"


def _binary_stl(tri_vertices: list[tuple[tuple[float, float, float], ...]]) -> bytes:
    """Assemble a well-formed binary STL from triangles of xyz tuples."""
    blob = b"\x00" * 80 + struct.pack("<I", len(tri_vertices))
    for tri in tri_vertices:
        rec = [0.0, 0.0, 0.0]
        for vert in tri:
            rec.extend(vert)
        blob += struct.pack("<12f", *rec) + struct.pack("<H", 0)
    return blob


# MuJoCo's ``LoadMSH`` refuses ``nvertex < 4``, so every fixture below carries
# at least four - a three-vertex .msh is a file the format's owner calls
# invalid, and a reader graded only against such a file is graded against
# nothing MuJoCo would ever hand it.
_MSH_TETRA_VERTS = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0)]
_MSH_TETRA_FACES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def _binary_msh(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    *,
    nnormal: int = 0,
    ntexcoord: int = 0,
) -> bytes:
    """Assemble a legacy MuJoCo binary mesh: the format LIBERO's compiled
    scenes declare for the bowl/plate visual assets."""
    blob = struct.pack("<4i", len(vertices), nnormal, ntexcoord, len(faces))
    for v in vertices:
        blob += struct.pack("<3f", *v)
    blob += b"\x00" * (4 * (3 * nnormal + 2 * ntexcoord))
    for f in faces:
        blob += struct.pack("<3i", *f)
    return blob


class TestLoadMeshGeometry:
    def test_obj_vertices_and_faces(self, tmp_path):
        asset = tmp_path / "tetra.obj"
        asset.write_text(_TETRA_OBJ, encoding="utf-8")
        points, counts, indices = load_mesh_geometry(str(asset))
        assert len(points) == 4
        assert counts == [3, 3, 3, 3]
        assert max(indices) == 3 and min(indices) == 0

    def test_obj_slash_syntax_and_quads(self, tmp_path):
        # ``i/t/n`` face refs and n-gon faces both come from real exports;
        # only the vertex index is consumed and polygons are kept as n-gons.
        asset = tmp_path / "quad.obj"
        asset.write_text(
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvn 0 0 1\nf 1/1/1 2/1/1 3/1/1 4/1/1\n",
            encoding="utf-8",
        )
        points, counts, indices = load_mesh_geometry(str(asset))
        assert len(points) == 4
        assert counts == [4]
        assert indices == [0, 1, 2, 3]

    def test_obj_negative_indices(self, tmp_path):
        asset = tmp_path / "neg.obj"
        asset.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n", encoding="utf-8")
        _points, counts, indices = load_mesh_geometry(str(asset))
        assert counts == [3]
        assert indices == [0, 1, 2]

    def test_obj_out_of_range_index_is_an_error(self, tmp_path):
        asset = tmp_path / "bad.obj"
        asset.write_text("v 0 0 0\nv 1 0 0\nf 1 2 3\n", encoding="utf-8")
        with pytest.raises(ValueError, match="out of range"):
            load_mesh_geometry(str(asset))

    def test_binary_stl(self, tmp_path):
        asset = tmp_path / "part.stl"
        tri1 = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        tri2 = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        asset.write_bytes(_binary_stl([tri1, tri2]))
        points, counts, indices = load_mesh_geometry(str(asset))
        # Shared vertices are deduplicated: 6 raw corners -> 4 unique.
        assert len(points) == 4
        assert counts == [3, 3]
        assert len(indices) == 6

    def test_ascii_stl(self, tmp_path):
        asset = tmp_path / "part.stl"
        asset.write_text(
            "solid part\n"
            " facet normal 0 0 1\n"
            "  outer loop\n"
            "   vertex 0 0 0\n   vertex 1 0 0\n   vertex 0 1 0\n"
            "  endloop\n"
            " endfacet\n"
            "endsolid part\n",
            encoding="utf-8",
        )
        points, counts, _indices = load_mesh_geometry(str(asset))
        assert len(points) == 3
        assert counts == [3]

    def test_binary_msh(self, tmp_path):
        # The format LIBERO's robosuite-compiled MJCFs reference for the
        # bowl/plate VISUAL meshes - the exact objects #2459 names - so it
        # must parse, or those objects stay box proxies.
        asset = tmp_path / "bowl_vis.msh"
        verts = _MSH_TETRA_VERTS
        faces = _MSH_TETRA_FACES
        asset.write_bytes(_binary_msh(verts, faces, nnormal=4, ntexcoord=4))
        points, counts, indices = load_mesh_geometry(str(asset))
        assert points == verts
        assert counts == [3, 3, 3, 3]
        assert indices == [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3]

    def test_msh_truncated_is_an_error(self, tmp_path):
        # Declared counts must reconcile with the byte length; a truncated
        # file must not parse as garbage geometry.
        asset = tmp_path / "torn.msh"
        blob = _binary_msh(_MSH_TETRA_VERTS, [(0, 1, 2)])
        asset.write_bytes(blob[:-4])
        with pytest.raises(ValueError, match="truncated"):
            load_mesh_geometry(str(asset))

    def test_msh_out_of_range_face_index_is_an_error(self, tmp_path):
        asset = tmp_path / "bad.msh"
        asset.write_bytes(_binary_msh(_MSH_TETRA_VERTS, [(0, 1, 4)]))
        with pytest.raises(ValueError, match="out of range"):
            load_mesh_geometry(str(asset))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_mesh_geometry(str(tmp_path / "nope.obj"))

    def test_unsupported_extension_raises(self, tmp_path):
        asset = tmp_path / "part.dae"
        asset.write_bytes(b"\x00")
        with pytest.raises(ValueError, match="unsupported mesh format"):
            load_mesh_geometry(str(asset))

    def test_empty_mesh_raises(self, tmp_path):
        # An asset with no faces renders nothing; downstream would misread
        # the blank as a scene property, so the parse is where it fails.
        asset = tmp_path / "empty.obj"
        asset.write_text("v 0 0 0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no triangle geometry"):
            load_mesh_geometry(str(asset))


class TestObjKeywordsAreWhitespaceDelimited:
    """An OBJ keyword may be followed by a tab as legitimately as by a space.

    OBJ is a whitespace-delimited format, and MuJoCo - the reference reader
    for every asset this module is pointed at - reads the tab form and the
    space form identically. Recognising the keyword by a ``"v "`` prefix
    skipped a tab-separated vertex line, which either dropped the vertex with
    nothing raised (leaving :func:`mesh_aabb` reporting bounds the file never
    declared, and so a collision proxy the MJCF scene loader derives from
    those bounds) or pushed a later face reference out of range and blamed
    the face for a vertex the parser had never recognised.

    The ASCII-STL parser in this same module always read its keyword off the
    split fields, so the two parsers answered the same question two ways;
    ``test_the_ascii_stl_parser_already_tolerated_a_tab`` pins that half.
    """

    def test_a_tab_separated_obj_parses_as_the_space_separated_one(self, tmp_path):
        spaces = tmp_path / "spaces.obj"
        spaces.write_text(_TETRA_OBJ, encoding="utf-8")
        tabs = tmp_path / "tabs.obj"
        tabs.write_text(_TETRA_OBJ.replace("v ", "v\t").replace("f ", "f\t"), encoding="utf-8")

        reference = load_mesh_geometry(str(spaces))
        try:
            measured = load_mesh_geometry(str(tabs))
        except ValueError as refused:
            raise AssertionError(
                "the same tetrahedron written with a tab after each v/f keyword was refused "
                f"({refused}), and the refusal names the face rather than the vertex lines the "
                "parser did not recognise - MuJoCo reads both forms identically"
            ) from refused
        assert measured == reference
        assert mesh_aabb(str(tabs)) == mesh_aabb(str(spaces))

    def test_a_tab_vertex_line_is_not_dropped_from_the_bounds(self, tmp_path):
        # The 9-unit spike is declared on a tab line and referenced by no
        # face, so dropping it raises nothing at all: the asset simply
        # reports the bounds of the three vertices that were recognised.
        # mesh_aabb is what the MJCF scene loader turns into a mesh-only
        # body's collision proxy, so a silently flat z is a silently flat box.
        asset = tmp_path / "spike.obj"
        asset.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv\t0 0 9\nf 1 2 3\n", encoding="utf-8")

        points, counts, _indices = load_mesh_geometry(str(asset))
        _center, size = mesh_aabb(str(asset))
        assert len(points) == 4, f"the tab-separated vertex was dropped: {points}"
        assert counts == [3]
        assert size[2] == pytest.approx(9.0), f"bounds the file never declared: z extent {size[2]}"

    def test_a_mixed_separator_obj_keeps_every_vertex(self, tmp_path):
        # One writer emitting both separators is the case that shifts every
        # subsequent 1-based face reference onto the wrong vertex.
        asset = tmp_path / "mixed.obj"
        asset.write_text(
            "v 0 0 0\nv\t1 0 0\nv 0 2 0\nv\t0 0 3\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n",
            encoding="utf-8",
        )
        points, counts, indices = load_mesh_geometry(str(asset))
        assert len(points) == 4
        assert counts == [3, 3, 3, 3]
        assert indices == [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3]

    def test_vertex_normals_and_texcoords_are_still_not_vertices(self, tmp_path):
        # ``vn`` / ``vt`` are their own keywords, not ``v`` plus a separator:
        # reading the keyword off the split must not widen the vertex rule
        # into them, in either the space or the tab spelling.
        asset = tmp_path / "attrs.obj"
        asset.write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nvt 0 0\nvn\t0 1 0\nvt\t1 0\nf 1 2 3\n",
            encoding="utf-8",
        )
        points, counts, indices = load_mesh_geometry(str(asset))
        assert len(points) == 3
        assert counts == [3]
        assert indices == [0, 1, 2]

    def test_blank_and_comment_lines_are_ignored(self, tmp_path):
        asset = tmp_path / "commented.obj"
        asset.write_text(
            "# Blender v3.2.2 OBJ File\n\nmtllib x.mtl\no Cube\n\t\n" + _TETRA_OBJ,
            encoding="utf-8",
        )
        _points, counts, _indices = load_mesh_geometry(str(asset))
        assert counts == [3, 3, 3, 3]

    def test_the_ascii_stl_parser_already_tolerated_a_tab(self, tmp_path):
        # Passes either way: the sibling parser in this module has always
        # read its keyword off the split, which is why one module answered
        # "is a tab a separator" two different ways.
        asset = tmp_path / "part.stl"
        asset.write_text(
            "solid part\nfacet normal 0 0 1\nouter loop\n"
            "vertex\t0 0 0\nvertex\t1 0 0\nvertex\t0 1 0\n"
            "endloop\nendfacet\nendsolid part\n",
            encoding="utf-8",
        )
        points, counts, _indices = load_mesh_geometry(str(asset))
        assert len(points) == 3
        assert counts == [3]


class TestAnUnterminatedAsciiStlFacetIsRefused:
    """A facet an ASCII STL never closes must not parse as a smaller mesh.

    The parser flushes a facet on its ``endfacet``, so vertices read before an
    ``endfacet`` that never arrives stay in ``points`` - and therefore in the
    bounds :func:`mesh_aabb` reports and the ``extent``
    :func:`convert_mesh_to_usd` authors - while the triangle they form reaches
    no face. Accepting that returns a mesh whose bounds describe geometry it
    does not carry, with nothing raised at all: the silent-proxy failure mode
    #2459 removed the default box proxy for.

    The binary STL and legacy MSH parsers in this module both refuse a
    truncated file by reconciling their declared counts against the byte
    length. An ASCII STL declares no counts, so the open facet is the only
    thing there is to reconcile - which is why this is the same refusal rather
    than a new rule.
    """

    # Two triangles sharing an edge, the second one 9 units tall, written so
    # each line number below is the one the refusals name.
    _CLOSED_FACET = "facet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\n"
    _SPIKE_BODY = "facet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 0 9\nendloop\n"

    def test_a_well_formed_two_facet_file_is_unchanged(self, tmp_path):
        # The control the three refusals below are measured against: with both
        # ``endfacet`` lines present this geometry parses, and the 9-unit
        # triangle is real - so there is something for the others to lose.
        asset = tmp_path / "closed.stl"
        asset.write_text(
            "solid part\n" + self._CLOSED_FACET + self._SPIKE_BODY + "endfacet\nendsolid part\n",
            encoding="utf-8",
        )
        points, counts, _indices = load_mesh_geometry(str(asset))
        _center, size = mesh_aabb(str(asset))
        assert len(points) == 4
        assert counts == [3, 3]
        assert size[2] == pytest.approx(9.0)

    def test_the_last_facet_without_an_endfacet_is_refused(self, tmp_path):
        # The headline: the file ends mid-facet, so the spike triangle reaches
        # no face while its vertices stay in the bounds. Nothing raised, and
        # mesh_aabb reported the closed file's extent for a one-face mesh.
        # Shaped like the OBJ tab test above: the diagnosis sits on the branch
        # that means the parser got it wrong, which for this file is acceptance.
        asset = tmp_path / "truncated.stl"
        asset.write_text(
            "solid part\n" + self._CLOSED_FACET + self._SPIKE_BODY + "endsolid part\n",
            encoding="utf-8",
        )
        try:
            points, counts, _indices = load_mesh_geometry(str(asset))
        except ValueError as refused:
            message = str(refused)
        else:
            _center, size = mesh_aabb(str(asset))
            raise AssertionError(
                f"an unterminated facet parsed: {len(counts)} of 2 faces from {len(points)} vertices, "
                f"and the z extent is {size[2]} - the bounds of a triangle the mesh does not carry"
            )
        assert "facet left unterminated" in message
        # Line 11 is the open facet's first vertex; the file's last line is 15.
        assert ":11:" in message
        assert "the end of the file" in message

    def test_the_only_facet_unterminated_is_not_reported_as_an_empty_asset(self, tmp_path):
        # Refused before this change too, but as "no triangle geometry
        # (empty vertices/faces)" - for a file declaring a facet with three
        # vertices, which sends a reader looking for the wrong defect.
        asset = tmp_path / "single.stl"
        asset.write_text("solid part\n" + self._SPIKE_BODY + "endsolid part\n", encoding="utf-8")
        with pytest.raises(ValueError, match="facet left unterminated") as refused:
            load_mesh_geometry(str(asset))
        message = str(refused.value)
        assert ":4:" in message
        assert "no triangle geometry" not in message

    def test_the_refusal_names_the_open_facet_not_the_line_that_closes_the_next_one(self, tmp_path):
        # A facet keyword arriving before the previous ``endfacet`` used to
        # accumulate both facets' vertices into one, so the file was refused
        # for "6 vertices (expected 3)" at line 14 - an arity no facet in the
        # file declares, blaming the one line that is correct.
        asset = tmp_path / "mid.stl"
        asset.write_text(
            "solid part\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\n"
            + self._SPIKE_BODY
            + "endfacet\nendsolid part\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="facet left unterminated") as refused:
            load_mesh_geometry(str(asset))
        message = str(refused.value)
        assert ":4:" in message, f"the open facet's first vertex is on line 4: {message}"
        assert "the facet starting on line 8" in message
        assert "6 vertices" not in message

    def test_a_facet_that_really_declares_four_vertices_still_reports_its_arity(self, tmp_path):
        # The pre-existing refusal for a genuinely malformed facet - one that
        # opens and closes around four vertices - must survive: the new guard
        # covers an absent ``endfacet``, not a wrong vertex count.
        asset = tmp_path / "quad_facet.stl"
        asset.write_text(
            "solid part\nfacet normal 0 0 1\nouter loop\n"
            "vertex 0 0 0\nvertex 1 0 0\nvertex 1 1 0\nvertex 0 1 0\n"
            "endloop\nendfacet\nendsolid part\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=r"facet with 4 vertices \(expected 3\)"):
            load_mesh_geometry(str(asset))

    def test_the_binary_sibling_already_refuses_a_truncated_file(self, tmp_path):
        # Passes either way, and is the reason this refusal is not a new rule:
        # the binary parser reconciles its declared triangle count against the
        # byte length, so a file cut mid-record is refused rather than read as
        # the triangles that survived.
        asset = tmp_path / "torn.stl"
        tri1 = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        tri2 = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 9.0))
        asset.write_bytes(_binary_stl([tri1, tri2])[:-50])
        with pytest.raises(ValueError, match="neither a well-formed binary STL nor an ASCII one"):
            load_mesh_geometry(str(asset))


class TestMeshAabb:
    def test_full_extents_and_center(self, tmp_path):
        asset = tmp_path / "tetra.obj"
        asset.write_text(_TETRA_OBJ, encoding="utf-8")
        center, size = mesh_aabb(str(asset))
        assert center == (0.5, 1.0, 1.5)
        assert size == (1.0, 2.0, 3.0)

    def test_scale_is_applied(self, tmp_path):
        asset = tmp_path / "tetra.obj"
        asset.write_text(_TETRA_OBJ, encoding="utf-8")
        center, size = mesh_aabb(str(asset), scale=(2.0, 1.0, 0.5))
        assert center == (1.0, 1.0, 0.75)
        assert size == (2.0, 2.0, 1.5)


class TestConvertMeshToUsd:
    """Needs ``pxr`` (the ``usd-core`` wheel from the ``sim-isaac`` extra)."""

    @pytest.fixture(autouse=True)
    def _require_pxr(self):
        pytest.importorskip("pxr")

    def test_authors_a_referenceable_usd(self, tmp_path):
        from pxr import Usd, UsdGeom  # type: ignore[import-not-found]

        asset = tmp_path / "tetra.obj"
        asset.write_text(_TETRA_OBJ, encoding="utf-8")
        out = convert_mesh_to_usd(str(asset), cache_dir=str(tmp_path / "cache"))
        stage = Usd.Stage.Open(out)
        prim = stage.GetDefaultPrim()
        assert prim.IsValid() and prim.IsA(UsdGeom.Mesh)
        mesh = UsdGeom.Mesh(prim)
        assert len(mesh.GetPointsAttr().Get()) == 4
        assert list(mesh.GetFaceVertexCountsAttr().Get()) == [3, 3, 3, 3]

    def test_conversion_is_cached_by_content(self, tmp_path):
        asset = tmp_path / "tetra.obj"
        asset.write_text(_TETRA_OBJ, encoding="utf-8")
        cache = str(tmp_path / "cache")
        first = convert_mesh_to_usd(str(asset), cache_dir=cache)
        second = convert_mesh_to_usd(str(asset), cache_dir=cache)
        assert first == second
        # Same bytes under a different name hit the same cache entry.
        copy = tmp_path / "renamed.obj"
        copy.write_text(_TETRA_OBJ, encoding="utf-8")
        assert convert_mesh_to_usd(str(copy), cache_dir=cache) == first

    def test_usd_input_is_passed_through(self, tmp_path):
        asset = tmp_path / "tetra.obj"
        asset.write_text(_TETRA_OBJ, encoding="utf-8")
        out = convert_mesh_to_usd(str(asset), cache_dir=str(tmp_path / "cache"))
        assert convert_mesh_to_usd(out, cache_dir=str(tmp_path / "other")) == out

    def test_missing_usd_input_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            convert_mesh_to_usd(str(tmp_path / "nope.usd"))

    def test_no_torn_cache_entry_on_failure(self, tmp_path):
        # A conversion that fails must leave nothing at the cache path a
        # later call would trust (the atomic-rename contract).
        asset = tmp_path / "bad.obj"
        asset.write_text("v 0 0 0\nf 1 2 3\n", encoding="utf-8")  # out-of-range face
        cache = tmp_path / "cache"
        with pytest.raises(ValueError):
            convert_mesh_to_usd(str(asset), cache_dir=str(cache))
        assert (
            not any(p.suffix == ".usda" and not p.name.startswith(".") for p in cache.glob("*")) or not cache.exists()
        )


def test_extension_vocabularies_are_disjoint():
    assert not set(MESH_EXTENSIONS) & set(USD_EXTENSIONS)


class TestTheMshLayoutIsGradedAgainstMujoco:
    """Grade the ``.msh`` reader against MuJoCo, the format's only owner.

    :func:`_binary_msh` writes the header fields in the order
    :func:`load_mesh_geometry` reads them and lays the blocks out in the order
    it walks them, so a round trip through it cannot tell a correct layout
    from a mirrored one: a reader that swapped two header fields would be
    graded by a fixture that swapped them too, and both halves would move
    together. The legacy binary mesh has no published specification beyond
    MuJoCo's own ``LoadMSH``, so these hand the same bytes to both and require
    the geometry to agree.

    MuJoCo does not store the authored frame: it recentres a mesh's vertices
    on the mesh frame and rotates them onto its principal inertia axes, so
    ``mesh_vert``'s extent is a permutation of the file's.
    ``mesh_pos + R(mesh_quat) @ mesh_vert`` recovers the authored frame, and
    that recovery is what makes the comparison exact rather than
    up-to-a-rigid-motion.
    """

    @staticmethod
    def _mujoco_file_frame(asset, tmp_path):
        """Read ``asset`` with MuJoCo; return its vertices in the file frame."""
        mujoco = pytest.importorskip("mujoco")
        import numpy as np

        xml = tmp_path / "one_mesh.xml"
        xml.write_text(
            f'<mujoco><asset><mesh name="m" file="{asset.name}"/></asset>'
            '<worldbody><body><geom type="mesh" mesh="m"/></body></worldbody></mujoco>',
            encoding="utf-8",
        )
        model = mujoco.MjModel.from_xml_path(str(xml))
        adr, num = int(model.mesh_vertadr[0]), int(model.mesh_vertnum[0])
        stored = np.asarray(model.mesh_vert[adr : adr + num], dtype=float)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, np.asarray(model.mesh_quat[0], dtype=float))
        authored = (rot.reshape(3, 3) @ stored.T).T + np.asarray(model.mesh_pos[0], dtype=float)
        return authored, int(model.mesh_facenum[0])

    @staticmethod
    def _vertex_set_delta(one, other):
        """Worst nearest-neighbour distance between two vertex sets.

        A lexicographic sort is not a canonical form here: MuJoCo stores
        vertices as ``float32``, so recovering the authored frame leaves about
        1e-6 of error, and that is enough to flip the order of two vertices
        whose leading coordinate ties - which reads as a large disagreement
        between sets that are in fact identical. Match on distance instead.
        """
        import numpy as np

        one = np.asarray(one, dtype=float)
        other = np.asarray(other, dtype=float)
        assert one.shape == other.shape, f"vertex counts differ: {one.shape} vs {other.shape}"
        worst = 0.0
        for point in one:
            worst = max(worst, float(np.linalg.norm(other - point, axis=1).min()))
        for point in other:
            worst = max(worst, float(np.linalg.norm(one - point, axis=1).min()))
        return worst

    def _assert_agrees_with_mujoco(self, tmp_path, *, nnormal, ntexcoord):
        asset = tmp_path / "graded.msh"
        asset.write_bytes(_binary_msh(_MSH_TETRA_VERTS, _MSH_TETRA_FACES, nnormal=nnormal, ntexcoord=ntexcoord))
        authored, facenum = self._mujoco_file_frame(asset, tmp_path)
        points, counts, _indices = load_mesh_geometry(str(asset))
        # No extent premise is needed here: MuJoCo runs qhull over the
        # vertices and refuses a coplanar mesh outright, so any fixture that
        # reaches this comparison already has real extent on every axis -
        # pinned by test_mujoco_refuses_a_coplanar_msh below.
        delta = self._vertex_set_delta(points, authored)
        assert delta < 1e-4, (
            f"the reader and MuJoCo disagree about the same bytes by {delta:.3g}: "
            f"reader {points}, MuJoCo (authored frame) {authored.tolist()}"
        )
        assert len(counts) == facenum, f"faces: reader {len(counts)}, MuJoCo {facenum}"

    def test_a_file_declaring_normals_and_texcoords_reads_as_mujoco_reads_it(self, tmp_path):
        self._assert_agrees_with_mujoco(tmp_path, nnormal=4, ntexcoord=4)

    def test_a_file_declaring_neither_reads_as_mujoco_reads_it(self, tmp_path):
        # Block order: with both optional blocks absent the face records sit
        # immediately after the vertices, so a reader that walked the blocks
        # in a different order lands on the faces here and not above.
        self._assert_agrees_with_mujoco(tmp_path, nnormal=0, ntexcoord=0)

    def test_a_file_declaring_texcoords_but_no_normals_reads_as_mujoco_reads_it(self, tmp_path):
        # Header field order: normals are three floats per vertex and
        # texcoords two, so a reader that swapped those two header fields
        # computes a different total length for these same bytes, and one of
        # the two readings lands the face block somewhere the other does not.
        self._assert_agrees_with_mujoco(tmp_path, nnormal=0, ntexcoord=4)

    def test_mujoco_refuses_a_three_vertex_msh(self, tmp_path):
        # The constraint the fixture repair above rests on: MuJoCo's LoadMSH
        # refuses ``nvertex < 4``, so a three-vertex .msh is not a file the
        # format's owner would ever hand this reader.
        pytest.importorskip("mujoco")
        asset = tmp_path / "too_few.msh"
        asset.write_bytes(_binary_msh([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [(0, 1, 2)]))
        with pytest.raises(Exception) as refused:  # noqa: B017 - the binding's own error type
            self._mujoco_file_frame(asset, tmp_path)
        assert "size" in str(refused.value).lower(), str(refused.value)

    def test_mujoco_refuses_a_coplanar_msh(self, tmp_path):
        # Why the comparisons above carry no extent premise: a degenerate
        # fixture could not distinguish one block reading from another, and
        # MuJoCo rules it out for us - it runs qhull over the vertices, which
        # a coplanar set has no hull for.
        pytest.importorskip("mujoco")
        asset = tmp_path / "flat.msh"
        flat = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.5, 1.0, 0.0)]
        asset.write_bytes(_binary_msh(flat, _MSH_TETRA_FACES))
        with pytest.raises(Exception) as refused:  # noqa: B017 - the binding's own error type
            self._mujoco_file_frame(asset, tmp_path)
        # Both spellings across the declared mujoco range: 3.5 reports a bare
        # "qhull error", 3.12 "coplanar vertices, cannot compute convex hull".
        message = str(refused.value).lower()
        assert "qhull" in message or "coplanar" in message, message

    def test_the_shared_fixture_is_one_mujoco_accepts(self, tmp_path):
        # Non-vacuity: if MuJoCo refused the shared fixture, every comparison
        # above would report a skip or an error rather than an agreement.
        assert len(_MSH_TETRA_VERTS) >= 4
        authored, facenum = self._mujoco_file_frame(self._written(tmp_path, nnormal=4, ntexcoord=4), tmp_path)
        assert len(authored) == len(_MSH_TETRA_VERTS)
        assert facenum == len(_MSH_TETRA_FACES)

    @staticmethod
    def _written(tmp_path, *, nnormal, ntexcoord):
        asset = tmp_path / "shared.msh"
        asset.write_bytes(_binary_msh(_MSH_TETRA_VERTS, _MSH_TETRA_FACES, nnormal=nnormal, ntexcoord=ntexcoord))
        return asset
