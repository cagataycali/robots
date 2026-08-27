"""A mesh asset declaring a vertex coordinate that is not finite is refused.

:mod:`strands_robots.simulation.isaac.mesh_assets` reads OBJ, ASCII STL,
binary STL and legacy MuJoCo MSH through four parse paths, and two things
downstream *measure* the vertices it returns:
:func:`~strands_robots.simulation.isaac.mesh_assets.mesh_aabb` (whose
``(center, size)`` becomes a scene object's collision proxy) and the
``extent`` :func:`~strands_robots.simulation.isaac.mesh_assets.convert_mesh_to_usd`
authors into the converted USD.

Neither measurement reports a non-finite coordinate. ``min``/``max`` order a
NaN as neither smaller nor larger than anything, so ``mesh_aabb``'s running
comparison drops it and returns the bounds of the vertices that *are*
finite - numerically indistinguishable from a mesh that declared only
those - while the extent is built by ``min`` over an iterable, which keeps a
leading NaN and drops a trailing one. One module, one file, two answers,
decided by declaration order. An infinite coordinate needs no subtlety: the
reported extent is simply unbounded.

The module's stated contract is that nothing here degrades to a silent
default box, and its ASCII-STL sibling refusal exists for precisely this
outcome from the other side ("bounds for geometry the mesh does not carry,
under no error at all"). MuJoCo - the ``.msh`` format's owner and a reader
of the other two - refuses the same input by name, so the disposition
graded here is the format family's own.

The classes below pin, in order: the refusal across all four parse paths and
both non-finite values; that it is one rule rather than four; that both
bounds consumers are covered because it lives at the reader they share; that
it precedes the optional ``pxr`` probe so a caller learns what is wrong with
the asset rather than what is missing from the environment; that a finite
mesh - including extreme-but-finite coordinates - is untouched; that every
pre-existing malformation is still named by its own message; and that MuJoCo
agrees.
"""

from __future__ import annotations

import ast
import inspect
import math
import re
import struct
from pathlib import Path
from typing import Any

import pytest

from strands_robots.simulation.isaac import mesh_assets
from strands_robots.simulation.isaac.mesh_assets import (
    MESH_EXTENSIONS,
    USD_EXTENSIONS,
    convert_mesh_to_usd,
    load_mesh_geometry,
    mesh_aabb,
)

NAN = float("nan")
POS_INF = float("inf")
NEG_INF = float("-inf")

#: A closed tetrahedron with four *distinct* vertices and a face list whose
#: first-appearance order matches the declaration order, so the vertex index
#: a refusal reports is the same in all four paths even though the two STL
#: parsers deduplicate shared vertices.
TETRA: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 2.0, 0.0),
    (0.0, 0.0, 3.0),
)
FACES: tuple[tuple[int, int, int], ...] = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))

#: The refusal's stable phrase - the part every format shares.
REFUSAL = "has a coordinate that is not finite"


# --------------------------------------------------------------------------
# fixture builders: one per parse path, all fed the same vertices + faces
# --------------------------------------------------------------------------
def _obj_bytes(verts: tuple[tuple[float, float, float], ...], faces: tuple[tuple[int, int, int], ...]) -> bytes:
    lines = [f"v {v[0]!r} {v[1]!r} {v[2]!r}" for v in verts]
    lines += [f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}" for t in faces]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _ascii_stl_bytes(verts: tuple[tuple[float, float, float], ...], faces: tuple[tuple[int, int, int], ...]) -> bytes:
    out = ["solid probe"]
    for tri in faces:
        out += ["facet normal 0 0 1", "outer loop"]
        out += [f"vertex {verts[i][0]!r} {verts[i][1]!r} {verts[i][2]!r}" for i in tri]
        out += ["endloop", "endfacet"]
    out.append("endsolid probe")
    return ("\n".join(out) + "\n").encode("utf-8")


def _binary_stl_bytes(verts: tuple[tuple[float, float, float], ...], faces: tuple[tuple[int, int, int], ...]) -> bytes:
    blob = b"\x00" * 80 + struct.pack("<I", len(faces))
    for tri in faces:
        blob += struct.pack("<3f", 0.0, 0.0, 1.0)
        for i in tri:
            blob += struct.pack("<3f", *verts[i])
        blob += struct.pack("<H", 0)
    return blob


def _msh_bytes(verts: tuple[tuple[float, float, float], ...], faces: tuple[tuple[int, int, int], ...]) -> bytes:
    blob = struct.pack("<4i", len(verts), 0, 0, len(faces))
    for v in verts:
        blob += struct.pack("<3f", *v)
    for tri in faces:
        blob += struct.pack("<3i", *tri)
    return blob


#: ``path label -> (builder, on-disk extension)``. Keyed by parse path, not by
#: extension: ``.stl`` reaches two different parsers, chosen by whether the
#: declared triangle count reconciles with the byte length.
PARSE_PATHS: dict[str, tuple[Any, str]] = {
    "obj": (_obj_bytes, ".obj"),
    "ascii-stl": (_ascii_stl_bytes, ".stl"),
    "binary-stl": (_binary_stl_bytes, ".stl"),
    "msh": (_msh_bytes, ".msh"),
}

NON_FINITE = {"nan": NAN, "posinf": POS_INF, "neginf": NEG_INF}


def _with_bad_x(which: int, value: float) -> tuple[tuple[float, float, float], ...]:
    """:data:`TETRA` with vertex ``which``'s x coordinate replaced by ``value``."""
    verts = list(TETRA)
    x, y, z = verts[which]
    verts[which] = (value, y, z)
    return tuple(verts)


def _write(tmp_path: Path, label: str, verts: tuple[tuple[float, float, float], ...], name: str = "asset") -> str:
    builder, ext = PARSE_PATHS[label]
    asset = tmp_path / f"{name}{ext}"
    asset.write_bytes(builder(verts, FACES))
    return str(asset)


def _source_of(func: Any) -> str:
    return inspect.getsource(func)


# --------------------------------------------------------------------------
class TestANonFiniteVertexIsRefusedByEveryParsePath:
    """The regression: every path, both values, wherever the vertex sits."""

    @pytest.mark.parametrize("label", sorted(PARSE_PATHS))
    @pytest.mark.parametrize("value_name", sorted(NON_FINITE))
    @pytest.mark.parametrize("which", [0, 3])
    def test_the_asset_is_refused_rather_than_sized(self, tmp_path, label, value_name, which):
        asset = _write(tmp_path, label, _with_bad_x(which, NON_FINITE[value_name]))
        with pytest.raises(ValueError, match=REFUSAL):
            load_mesh_geometry(asset)

    @pytest.mark.parametrize("label", sorted(PARSE_PATHS))
    @pytest.mark.parametrize("which", [0, 3])
    def test_the_refusal_names_the_offending_vertex(self, tmp_path, label, which):
        asset = _write(tmp_path, label, _with_bad_x(which, NAN))
        with pytest.raises(ValueError) as excinfo:
            load_mesh_geometry(asset)
        found = re.search(r"vertex (\d+) " + REFUSAL, str(excinfo.value))
        assert found is not None, str(excinfo.value)
        assert int(found.group(1)) == which

    @pytest.mark.parametrize("label", sorted(PARSE_PATHS))
    def test_the_refusal_names_the_asset_and_the_value(self, tmp_path, label):
        asset = _write(tmp_path, label, _with_bad_x(0, NAN))
        with pytest.raises(ValueError) as excinfo:
            load_mesh_geometry(asset)
        message = str(excinfo.value)
        assert asset in message
        assert "nan" in message.lower()

    def test_a_text_coordinate_whose_exponent_overflows_is_refused(self, tmp_path):
        # ``float("1e400")`` is ``inf``, so a text asset need not spell a
        # non-finite token to declare one: an out-of-range exponent from an
        # exporter reaches the same place, and used to be measured.
        asset = tmp_path / "overflow.obj"
        asset.write_bytes(_obj_bytes(((1e400, 0.0, 0.0), *TETRA[1:]), FACES))
        with pytest.raises(ValueError, match=REFUSAL):
            load_mesh_geometry(str(asset))


class TestTheFinitenessRuleIsSharedByEveryFormat:
    """One rule at the reader the four parse paths meet, not four copies."""

    def test_the_finiteness_check_appears_once_in_the_module(self):
        source = Path(inspect.getsourcefile(mesh_assets) or "").read_text(encoding="utf-8")
        # The guard's fast path. A second copy would be a per-format rule and
        # could drift from its siblings, which is what this module already
        # went out of its way to avoid for the OBJ keyword handling.
        assert source.count("chain.from_iterable(points)") == 1

    def test_the_guard_follows_the_dispatch_rather_than_preceding_it(self):
        tree = ast.parse(_source_of(load_mesh_geometry).lstrip())
        calls = {}
        guard_line = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id.startswith("_parse_"):
                    calls[node.func.id] = node.lineno
            if isinstance(node, ast.Call) and "from_iterable" in ast.unparse(node):
                guard_line = node.lineno
        assert set(calls) == {"_parse_obj", "_parse_msh", "_parse_stl"}, calls
        assert guard_line is not None
        assert guard_line > max(calls.values()), (guard_line, calls)

    def test_every_parseable_extension_is_covered(self):
        # Derived from the module's own vocabulary, so a fifth format added
        # to MESH_EXTENSIONS without a parse path here fails rather than
        # inheriting an untested tolerance.
        reachable = {ext for _builder, ext in PARSE_PATHS.values()}
        assert reachable == set(MESH_EXTENSIONS)

    @pytest.mark.parametrize("label", sorted(PARSE_PATHS))
    def test_the_wording_does_not_vary_by_format(self, tmp_path, label):
        asset = _write(tmp_path, label, _with_bad_x(0, NAN))
        with pytest.raises(ValueError) as excinfo:
            load_mesh_geometry(asset)
        # Everything after the path and before the value is format-agnostic.
        assert str(excinfo.value).startswith(f"mesh {asset}: vertex 0 {REFUSAL}")


class TestBothBoundsConsumersAreCovered:
    """Placing the guard at the shared reader covers everything that measures."""

    @pytest.mark.parametrize("label", sorted(PARSE_PATHS))
    def test_mesh_aabb_refuses_instead_of_reporting_a_box(self, tmp_path, label):
        asset = _write(tmp_path, label, _with_bad_x(0, NAN))
        with pytest.raises(ValueError, match=REFUSAL):
            mesh_aabb(asset)

    def test_mesh_aabb_no_longer_answers_what_a_finite_sibling_answers(self, tmp_path):
        # The measurement that makes this a defect rather than a nicety: the
        # bad asset used to report the *same* box as the clean one, so
        # nothing distinguished them.
        clean = _write(tmp_path, "msh", TETRA, name="clean")
        bad = _write(tmp_path, "msh", _with_bad_x(0, NAN), name="bad")
        assert mesh_aabb(clean) == ((0.5, 1.0, 1.5), (1.0, 2.0, 3.0))
        with pytest.raises(ValueError, match=REFUSAL):
            mesh_aabb(bad)

    def test_the_usd_conversion_refuses_the_same_asset(self, tmp_path):
        asset = _write(tmp_path, "obj", _with_bad_x(0, NAN))
        with pytest.raises(ValueError, match=REFUSAL):
            convert_mesh_to_usd(asset, cache_dir=str(tmp_path / "cache"))

    def test_the_refusal_leaves_no_cache_entry_to_be_trusted_later(self, tmp_path, monkeypatch):
        # Also exercises the default cache location, which is where a caller
        # that passes no ``cache_dir`` lands.
        monkeypatch.setenv("STRANDS_BASE_DIR", str(tmp_path / "base"))
        asset = _write(tmp_path, "obj", _with_bad_x(0, NAN))
        with pytest.raises(ValueError, match=REFUSAL):
            convert_mesh_to_usd(asset)
        cache = Path(mesh_assets.mesh_usd_cache_dir())
        assert cache.is_dir()
        assert list(cache.glob("*.usda")) == []


class TestTheRefusalPrecedesTheOptionalDependencyProbe:
    """A caller learns what is wrong with the asset, not what the env lacks."""

    def test_the_asset_is_read_before_pxr_is_imported(self):
        tree = ast.parse(_source_of(convert_mesh_to_usd).lstrip())
        read_line = pxr_line = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "load_mesh_geometry":
                read_line = node.lineno
            if isinstance(node, ast.ImportFrom) and node.module == "pxr":
                pxr_line = node.lineno
        assert read_line is not None and pxr_line is not None
        assert read_line < pxr_line, (read_line, pxr_line)

    @pytest.mark.parametrize(
        ("asset_bytes", "expected"),
        [
            pytest.param(_obj_bytes(_with_bad_x(0, NAN), FACES), REFUSAL, id="non-finite-vertex"),
            pytest.param(b"v 0 0 0\nf 1 2 3\n", "out of range", id="face-index-out-of-range"),
            pytest.param(b"v 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", "fewer than 3 components", id="short-vertex"),
        ],
    )
    def test_a_malformed_asset_is_named_rather_than_the_missing_extra(self, tmp_path, asset_bytes, expected):
        # Whether or not ``pxr`` is installed, the malformation is what the
        # caller is told about. Before the read moved ahead of the import,
        # an install without the ``sim-isaac`` extra got an ImportError
        # pointing at ``usd-core`` for an asset that was simply broken.
        asset = tmp_path / "asset.obj"
        asset.write_bytes(asset_bytes)
        with pytest.raises(ValueError, match=expected):
            convert_mesh_to_usd(str(asset), cache_dir=str(tmp_path / "cache"))

    def test_a_usd_input_is_referenced_verbatim_and_never_measured(self, tmp_path):
        # The scope boundary: the rule is about the formats this module
        # parses. A USD asset is handed on untouched, so nothing reads its
        # vertices here and nothing can refuse them.
        already_usd = tmp_path / "scene.usda"
        already_usd.write_text("#usda 1.0\n", encoding="utf-8")
        assert convert_mesh_to_usd(str(already_usd), cache_dir=str(tmp_path / "cache")) == str(already_usd)
        assert set(USD_EXTENSIONS).isdisjoint(MESH_EXTENSIONS)
        # Handed on untouched is not the same as unchecked: a USD input that is
        # not there is still refused, before any of the above applies.
        with pytest.raises(FileNotFoundError, match="mesh asset not found"):
            convert_mesh_to_usd(str(tmp_path / "absent.usda"), cache_dir=str(tmp_path / "cache"))


class TestAFiniteMeshIsUnchanged:
    """The over-reach controls: nothing the module used to accept is lost."""

    @pytest.mark.parametrize("label", sorted(PARSE_PATHS))
    def test_the_clean_tetrahedron_still_parses(self, tmp_path, label):
        points, counts, indices = load_mesh_geometry(_write(tmp_path, label, TETRA))
        assert set(points) == set(TETRA)
        assert counts == [3, 3, 3, 3]
        assert len(indices) == 12

    @pytest.mark.parametrize("label", sorted(PARSE_PATHS))
    def test_the_clean_bounds_are_the_declared_extent(self, tmp_path, label):
        assert mesh_aabb(_write(tmp_path, label, TETRA)) == ((0.5, 1.0, 1.5), (1.0, 2.0, 3.0))

    @pytest.mark.parametrize(
        ("value_name", "value"),
        [
            ("float32-max", 3.4028234663852886e38),
            ("negative-float32-max", -3.4028234663852886e38),
            ("float32-tiny", 1.1754943508222875e-38),
            ("negative-zero", -0.0),
            ("float64-large", 1e300),
        ],
    )
    def test_an_extreme_but_finite_coordinate_is_accepted(self, tmp_path, value_name, value):
        # The guard tests finiteness, not magnitude: a legitimately huge or
        # vanishing coordinate is a position and stays one.
        asset = tmp_path / "extreme.obj"
        asset.write_bytes(_obj_bytes(((value, 0.0, 0.0), *TETRA[1:]), FACES))
        points, _counts, _indices = load_mesh_geometry(str(asset))
        assert points[0] == (value, 0.0, 0.0)
        assert all(math.isfinite(c) for v in points for c in v)


class TestEachMalformationIsStillNamedByItsOwnMessage:
    """The new refusal joins the module's named refusals; it replaces none."""

    @pytest.mark.parametrize(
        ("name", "blob", "expected"),
        [
            pytest.param(
                "sv.obj",
                b"v 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
                "vertex with fewer than 3 components",
                id="obj-short-vertex",
            ),
            pytest.param(
                "nn.obj",
                b"v a b c\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
                "non-numeric vertex component",
                id="obj-non-numeric-vertex",
            ),
            pytest.param(
                "sf.obj", b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2\n", "face with fewer than 3 vertices", id="obj-short-face"
            ),
            pytest.param(
                "nf.obj",
                b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 x\n",
                "non-integer face index",
                id="obj-non-integer-face-index",
            ),
            pytest.param(
                "oor.obj", b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n", "out of range", id="obj-face-index-out-of-range"
            ),
            pytest.param("empty.obj", b"v 0 0 0\nv 1 0 0\nv 0 1 0\n", "no triangle geometry", id="obj-no-faces"),
            pytest.param(
                "sv.stl",
                b"solid p\nfacet normal 0 0 1\nouter loop\nvertex 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid p\n",
                "vertex with fewer than 3 components",
                id="stl-short-vertex",
            ),
            pytest.param(
                "nn.stl",
                b"solid p\nfacet normal 0 0 1\nouter loop\nvertex a b c\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid p\n",
                "non-numeric vertex component",
                id="stl-non-numeric-vertex",
            ),
            pytest.param("empty.stl", b"solid p\nendsolid p\n", "no triangle geometry", id="stl-no-facets"),
            pytest.param(
                "garbage.stl",
                b"not an stl at all, and not long enough to be binary\n",
                "no 'solid' header",
                id="stl-no-header",
            ),
            pytest.param("short.msh", struct.pack("<2i", 4, 0), "16-byte", id="msh-short-header"),
            pytest.param("zero.msh", struct.pack("<4i", 4, 0, 0, 0), "non-positive geometry counts", id="msh-no-faces"),
            pytest.param(
                "neg.msh", struct.pack("<4i", -4, 0, 0, 4), "non-positive geometry counts", id="msh-negative-vertices"
            ),
        ],
    )
    def test_the_malformation_reports_its_own_diagnosis(self, tmp_path, name, blob, expected):
        asset = tmp_path / name
        asset.write_bytes(blob)
        with pytest.raises(ValueError, match=expected) as excinfo:
            load_mesh_geometry(str(asset))
        # ...and is not re-badged as the finiteness refusal.
        assert REFUSAL not in str(excinfo.value)

    def test_a_missing_file_and_a_foreign_extension_are_unchanged(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="mesh asset not found"):
            load_mesh_geometry(str(tmp_path / "absent.obj"))
        foreign = tmp_path / "asset.ply"
        foreign.write_bytes(b"ply\n")
        with pytest.raises(ValueError, match="unsupported mesh format"):
            load_mesh_geometry(str(foreign))


class TestMujocoRefusesTheSameAsset:
    """The premise: refusing is the format family's disposition, not a local one.

    MuJoCo owns ``.msh`` and reads OBJ and binary STL, so it is the available
    oracle for what these formats mean - the same role it plays for the
    ``.msh`` layout elsewhere in this suite. It reports
    ``vertex coordinate N is not finite``.
    """

    @pytest.fixture(autouse=True)
    def _require_mujoco(self):
        pytest.importorskip("mujoco")

    @staticmethod
    def _compile(tmp_path: Path, label: str, verts: tuple[tuple[float, float, float], ...], name: str) -> None:
        import mujoco

        # A directory per fixture: MuJoCo caches a compiled mesh, so reusing
        # one filename for two different files reports the first one twice.
        scene_dir = tmp_path / name
        scene_dir.mkdir()
        builder, ext = PARSE_PATHS[label]
        (scene_dir / f"asset{ext}").write_bytes(builder(verts, FACES))
        (scene_dir / "scene.xml").write_text(
            f'<mujoco><asset><mesh name="m" file="asset{ext}"/></asset>'
            '<worldbody><body><geom type="mesh" mesh="m"/></body></worldbody></mujoco>',
            encoding="utf-8",
        )
        mujoco.MjModel.from_xml_path(str(scene_dir / "scene.xml"))

    @pytest.mark.parametrize("label", ["obj", "binary-stl", "msh"])
    def test_mujoco_compiles_the_finite_tetrahedron(self, tmp_path, label):
        self._compile(tmp_path, label, TETRA, f"clean-{label}")

    @pytest.mark.parametrize("label", ["obj", "binary-stl", "msh"])
    def test_mujoco_refuses_a_non_finite_vertex(self, tmp_path, label):
        with pytest.raises(ValueError, match="not finite"):
            self._compile(tmp_path, label, _with_bad_x(0, NAN), f"nan-{label}")

    @pytest.mark.parametrize("label", ["obj", "binary-stl", "msh"])
    def test_this_module_now_agrees_with_mujoco(self, tmp_path, label):
        asset = _write(tmp_path, label, _with_bad_x(0, NAN))
        with pytest.raises(ValueError, match=REFUSAL):
            load_mesh_geometry(asset)
