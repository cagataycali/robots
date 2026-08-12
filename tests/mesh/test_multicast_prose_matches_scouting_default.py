"""Mesh prose must describe the scouting default the config actually emits.

:func:`strands_robots.mesh._zenoh_config.scouting_block` ships
``scouting/multicast/enabled = false`` and ``scouting/gossip/enabled = true``:
gossip plus explicit ``ZENOH_CONNECT`` endpoints is the default posture, and
LAN multicast is an opt-in via ``STRANDS_MESH_MULTICAST=true``.  Prose that
presents multicast as the mechanism sends an operator looking for a peer that
cross-host discovery was never going to find.

:mod:`strands_robots.mesh.session` stated both readings at once -- its module
docstring called multicast automatic while ``_build_config``'s docstring
recorded "gossip on, multicast off by default" -- so the drift was already
visible inside one module.

The rule below is deliberately block-level rather than sentence-level: a
paragraph explaining *why* multicast is dangerous mentions it several times
without claiming it is on, so the whole block is what must carry the opt-in
marker.  That is the same reasoning
:meth:`strands_robots.mesh.core.Mesh.start`'s multicast warning already
applies to the value it reports -- it reads the flag through the helper the
session config uses so the warning "can never disagree with the value
actually applied".  Here the prose is checked against the block that builds
that value, so it cannot disagree either.

Scope is the mesh package, the README and the shipped diagrams.
``strands_robots/device_connect`` is excluded on measured grounds: it runs the
external ``device_connect_edge`` runtime and never touches
:func:`~strands_robots.mesh._zenoh_config.scouting_block`, so its D2D prose
describes a different transport's defaults (pinned below).
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re

from strands_robots.mesh import _zenoh_config as zc

#: Any spelling of the multicast discovery channel.
_MULTICAST = re.compile(r"multicast", re.IGNORECASE)

#: A block mentioning multicast must also mark it opt-in or state it is off.
_OPT_IN = re.compile(
    r"STRANDS_MESH_MULTICAST"
    r"|opt[- ]in"
    r"|opts? (?:back )?into"
    r"|off by default"
    r"|off \(default\)"
    r"|Default ``false``",
    re.IGNORECASE,
)

_MESH_DIR = pathlib.Path(inspect.getfile(zc)).parent
_REPO_ROOT = _MESH_DIR.parent.parent


def _python_blocks(source: str) -> list[tuple[int, str]]:
    """Return ``(lineno, text)`` for every docstring and comment run.

    A docstring is one block even when it contains blank lines, so a
    multi-paragraph explanation is judged as a whole.
    """
    blocks: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            text = ast.get_docstring(node)
            if text:
                blocks.append((node.body[0].lineno, text))

    run: list[str] = []
    start: int | None = None
    for lineno, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            start = lineno if start is None else start
            run.append(stripped.lstrip("#").strip())
            continue
        if run and start is not None:
            blocks.append((start, "\n".join(run)))
        run, start = [], None
    if run and start is not None:
        blocks.append((start, "\n".join(run)))
    return blocks


def _markdown_blocks(source: str) -> list[tuple[int, str]]:
    """Return ``(lineno, text)`` for each blank-line-separated paragraph."""
    blocks: list[tuple[int, str]] = []
    run: list[str] = []
    start: int | None = None
    for lineno, line in enumerate(source.splitlines(), 1):
        if line.strip():
            start = lineno if start is None else start
            run.append(line)
            continue
        if run and start is not None:
            blocks.append((start, "\n".join(run)))
        run, start = [], None
    if run and start is not None:
        blocks.append((start, "\n".join(run)))
    return blocks


def _svg_blocks(source: str) -> list[tuple[int, str]]:
    """Return the diagram's whole text content as a single block."""
    return [(1, " ".join(re.sub(r"<[^>]*>", " ", source).split()))]


def _blocks(path: pathlib.Path) -> list[tuple[int, str]]:
    source = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        return _python_blocks(source)
    if path.suffix == ".md":
        return _markdown_blocks(source)
    return _svg_blocks(source)


def _scoped_files() -> list[pathlib.Path]:
    """Every file whose prose describes this mesh's discovery posture."""
    return [
        *sorted(_MESH_DIR.rglob("*.py")),
        _REPO_ROOT / "README.md",
        *sorted((_REPO_ROOT / "examples").rglob("*.svg")),
        *sorted((_REPO_ROOT / "docs" / "assets").rglob("*.svg")),
    ]


def _unmarked_multicast_blocks() -> list[tuple[pathlib.Path, int, str]]:
    """Return blocks naming multicast without marking it opt-in or off."""
    found: list[tuple[pathlib.Path, int, str]] = []
    for path in _scoped_files():
        for lineno, block in _blocks(path):
            if _MULTICAST.search(block) and not _OPT_IN.search(block):
                found.append((path, lineno, " ".join(block.split())[:160]))
    return found


def _paragraph_containing(path: pathlib.Path, needle: str) -> str:
    for _lineno, block in _blocks(path):
        if needle in block:
            return block
    raise AssertionError(f"{path}: no block contains {needle!r}")


class TestTheShippedDefaultIsGossipOnly:
    """The config default is what the prose rule below is keyed on."""

    def test_multicast_is_off_and_gossip_is_on_by_default(self) -> None:
        emitted = dict(zc.scouting_block())
        assert emitted["scouting/multicast/enabled"] == "false", (
            "the prose rule in this module assumes multicast ships off; if the "
            "default flips, update the mesh prose and this guard together"
        )
        assert emitted["scouting/gossip/enabled"] == "true"


class TestNoMeshProseClaimsMulticastByDefault:
    """Every multicast mention in scoped prose marks it opt-in or off."""

    def test_no_block_presents_multicast_as_the_default_mechanism(self) -> None:
        unmarked = _unmarked_multicast_blocks()
        rendered = "\n".join(
            f"  {path.relative_to(_REPO_ROOT)}:{lineno}\n    {snippet}" for path, lineno, snippet in unmarked
        )
        assert not unmarked, (
            "multicast scouting is off by default (scouting_block emits "
            "scouting/multicast/enabled=false), so prose naming it must say it "
            "is opt-in -- name STRANDS_MESH_MULTICAST or state it is off by "
            f"default in these blocks:\n{rendered}"
        )

    def test_the_scan_reaches_real_prose(self) -> None:
        files = _scoped_files()
        assert len(files) > 20, f"scope collapsed to {len(files)} files"
        assert all(path.exists() for path in files), "scoped file missing"
        blocks = sum(len(_blocks(path)) for path in files)
        assert blocks > 200, f"only {blocks} prose blocks parsed"

    def test_multicast_is_still_discussed_somewhere_in_scope(self) -> None:
        mentions = [
            (path, lineno) for path in _scoped_files() for lineno, block in _blocks(path) if _MULTICAST.search(block)
        ]
        assert len(mentions) >= 4, (
            "the rule passes vacuously if nothing mentions multicast; expected "
            f"the security rationale and the opt-in docs, got {mentions}"
        )

    def test_an_unmarked_claim_is_flagged(self) -> None:
        planted = '"""Peers discover each other via multicast scouting."""\n'
        blocks = _python_blocks(planted)
        assert [b for _l, b in blocks if _MULTICAST.search(b) and not _OPT_IN.search(b)], (
            "the rule must flag a block claiming multicast with no opt-in marker"
        )

    def test_a_marked_claim_is_not_flagged(self) -> None:
        planted = '"""Multicast is opt-in via ``STRANDS_MESH_MULTICAST=true``."""\n'
        blocks = _python_blocks(planted)
        assert not [b for _l, b in blocks if _MULTICAST.search(b) and not _OPT_IN.search(b)], (
            "a block that names the opt-in flag must pass"
        )


class TestMeshProseNamesTheRealDefault:
    """The corrected sites say what the default actually is."""

    def test_readme_mesh_paragraph_names_gossip_and_the_opt_in(self) -> None:
        paragraph = _paragraph_containing(_REPO_ROOT / "README.md", "is automatically a peer on a local Zenoh")
        assert "gossip scouting" in paragraph
        assert "ZENOH_CONNECT" in paragraph, "cross-host peers need explicit endpoints"
        assert "STRANDS_MESH_MULTICAST" in paragraph, "multicast must read as the opt-in it is"

    def test_session_module_docstring_names_the_opt_in(self) -> None:
        from strands_robots.mesh import session

        docstring = session.__doc__ or ""
        assert "gossip scouting" in docstring
        assert "STRANDS_MESH_MULTICAST" in docstring, (
            "the connection-strategy list must name the flag that turns "
            "multicast on, and the env-var list must document it"
        )
        assert "handles LAN discovery automatically" not in docstring

    def test_the_env_var_matrix_documents_the_flag_the_prose_names(self) -> None:
        """A flag the corrected prose names must be findable in the matrix.

        The README paragraph and :mod:`strands_robots.mesh.session`'s docstring
        both send a reader to ``STRANDS_MESH_MULTICAST``, and the mesh section
        points at the Configuration matrix for the ``STRANDS_MESH_*`` knobs.  A
        flag named in prose but absent from that table is a dead end.
        """
        readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
        rows = [line for line in readme.splitlines() if line.startswith("| `STRANDS_MESH_MULTICAST`")]
        assert len(rows) == 1, f"expected one env-var matrix row for the opt-in flag, got {rows}"
        assert _OPT_IN.search(rows[0]), "the matrix row must record that multicast is off by default"

    def test_the_architecture_diagram_labels_the_default_transport(self) -> None:
        svg = (_REPO_ROOT / "examples" / "lerobot" / "architecture.svg").read_text(encoding="utf-8")
        assert ">Zenoh gossip (default)</text>" in svg
        assert "multicast (default)" not in svg


class TestDeviceConnectIsADifferentTransport:
    """The scope boundary is measured, not assumed."""

    def test_device_connect_never_builds_the_mesh_scouting_config(self) -> None:
        device_connect = _REPO_ROOT / "strands_robots" / "device_connect"
        offenders = [
            path.relative_to(_REPO_ROOT)
            for path in sorted(device_connect.rglob("*.py"))
            if "scouting_block" in path.read_text(encoding="utf-8")
        ]
        assert not offenders, (
            "device_connect prose is excluded from the scan because it runs the "
            "external device_connect_edge runtime; if it starts building this "
            f"scouting config, bring it into scope: {offenders}"
        )
