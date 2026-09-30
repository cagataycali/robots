"""Every picture token on a page has its files, and every committed picture has a page.

``docs/hooks/visuals.py`` expands ``{{drawing:<id>}}`` into the paper and dark
SVG exports of ``docs/drawings/<id>.excalidraw`` and ``{{sim:<id>}}`` into the
frame under ``docs/assets/sim``. A token with no files would ship as literal
text (the hook warns, and ``--strict`` fails the build, but only when the build
runs); an export with no page is a picture nobody sees and a scene nobody
rebuilds. Both are graded here, from the sources, without a build. The hook
itself is loaded by path: the docs venv is not the test venv.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DOCS = _REPO / "docs"
_HOOK = _DOCS / "hooks" / "visuals.py"


def _hook():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("docs_visuals_hook", _HOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pages() -> list[Path]:
    return sorted(p for p in _DOCS.rglob("*.md") if "hooks" not in p.parts)


def _references() -> set[tuple[str, str]]:
    hook = _hook()
    refs: set[tuple[str, str]] = set()
    for page in _pages():
        refs |= hook.referenced_ids(page.read_text(encoding="utf-8"))
    return refs


def test_every_visual_token_resolves_to_committed_files() -> None:
    hook = _hook()
    missing = []
    for kind, ident in sorted(_references()):
        out = hook.drawing_html(ident, "") if kind == "drawing" else hook.sim_html(ident, None, "")
        if out is None:
            missing.append(f"{{{{{kind}:{ident}}}}}")
    assert not missing, (
        f"visual tokens with no files: {missing}. A drawing needs docs/drawings/<id>.excalidraw and both "
        "docs/assets/drawings/<id>.{paper,dark}.svg (run docs/drawings/_tools/render.py); a sim frame needs "
        "docs/assets/sim/<id>.png."
    )


def test_every_committed_drawing_is_placed_on_a_page() -> None:
    placed = {ident for kind, ident in _references() if kind == "drawing"}
    scenes = {p.stem for p in (_DOCS / "drawings").glob("*.excalidraw")}
    exports = {p.name.split(".")[0] for p in (_DOCS / "assets" / "drawings").glob("*.svg")}
    orphans = sorted((scenes | exports) - placed)
    assert not orphans, (
        f"drawings no page references: {orphans}. Place {{{{drawing:<id>}}}} on the page that explains it, or delete the scene and its exports."
    )
    unexported = sorted(scenes - exports)
    assert not unexported, f"scenes with no SVG export: {unexported}. Run docs/drawings/_tools/render.py."


def test_every_committed_sim_frame_is_placed_on_a_page() -> None:
    sim_dir = _DOCS / "assets" / "sim"
    if not sim_dir.is_dir():
        return
    placed = {ident for kind, ident in _references() if kind == "sim"}
    frames = {p.stem for p in sim_dir.glob("*.png")}
    orphans = sorted(frames - placed)
    assert not orphans, (
        f"sim frames no page references: {orphans}. Place {{{{sim:<id>}}}} under the fence that built it."
    )


def test_the_hook_emits_lazy_images_for_both_schemes() -> None:
    hook = _hook()
    ident = next((i for k, i in _references() if k == "drawing"), None)
    if ident is None:
        return
    out = hook.drawing_html(ident, "../")
    assert out is not None
    assert out.count('loading="lazy"') == 2, out
    assert "#only-light" in out and "#only-dark" in out, out


def test_every_sim_frame_has_a_manifest_entry_and_every_entry_a_frame() -> None:
    """``docs/hooks/sim_frames.py`` renders frames from ``docs/hooks/data/sim_frames.json``;
    a frame nobody can regenerate, or an entry nobody rendered, is graded here."""
    manifest_path = _DOCS / "hooks" / "data" / "sim_frames.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = {p.stem for p in (_DOCS / "assets" / "sim").glob("*.png")}
    promised: dict[str, str] = {}
    for ident, entry in manifest.items():
        for frame in entry["frames"] if "script" in entry else [ident]:
            promised[frame] = entry["page"]
    assert sorted(set(promised) - frames) == [], "manifest entries with no frame; run docs/hooks/sim_frames.py"
    assert sorted(frames - set(promised)) == [], (
        "frames with no manifest entry; add them to docs/hooks/data/sim_frames.json"
    )
    placed = {ident for kind, ident in _references() if kind == "sim"}
    for frame, page_path in promised.items():
        page = _DOCS / page_path
        assert page.is_file(), f"{frame}: page {page_path} does not exist"
        assert frame in placed, f"{frame}: no page places {{{{sim:{frame}}}}} (the manifest names {page_path})"
