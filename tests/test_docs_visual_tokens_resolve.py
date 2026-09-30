"""Every picture token on a page has its files, and every committed SVG is what its scene renders.

``docs/hooks/visuals.py`` expands ``{{drawing:<id>}}`` into the paper and dark
SVGs that ``docs/drawings/_tools/scene.py`` renders from the scene module
``docs/drawings/scenes/<id>.py``, and ``{{sim:<id>}}`` into the frame under
``docs/assets/sim``. A token with no files would ship as literal text (the hook
warns, and ``--strict`` fails the build, but only when the build runs); an SVG that
no longer matches its scene is a drawing whose source lies. Both are graded here,
from the sources, without a build or a browser. The pipeline lands before any page
places a token, so a picture with no page is not graded yet; the pages that place
them bring that rule. The hook and the renderer
are loaded by path: the docs venv is not the test venv.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from tests._docs_hooks import docs_hook

_REPO = Path(__file__).resolve().parents[1]
_DOCS = _REPO / "docs"
_SCENE_TOOL = _DOCS / "drawings" / "_tools" / "scene.py"
_SCENES = _DOCS / "drawings" / "scenes"
_SVGS = _DOCS / "assets" / "drawings"


def _pages() -> list[Path]:
    return sorted(p for p in _DOCS.rglob("*.md") if "hooks" not in p.parts)


def _references() -> set[tuple[str, str]]:
    hook = docs_hook("visuals")
    refs: set[tuple[str, str]] = set()
    for page in _pages():
        refs |= hook.referenced_ids(page.read_text(encoding="utf-8"))
    return refs


def _renderer():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("scene", _SCENE_TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("scene", module)
    spec.loader.exec_module(module)
    return module


def test_every_visual_token_resolves_to_committed_files() -> None:
    hook = docs_hook("visuals")
    missing = []
    for kind, ident in sorted(_references()):
        out = hook.drawing_html(ident, "") if kind == "drawing" else hook.sim_html(ident, None, "")
        if out is None:
            missing.append(f"{{{{{kind}:{ident}}}}}")
    assert not missing, (
        f"visual tokens with no files: {missing}. A drawing needs docs/drawings/scenes/<id>.py and both "
        "docs/assets/drawings/<id>.{paper,dark}.svg (run docs/drawings/_tools/scene.py --all); a sim frame needs "
        "docs/assets/sim/<id>.png."
    )


def test_every_scene_has_its_two_exports() -> None:
    scenes = {p.stem for p in _SCENES.glob("*.py") if not p.name.startswith("_")}
    exports = {p.name.split(".")[0] for p in _SVGS.glob("*.svg")}
    unexported = sorted(scenes - exports)
    assert not unexported, f"scenes with no SVG: {unexported}. Run docs/drawings/_tools/scene.py --all."
    for name in sorted(scenes):
        for scheme in ("paper", "dark"):
            assert (_SVGS / f"{name}.{scheme}.svg").is_file(), f"{name}: no {scheme} export"


def test_every_committed_svg_is_what_its_scene_renders() -> None:
    """The scene module is the source of truth; a hand-edited or stale SVG is refused."""
    renderer = _renderer()
    scenes = renderer.load_scenes()
    assert scenes, f"no scenes under {_SCENES}"
    drift = renderer.verify_svgs(scenes, _SVGS)
    assert drift == [], (
        f"committed SVGs that differ from their scene: {drift}. Edit docs/drawings/scenes/<id>.py, then run "
        "docs/drawings/_tools/scene.py --all and commit both."
    )
    stray = sorted(p.name for p in _SVGS.iterdir() if p.suffix != ".svg")
    assert stray == [], f"only SVGs are committed under docs/assets/drawings; PNGs go to a shots directory: {stray}"


def test_every_scene_keeps_the_brand_rules() -> None:
    """One accent element, a title, a lead, a footnote, no em or en dash, every identifier in the docs."""
    renderer = _renderer()
    problems = renderer.check_labels(renderer.load_scenes(), _DOCS)
    # The pages that name every identifier the drawings draw land with the content; until
    # then a name the package itself defines (EmbodimentMap) is not an invention.
    package_text = "\n".join(f.read_text(encoding="utf-8") for f in (_REPO / "strands_robots").rglob("*.py"))
    problems = [
        line for line in problems if not (": name not in docs: " in line and line.rsplit(": ", 1)[1] in package_text)
    ]
    assert problems == [], problems
    for name, scene in renderer.load_scenes().items():
        svg = scene.svg("paper")
        accents = (
            svg.count('class="card accent-card"')
            + svg.count('class="chip accent-chip"')
            + svg.count('class="wire accent-wire')
        )
        assert accents >= 1, f"{name}: no accent element; every drawing has exactly one green thing it is about"
        assert accents == 1, f"{name}: {accents} accent elements; one green element per drawing"
        assert scene.title and scene.lead, f"{name}: a drawing has a mono title and a Grotesk lead"
        assert scene.empty_fraction() <= 0.2, (
            f"{name}: {scene.empty_fraction():.0%} of the canvas is empty (limit one fifth)"
        )
        assert 'class="grot muted" x="600.0"' in svg, f"{name}: no centred footnote"


def test_the_hook_emits_lazy_images_for_both_schemes() -> None:
    hook = docs_hook("visuals")
    ident = sorted(p.stem for p in _SCENES.glob("*.py") if not p.name.startswith("_"))[0]
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
