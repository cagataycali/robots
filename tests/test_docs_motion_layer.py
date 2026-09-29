"""The docs site's motion is a preference, and it survives instant navigation.

``docs/stylesheets/extra.css`` and ``docs/assets/motion.js`` add the site's small
motions: the content column fades in after an instant-navigation swap, the
landing's proof numbers count up, a copy pill says "copied", the "On this
page" bar travels, the theme toggle cross-fades. Two things about that layer
are invisible to MkDocs and to the browser until a reader hits them:

* a reader who asked their OS for less motion must get the end state at once,
  so the stylesheet needs the universal ``prefers-reduced-motion`` override and
  every script that animates has to read the same media query;
* ``navigation.instant`` swaps ``<main>`` without reloading scripts, so a script
  that wires the page once at load leaves the next page dead (the robot picker
  on the landing was exactly that); every local script therefore re-runs on
  Material's ``document$``.

This guard pins both, plus the one CSS join a rename would silently break: the
JavaScript sets ``--sr-toc-y`` / ``--sr-toc-h`` that the stylesheet reads.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"
DOCS_DIR = REPO_ROOT / "docs"
STYLESHEET = DOCS_DIR / "stylesheets" / "extra.css"

_REDUCED_MOTION_BLOCK = re.compile(r"@media \(prefers-reduced-motion: reduce\)\s*\{(.*?)\n\}", re.S)
_ANIMATES = re.compile(r"requestAnimationFrame|startViewTransition|\.animate\(|classList\.add\(\"sr-enter\"")


def _declared_scripts() -> list[Path]:
    text = MKDOCS_YML.read_text(encoding="utf-8")
    block = re.search(r"^extra_javascript:\s*$(.*?)^\S", text, re.M | re.S)
    assert block, "mkdocs.yml declares no extra_javascript block"
    paths = re.findall(r"^\s+-\s+(?:path:\s+)?(\S+\.js)\s*$", block.group(1), re.M)
    return [DOCS_DIR / p for p in paths]


def _features() -> list[str]:
    text = MKDOCS_YML.read_text(encoding="utf-8")
    block = re.search(r"^  features:\s*$(.*?)^  \S", text, re.M | re.S)
    assert block, "mkdocs.yml theme has no features block"
    return re.findall(r"^\s+-\s+(\S+)", block.group(1), re.M)


def test_reduced_motion_ends_every_animation_and_transition() -> None:
    """One universal override, so a rule added later cannot forget to opt out."""
    css = STYLESHEET.read_text(encoding="utf-8")
    blocks = _REDUCED_MOTION_BLOCK.findall(css)
    assert blocks, f"{STYLESHEET.name} has no prefers-reduced-motion block"
    universal = [b for b in blocks if re.search(r"^\s*\*,\s*\*::before,\s*\*::after\s*\{", b, re.M)]
    assert universal, "the reduced-motion block does not carry the universal `*, *::before, *::after` override"
    body = universal[0]
    for prop in ("animation-duration", "transition-duration", "animation-iteration-count"):
        assert f"{prop}: 0.01ms !important" in body or f"{prop}: 1 !important" in body, (
            f"the universal reduced-motion override does not pin {prop}; a keyframe or transition would still run"
        )


def test_every_script_that_animates_reads_the_motion_preference() -> None:
    """A script may animate only after asking prefers-reduced-motion."""
    offenders = []
    for script in _declared_scripts():
        text = script.read_text(encoding="utf-8")
        if _ANIMATES.search(text) and "prefers-reduced-motion" not in text:
            offenders.append(script.name)
    assert not offenders, f"scripts that animate without reading prefers-reduced-motion: {offenders}"


def test_every_local_script_survives_instant_navigation() -> None:
    """With navigation.instant on, a script that wires the page once leaves the next page dead."""
    if "navigation.instant" not in _features():
        return
    stale = [s.name for s in _declared_scripts() if "document$" not in s.read_text(encoding="utf-8")]
    assert not stale, (
        f"navigation.instant is on and these scripts never re-run on Material's document$: {stale}. "
        "Subscribe the page wiring to window.document$ (falling back to a direct call)."
    )


def test_the_toc_bar_variables_join_script_and_stylesheet() -> None:
    """The stylesheet reads the two variables the script writes; a rename on one side hides the bar."""
    css = STYLESHEET.read_text(encoding="utf-8")
    motion = (DOCS_DIR / "assets" / "motion.js").read_text(encoding="utf-8")
    for var in ("--sr-toc-y", "--sr-toc-h"):
        assert f"var({var}" in css, f"{STYLESHEET.name} does not read {var}"
        assert f'"{var}"' in motion, f"motion.js does not set {var}"
