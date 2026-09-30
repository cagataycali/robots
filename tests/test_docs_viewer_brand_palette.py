"""The viewer's brand palette: which model colours the docs repaint, and with what.

``docs/hooks/manifest.py`` carries ``_BRAND_PALETTES``: for a robot, a list of
rules mapping a colour the MJCF spells (``from``, rgb in 0..1) to a token the
viewer resolves against the page theme (``to``). The SO-101's printed parts are
yellow in the menagerie model and Strands green on the site; its servos stay
black. Two things would fail silently without this guard: a token the viewer's
``_theme()`` does not define paints the part black, and a rule for a robot the
viewer cannot render (or that is not in the registry) is dead configuration.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests._docs_hooks import docs_hook

REPO_ROOT = Path(__file__).resolve().parent.parent
VIEWER = REPO_ROOT / "docs" / "assets" / "viewer" / "robot-viewer.js"


def _hook():
    return docs_hook("manifest")


def _theme_tokens() -> set[str]:
    """The keys ``_theme()`` returns, read from the viewer source."""
    src = VIEWER.read_text(encoding="utf-8")
    body = re.search(r"_theme\(\) \{.*?return \{(.*?)\n    \};", src, re.S)
    assert body, "robot-viewer.js has no _theme() returning an object literal"
    return set(re.findall(r"^\s*(\w+):", body.group(1), re.M))


def test_every_palette_rule_is_well_formed_and_resolvable() -> None:
    hook = _hook()
    tokens = _theme_tokens()
    for robot, rules in hook._BRAND_PALETTES.items():
        assert rules, f"{robot}: empty palette"
        for rule in rules:
            rgb = rule["from"]
            assert len(rgb) == 3 and all(0.0 <= float(c) <= 1.0 for c in rgb), (
                f"{robot}: from={rgb} is not an rgb triple in 0..1"
            )
            assert rule["to"] in tokens, (
                f"{robot}: token {rule['to']!r} is not a key of the viewer's _theme(); it would paint black"
            )


def test_every_palette_names_a_robot_the_viewer_renders() -> None:
    manifest = _hook().build_manifest()["robots"]
    for robot, rules in _hook()._BRAND_PALETTES.items():
        assert robot in manifest, f"{robot} is not in the registry"
        assert manifest[robot]["viewer"], f"{robot} never reaches the viewer; its palette is dead configuration"
        assert manifest[robot]["palette"] == rules, f"{robot}: the manifest does not carry its palette"


def test_robots_without_a_palette_keep_their_own_colours() -> None:
    manifest = _hook().build_manifest()["robots"]
    painted = {name for name, entry in manifest.items() if entry["palette"]}
    assert painted == set(_hook()._BRAND_PALETTES), (
        f"manifest paints {sorted(painted)}; the hook lists {sorted(_hook()._BRAND_PALETTES)}"
    )


def test_the_so101_prints_in_strands_green() -> None:
    """The one rule this landed with: the menagerie's yellow (1, 0.82, 0.12) becomes the accent."""
    rules = _hook()._BRAND_PALETTES["so101"]
    assert {"from": [1.0, 0.82, 0.12], "to": "accent"} in rules
