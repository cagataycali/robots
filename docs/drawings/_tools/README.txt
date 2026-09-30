# Drawings

Every diagram on the site is a scene module under docs/drawings/scenes/<id>.py that returns a Scene
built from box / chip / chips / arrow / down / section / para / footnote calls. scene.py renders each
one to docs/assets/drawings/<id>.paper.svg and <id>.dark.svg with the site's tokens (Space Grotesk
and JetBrains Mono embedded from docs/assets/fonts, one green accent, 1px borders, 8px radius, no
shadows). The scene module is the source of truth; the SVGs are committed so the build needs no
browser, and a grader re-renders every scene and refuses a drift between the two.

    python3 docs/drawings/_tools/scene.py --all                  # rebuild every SVG
    python3 docs/drawings/_tools/scene.py d01_what_is            # one scene
    python3 docs/drawings/_tools/scene.py --all --verify         # exit 1 when a committed SVG is stale
    python3 docs/drawings/_tools/scene.py --all --check docs     # exit 1 on a name absent from docs/**/*.md
    python3 docs/drawings/_tools/scene.py --all --png --shots ~/shots   # 2x PNGs to LOOK at (playwright chromium)

Every drawing has: a mono title centred at the top and a Grotesk lead under it; uppercase tracked
section captions above each group; boxes with a LEFT-aligned mono title and a Grotesk sub-line;
identifiers as chips; ONE green element (soft fill, accent border, accent text) on the thing the
drawing is about; dashed boxes for a layer; 1.5px orthogonal wires with filled heads and the label
beside the wire, dashed for a return or refused path; a muted footnote; the canvas filled (no more
than a fifth empty). No em or en dashes. A page places a drawing with {{drawing:<id>}}.
