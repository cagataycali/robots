# Artifact: mesh discovery-posture prose (strands-labs/robots #2192)

`discovery_posture_prose.png` - the layer-5 diagram band redrawn from the SVG's own
`<rect>`/`<text>` geometry and CSS classes (Thor has no SVG rasterizer), plus the
measured per-block census of the shipped guard.

Reproduce, from a checkout of the PR branch:

    PYTHONPATH=. python3 capture.py    # writes facts.json (both trees, via `git show upstream/main:`)
    PYTHONPATH=. python3 compose.py    # asserts every drawn number, then writes the PNG

`capture.py` imports the shipped guard (`tests/mesh/test_discovery_posture_prose.py`)
and uses its own block readers, so the figure measures the rule that ships.

Supporting probes used to choose the scope with no exclusion list:

  * `measure_blocks.py`  - the same rule over four candidate scopes (A/B/C/D).
    Scope C (mesh package + README + repository SVGs) flags exactly the two stale
    sites; scope D (whole tree) picks up Device Connect's D2D pages and DDS/RTPS
    multicast, both a different transport's default.
  * `why_compliant.py`   - per block, which clause makes it compliant. 8 of the 10
    blocks that mention multicast are compliant on main: 6 name
    `STRANDS_MESH_MULTICAST`, and `session._build_config` says it is off without
    naming the flag - so the rule accepts either form.
