# Artifact: IsaacSimulation.add_object rejects a keyword it cannot use

`measure_artifact.py` records what `add_object` compiles for the documented
`scale` alias and for a plausible-but-wrong key, run once per tree.
`render_artifact.py` replays each measured extent into a headless MuJoCo scene.
`compose_artifact.py` builds the figure and re-derives every number it prints.
`before.json` / `after.json` are the measurements (each records its own tree).
