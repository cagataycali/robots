# Artifact: Newton viewer window size vs frame size

`newton_viewer_dims.png` for the PR that puts `NewtonSimEngine.open_viewer`'s
`width` / `height` on the same pixel-count floor `add_camera` and the render
family already share.

- `capture.py` - run once per tree (`upstream/main` and the branch). Measures
  each candidate dimension through `_resolve_camera_view` (the funnel the render
  surfaces apply the domain in) and through `open_viewer` against a recording
  stand-in for `newton.viewer`, plus the single-viewer-slot recovery ledger, plus
  one real offscreen MuJoCo frame at the viewer's default 1280x720 as the
  unchanged-path grounding. Each run prints the tree it imported from.
- `compose.py` - builds the figure and asserts every rendered number against the
  two dumps (13 rows, 12 disagreements before / 0 after, the slot ledger, and
  `max|main - branch| <= 2` on the grounding frame), that the two dumps came from
  different trees, and that the figure's border is clean.
