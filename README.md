# render_width: what the planner is actually shown

`vera_render_width.png` is generated from two measured runs of `capture.py`, one in a
worktree at `upstream/main` and one on the PR branch. Each run renders a real MuJoCo
frame headless (`MUJOCO_GL=egl`), drives the production `VeraPolicy.get_actions` path,
and saves the view the provider put on the wire plus `view_widths`.

`compose.py` asserts every claim in the figure before saving it, including that the two
halves came from different trees and that the honored `render_width=128` view is
byte-identical across them.

- `facts_main.json` - measured on `upstream/main`
- `facts_branch.json` - measured on the PR branch
