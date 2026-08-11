# vera-ik-bridge-lazy-build

Measurement artifacts for the VERA lazy `MinkIKBridge` build coverage PR.

* `capture.py` - drives a real eef-delta rollout through `VeraPolicy.get_actions`
  against a Panda at its `home` keyframe (headless MuJoCo, `MUJOCO_GL=egl`),
  records the bridge the first inference built and renders before/after.
  Self-audits: the bridge was `None` before and built after, the descent exceeds
  2 cm, the panels differ on >10% of pixels, and the arm fills >50% of the frame.
* `compose.py` - builds the figure. Asserts every rendered number against
  `facts.json`, the derived row pitches, that all text lands inside its panel,
  and that the 8 px border is pure white.
* `mutate.py` - the two-arm mutation table: each plausible regression in
  `_ensure_ik_bridge` is applied with an AST-scoped anchor (printing `in_fn` vs
  `in_file`), run against these tests and against the pre-existing vera suite,
  then restored byte-identically.
* `facts.json` - every measured value the figure and the PR body quote.
