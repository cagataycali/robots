# Isaac articulation read/write surfaces

Scripts behind the artifact on strands-labs/robots PR #2212 (tests only, 0 production lines).

- `limits_probe.py` - the empirical probe: every documented behaviour of
  `_articulation_dof_limits`, `_read_joint_positions` and
  `_apply_position_targets`, driven directly. Every one is already correct;
  this is a coverage slice, not a fix.
- `cam_sweep.py` - the measured camera sweep behind the render framing
  (24 candidates; the chosen one gives 47.4% arm pixels / 18.2% differing).
- `mutate.py` - the mutation table: 8 regressions x 2 arms, AST-scoped to the
  enclosing function, restored byte-identically in a `finally`.
- `capture.py` / `compose.py` / `facts.json` - the artifact. `capture.py` drives
  the real Isaac primitives against a faked articulation (Isaac Sim is not
  required to reach either limit-source decision) and replays the joint targets
  they command onto a MuJoCo arm declaring the same joint vocabulary and limits.

Run `capture.py` with `PYTHONPATH=<repo> MUJOCO_GL=egl`; both scripts print the
tree they resolved so a measurement is always attributable.
