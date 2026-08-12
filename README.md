# teleop receive stream-replacement — measurement artifacts

Reproducibles for the PR "pin the receive side of the teleop stream-replacement contract".

* `teleop_receive_replacement.png` — the four-cell matrix (before/after), the
  mutation matrix, and the gate.
* `mutation_table.py` — introduces five plausible regressions into
  `Robot.start_teleop_receive`, one at a time, AST-scoped to that function
  (printing `in_fn` vs `in_file` per anchor), and runs each against two arms:
  the four new cases and the 175 pre-existing ones. Restores the source
  byte-identically in a `finally`.
* `mutation_results.json` — its measured output.
* `compose.py` — builds the figure; every rendered number is asserted against
  the measured JSON before the file is written.

M3 (tear down before validating) is invisible to **both** arms and cannot be
made observable on this surface: both refusable arguments are part of the
registry key, so a refused value can never name a registered entry. Ordering is
observable on the publish surface — `hz` is refused while `device_name` still
names the live stream — and is pinned there with the rate guard.
