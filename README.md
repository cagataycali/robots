# pose_tool incremental_move: the delta domain's two deferrals

Measurement scripts for the PR that pins both deferrals on
`_joint_delta_error` and corrects a reachability claim about
`degrees_to_position`'s clamp.

* `capture.py` -- sweeps the delta from -400 to +400 deg against a joint parked
  at +169.89 deg and records, per delta, the domain's verdict and the
  `Goal_Position` actually written to the bus. Also drives the three ledger rows
  through the public tool. Run with `PYTHONPATH=<repo>`.
* `mutate.py` -- applies five plausible regressions, each AST-scoped to its
  enclosing function (printing `in_fn` / `in_file` as the justification), and
  runs both arms: the two new test classes and the 249 pre-existing pose_tool
  cases. Restores the source byte-identically in a `finally`.
* `compose.py` -- builds the figure. Every drawn number is read from the two
  JSON dumps and asserted before the image is saved.
* `art-mutations.json`, `art-facts-*.json` -- the raw measurements.

Measured on an NVIDIA Jetson AGX Thor with the arms unplugged: the servo bus is
a recording stand-in, so the `Goal_Position` values are what the tool would have
written to hardware.
