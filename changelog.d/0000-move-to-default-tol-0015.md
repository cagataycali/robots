### Changed: `move_to` default `tol` is 0.015 m (was 0.01)

On the bundled so100, a reachable target (end-effector +0.10 m) returned
`did not reach ... within tol=0.01 m after max_steps=200 (residual 0.0105 m;
IK residual was 0.0085 m)`; retrying the same target with `tol=0.015` from that
pose "reached ... in 1 steps" - the arm was already there. One wasted caller
turn per motion for a 0.5 mm miss. 0.015 m is where those position servos
settle inside the step budget (from home the same move reaches in 31 steps at
0.0149 m); the Isaac primitive carries the same default so the shared
vocabulary agrees. Pass `tol=0.01` to keep the old bound.
