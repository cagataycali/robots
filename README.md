# RLTrainSpec.gamma discount-factor domain

`gamma_domain.png` is generated from two runs of one capture script, executed in
two separate trees (a `git worktree` at `upstream/main`, and the branch); each
dump records its own tree and the compose step asserts the two differ.

Every number rendered in the figure is re-derived from `measured_main.json` /
`measured_branch.json` and asserted before the figure is saved:

* the divergence row comes from the real `strands_robots.training.rl.ppo.compute_gae`
  with unit rewards, at horizons 12/24/48/96;
* `gamma=1.5` at T=96 must exceed 1e14 and `gamma=5.0` must be `inf`;
* `gamma=0.99` and `gamma=1.0` must stay below 20 at every horizon;
* `gamma=-0.5` must be exactly 1.0;
* every accepted probe must be accepted on both trees, and every refused probe
  must be accepted on main and refused on the branch, for both backends;
* the honored real PPO run's trained `|w|max` must be **equal** across the two
  trees, and the two renders must agree to `max|delta| <= 2` of 255;
* the render must be a real scene (saturated-pixel fraction > 0.05);
* every text label must land inside its axes, and the 8px figure border must be
  pure white.
