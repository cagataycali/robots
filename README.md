# Artifact: the learning-rate domain

`lr_domain.png` was generated on an aarch64 Linux host, torch 2.11.0, from two
measured trees:

* `before-main.json` - measured in a `git worktree` at `strands-labs/robots@0692a439`
* `after-branch.json` - measured on the branch

The generator asserts every number in the figure against those two dumps before
saving (accepted-cell counts, the zero weight delta, the NaN divergence, and that
the two dumps came from different trees), plus a pure-white border check.
