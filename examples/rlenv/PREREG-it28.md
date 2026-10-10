# it28 PRE-REGISTRATION (written before any grid episode is run)

it27 falsified it26's "decoupling is bought with competence" on a third arm. What survived was an
observation, not a rule: the residual shortcut lands on a different FEATURE per arm --
so100 lateral (cube_y 0.69), koch radial (radius 0.7194, |cube_y| 0.6942), so101 none (all at chance).

An observation on three arms is worth as much as it26's rule was on two, so it is tested here as a
prediction from an INDEPENDENT measurement: `shortcut_axis.py` drives the goal-aware expert at the same
sigma=0.05 over a GRID of cube positions spanning each arm's harvest band (band_scale 1.0), equal
episodes per cell, and reports the marginal spread of success rate along x (forward/radial) and along y
(lateral). That landscape is a property of the arm and its band; it never touches the recorded pairs,
so the test is not circular.

PREDICTIONS (primary, all three must hold):
  so100 push: spread_y_lateral > spread_x_forward          (its pair's shortcut is cube_y 0.69)
  koch  push: spread_x_forward > spread_y_lateral          (its pair's shortcut is radius 0.7194)
  so101 push: max(spread) < 0.25, i.e. predicted_flat true (its pair has no shortcut at all)

FALSIFIED IF any arm's prediction fails. Then the lane reports that the shortcut axis is NOT predictable
from the expert's success landscape either, and stops offering a geometric explanation at all -- the
honest fallback being that each band must be screened by measuring its pair, which is what the lane
already does and the only claim that has survived every test so far.

Grid: 4 bins per axis, 8 episodes per cell = 128 episodes per arm, sigma=0.05, seeds disjoint from every
published shard's stream (base 4242 + 1000*ix + 100*iy + k vs the shards' harvest seeds).
