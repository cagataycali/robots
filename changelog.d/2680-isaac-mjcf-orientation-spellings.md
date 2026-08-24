### Fixed: the Isaac scene loader reads every MJCF orientation spelling, not just `quat`

MJCF gives a body or a geom five mutually exclusive ways to state one rotation -
`quat`, `euler`, `axisangle`, `xyaxes` and `zaxis`. The scene loader read only
`quat`, so an object a model rotates with any of the other four was reported
unrotated and the load reported success. Identity is a valid orientation, so
nothing downstream could tell a body that was never rotated from one whose
rotation had been dropped.

The angle units and the Euler axis sequence are model-global, so they now come
from `<compiler angle>` and `<compiler eulerseq>` resolved across `<include>` the
way `meshdir` already is. A lower-case `eulerseq` rotates about the fixed axes
and an upper-case one about the moving axes, which is the same three rotations
composed in the opposite order; both readings and all six axis permutations are
graded against `mujoco.MjModel`. A body or geom declaring two spellings at once
is refused rather than resolved, because MuJoCo refuses that model outright and
any rotation the reader picked would be a guess.

Graded against MuJoCo's own compiler: 1115 cases across five compiler
configurations, 0 mismatches. Over 570 real MJCF assets three change, all mesh
rotations previously reported as identity, with no change of verdict.
