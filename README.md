# A refused weld removal must leave the constraint in place

Measurement backing the fix to `remove_equality_constraint` in
`strands_robots/simulation/mujoco/scene_ops.py`.

## Method

One script, run unchanged against `upstream/main` (in a detached worktree) and
against the branch. Each arm records the tree it imported so the two dumps
cannot be confused. Scene: a **static** carrier holding a **dynamic** cube by a
weld equality, so the cube hangs in mid-air only while that equality exists --
"is the weld still there" is directly visible.

Three states are captured per tree:

1. **reference** -- attach, settle 400 steps, render. Untouched by the change.
2. `_recompile_preserving_state` forced to fail **for the detach only**, then
   `detach_bodies`, then the identical retry.
3. the same refusal, then an unrelated `add_object`, then settle and render.

## Measured

| | `upstream/main` | this change |
|---|---|---|
| refused detach reports | `error` | `error` |
| weld on the live spec after | **gone** | intact |
| weld in the compiled model after | 1 | 1 |
| identical retry | **`error` ("not found")** | `success` |
| model `neq` after an unrelated `add_object` | **0** | 1 |
| cube settles at | **0.0849 m** (floor) | 0.4996 m (held) |

The reference render is identical across the two trees (`max|delta| = 1/255`);
panels 2 and 3 differ on **25.25%** of pixels.

`facts_main.json` / `facts_branch.json` are the raw dumps the figure is built
from. Every number rendered in the figure is asserted against them by the
compose step before it saves, along with the cross-tree reference identity and
the panel-difference floor.
