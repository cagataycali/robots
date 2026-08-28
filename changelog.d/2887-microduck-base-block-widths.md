### Fixed: `build_observation` refuses a floating-base block it cannot read

`MicroduckPolicy`'s observation vector is assembled from four inputs. Three of them are the policy's own
state and are held to a width at the policy seam: `default_pose` against `len(joint_names)` in
`_ensure_config`, a `command` override against the width `command_names` declares in
`_apply_command_kwargs`, and the graph's returned action - which becomes the next tick's `last_action` -
in `get_actions`. The fourth arrives from the caller's observation dict: the two floating-base blocks
`base_ang_vel` (3) and `base_quat` (4, wxyz). Those are read only inside the builder, so no seam ever
sees them, and a truncating `[:3]` / `[:4]` slice took whatever width arrived.

That is silent in both directions. One component short, `q[1:4]` is a 2-vector `np.cross` reads as
planar, so the quaternion silently loses its `z`: the gravity block keeps the documented width and stays
finite while the direction the robot is told is "down" moves 7.5 degrees for a small-yaw pose and 20.7
for a roll-then-yaw one. Its norm drifts with it (0.991 and 0.935), but this module normalises nothing
and reads no norm, so at a percent off unity that is not a signal anything acts on. Over-long, a
7-element `[base_pos, base_quat]` slice - a caller handing over a floating-base `qpos` slice - is read as
a quaternion made of positions, 70.9 degrees off. A short `base_ang_vel` instead falsified the builder's
own documented `48 + len(command)` return width, handing the graph fewer values than its
`observation_names` metadata declares.

Both blocks now route through one reader that refuses any width other than the one the layout defines,
naming the block, both widths and the contract it protects. The sibling locomotion observation builders
in this package already hold every sub-vector this way via `wbc.observation._require_len`; this builder
was the one that did not. NumPy 2.0 deprecated the 2-vector cross the short read relied on, so that path
was already heading for a raise from inside numpy naming neither block.

The shipped simulation backends slice these blocks at fixed widths, so the policy path is exactly right
today and this is latent there. `build_observation` is a public export, and a caller assembling the dict
from an IMU or teleop bridge has no seam at all. Finiteness of the base blocks is a separate axis and is
pinned as deliberately unchanged.
