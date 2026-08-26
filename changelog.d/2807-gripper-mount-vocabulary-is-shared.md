### Fixed: `gripper_body` is answered from one gripper-hint vocabulary

`list_bodies(robot_name=...)` advertises a best-guess gripper/end-effector mount in its
`gripper_body` field -- the surface a caller reads to resolve `add_camera(parent_body=...)` for a
wrist view. Both backends matched that field through one shared rule, `hint_matches_name`, and each
kept a hint vocabulary of its own. The vocabularies had drifted: only one of them carried `jaw`.

So the SO-100 reported its mount on one backend and reported `gripper_body: None` on the other, in
the same payload that listed both of its jaw bodies. It is a shipped registry robot whose gripper
bodies are named `Fixed_Jaw` and `Moving_Jaw`, `docs/simulation/newton.md` already documented its
`gripper_body` as a jaw, and `docs/simulation/world-building.md` documented the narrower vocabulary
as the whole rule and told the reader that `None` means the robot has no gripper-like body. On a
loaded SO-100 the field read `None` while the bodies list beside it read `arm/Fixed_Jaw`,
`arm/Moving_Jaw`.

Sharing the matcher is not on its own enough for two surfaces to agree; they agree only where they
also read the same vocabulary. `gripper_body` is one question about one robot, so the vocabulary
behind it now has one owner, `simulation.ik.GRIPPER_BODY_HINTS`, which both backends read.
`discover_ee_frame` keeps its own words on purpose: it answers a different question -- which frame to
solve IK *to* -- and still resolves the SO-100's wrist rather than a jaw. The two are pinned as a
difference so neither can be quietly folded into the other.

Measured over the 59 loadable registry assets, 58 report the mount they always reported and one
changes: `so100`, from `None` to `Fixed_Jaw`. The added word matches three bodies in the whole
corpus and every one of them is a real gripper jaw, so no asset gains a mount it should not have.
