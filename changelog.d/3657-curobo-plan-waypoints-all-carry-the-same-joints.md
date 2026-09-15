### Fixed: a cuRobo plan whose waypoints do not all describe the same joints is refused, not commanded partially

`CuroboPolicy._next_chunk` resolves the joint key names once per chunk, from the
width of that chunk's *first* waypoint, and then paired those keys with every
waypoint in the chunk under `zip(..., strict=False)`. Those keys are a claim about
one waypoint applied to all of them, and `strict=False` is what made a waypoint
the claim did not describe silent rather than reported. Measured through
`get_actions_sync` against the stub-planner seam `_extract_trajectory` documents,
on a 7-joint plan:

    waypoint narrower than the first  -> commanded on 4 of its 7 keys
    waypoint wider than the first     -> its trailing positions dropped
    waypoint carrying no position     -> `{}`, a command that moves no joint,
                                         inside a chunk every downstream
                                         `if not actions` guard therefore passes

Driven into MuJoCo physics, a plan whose second half reported only 4 of 7 joints
moved the arm 0.15 rad over a window it asked 1.17 rad of, and finished 1.17 rad
from its own endpoint on joint 1 where the same plan intact finished 0.15 rad
from it - under a successful plan, with no refusal and no warning.

Which silent wrong thing happened depended on `action_horizon`, a streaming
parameter that is no part of the plan. With the narrow waypoints at a chunk head,
`_resolve_joint_keys(4)` found no matching `set_robot_state_keys` entry and fell
back to positional `joint_0..joint_3` labels, which name no actuator at all, so
every value was dropped at the robot. Inside a chunk whose first waypoint was
full, the same waypoints were keyed correctly with only 4 of 7 filled, leaving
three joints holding mid-motion.

A planner's degree-of-freedom count does not change mid-plan, so a plan that is
not rectangular is a broken plan. It is graded when it is cached, beside the
`_MAX_TRAJECTORY_WAYPOINTS` guard, rather than when a chunk of it is served: the
offending waypoint can sit in the second or the tenth chunk, and refusing at
serve time would refuse after the arm had already run the first. The reason names
the waypoint and both widths, and nothing is cached, so the next call re-plans
instead of serving the rest. `MoveIt2Policy._unpack_trajectory` refuses a
positionless waypoint for the same reason, and `docs/policies/moveit2.md` already
stated the rule the two planner policies now share - a plan that commands nothing
is a planning failure, not a successful no-op plan. The pairing itself is now
strict, so a future path that caches an ungraded plan is refused rather than
truncated.
