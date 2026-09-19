### Fixed: the planner providers plan from the robot's own observation

A robot publishes proprioception as per-joint scalars keyed by joint name and
writes no flat `observation.state` vector, and `CuroboPolicy` / `MoveIt2Policy`
read only that flat key -- so every simulated rollout planned from no start
state at all, and cuRobo `main` (which dereferences the start state) failed
inside the vendor library with `status=error` and 0 steps. Both providers now
read either shape through one helper in `strands_robots.policies._state_keys`,
the start state is projected onto the joints cuRobo plans over using cuRobo's
own two rosters (`franka.yml` locks the two Panda fingers: a 7-joint plan space
written into 9-joint waypoints), and a waypoint the declared action-key roster
cannot key is keyed by the joint names the robot published its state under
instead of fabricated `joint_<i>` labels that resolve to no actuator. An
observation carrying neither shape is refused naming both. The documented Panda
rollout now reaches 6.2 mm of the commanded `target_pose` where it previously
never moved; `docs/policies/curobo.md` also names the two dependencies cuRobo's
own install does not pull and the start pose the Panda model needs.
