### Fixed: `get_robot_state` answers with a configuration the robot was actually in

The MuJoCo readback assembled one answer out of several mjData arrays - each
joint's `qpos`/`qvel`, a floating base's pose and twist, and the world position
of the frame `move_to` drives - without holding the simulation lock. A
concurrent `mj_step` from a `PolicyRunner` worker, the `step()` loop or the
camera recorder lands between two of those reads, so the returned state was a
splice of two physics steps: joint angles that never coexisted, or joints from
one step beside an `end_effector` position from another. That second shape is
the damaging one, because the documented use of that field is to offset a
`move_to` target from it, so the target was computed in a configuration the arm
was not in. With a writer alternating an SO-101 between two coherent poses,
4,323 of 45,414 samples returned a mixed joint vector and 2,958 crossed
configurations; serialised, 0 of 40,791 do.

Every read now happens in one critical section - splitting it per section would
leave the same gap - matching how `render`, `render_depth`, `get_frame`,
`get_observation`, `get_body_state` and the joint writers already serialise
theirs. `apply_force` was the other public method touching mjData off-lock: its
default application point (the body's centre of mass) is now read inside the
critical section that latches the wrench, so the point it reports and the wrench
it applies come from the same configuration.
