### Added: `kimodo` hardware bridge for the Unitree G1

`strands_robots.policies.kimodo.hardware` provides a name-rename adapter
between Kimodo's URDF-named joint output (`left_hip_pitch_joint`, ...) and
lerobot's `unitree_g1` driver's enum-named action keys (`kLeftHipPitch.q`,
...). Both sides expose exactly the same 29 joints in the same canonical
order, so the bridge is a pure key rename — no reordering, no scaling.

* `get_joint_map()` — the lazily-built rename table, verified 29/29 at
  first-call time so a lerobot rename surfaces at import rather than silently
  mid rollout.
* `kimodo_action_to_lerobot_g1(action)` — rename a single per-tick dict.
* `build_lerobot_g1_action_dict(action, extra_action_keys=None)` — the
  hardware run loop's one-stop wrapper (renames + merges optional locomotion
  remote inputs).

Together with the sim path (`sim.run_policy(policy_provider="kimodo", ...)`,
verified end-to-end on `Robot("g1", mesh=False)`), this closes the gap
between Kimodo's kinematic output and the two runtime targets — the same
`KimodoPolicy` object drives sim actuators and the real robot's DDS lowcmd
path through lerobot's `UnitreeG1.send_action`.

11 unit tests pin the rename table + failure surfaces (missing joint =
`KeyError`, unknown input keys dropped, extras override the rename) without
requiring a real robot or the `unitree_sdk2py` runtime.
