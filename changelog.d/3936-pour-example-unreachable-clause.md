### Fixed: the pour demo attributes its zero score to the scene, not the policy

`examples/17_pour_task.py` authors a benchmark whose success clause needs
`cap_slide` past 0.06 m, mounts the carton at x = 0.55, and scored 0 while
blaming the policy - "the `mock` policy, which does not act" - then closed by
inviting the reader to "point `--policy` at a trained provider".

No provider can win that clause on the robot the spec names. Measured on the
shipped `so100` and `sliding_carton`: the arm's tool point reaches x = 0.4405 m
and even the bounding sphere of its outermost geom stops at 0.4457, against the
cap plate's near face at 0.500 - 52.5 mm of daylight at the closest approach of
any arm geom over 20k sampled configurations. `mock` does drive the joints
(0.6-1.0 rad each, 3.29 m of tool path per episode); it simply never comes near.
Moving the station inside the reach envelope does not make the clause winnable
either: the cap's push face is a 4 mm strip flush under the carton walls, which
the gripper hits first, so a scripted IK push from every approach height creeps
the cap 8 mm of the 60 mm required.

The docstrings now say that, with the numbers they are graded against, and the
closing note points at the scripted route instead of at a policy.

`tests/test_examples_pour_task_cap_is_out_of_the_arms_reach.py` measures the
reach on the real assets, binds the quoted figures to the model, and pins the
scripted `set_joint_positions` route that does pour.
