### Fixed: `set_gripper("close")`, `move_to([x, y, z])` and `rotate_wrist(0.3)` say the value went to `robot_name`

The motion primitives take `robot_name` first, so a payload passed positionally
lands there and the payload argument stays unset. `set_gripper` then answered
`'state' must be "open" or "close", got None` for a call that passed `"close"`,
and `move_to` / `rotate_wrist` only said the argument was required. All three
refusals, on MuJoCo and Isaac, now name the slot that received the value and the
keyword to use: `The first positional argument is 'robot_name', and it received
'close'; name the argument instead: set_gripper(state=...).` Which calls succeed
is unchanged; the `GRASP PRESERVATION` docstrings now show the keyword form.
