### Fixed: `move_to` says when what the fingers held was left behind

On the SO-100 / SO-101 a friction pinch does not lift an object in MuJoCo, and
nothing said so: `set_gripper` reported "Closed on 'cube'" and the lift
`move_to` reported success while the cube stayed on the table. `move_to` now
reads which free bodies touch the fingers when it starts and, after moving at
least 2 cm, reports each one that moved less than half as far - in the text and
as `left_behind` in the JSON block - with the `attach_bodies(parent=...,
child=..., mode="weld")` call that carries it as a grasp assist.
