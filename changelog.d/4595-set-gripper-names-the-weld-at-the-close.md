### Fixed: `set_gripper` says at the close when the squeeze cannot lift what it touches

On MuJoCo, `set_gripper(state="close")` now weighs the squeeze against the
object's weight: the weaker finger side's normal force times friction, against
`mass * g`. When the fingers cannot hold a free body (the SO-100 jaw pushes the
README cube into the floor while the fixed jaw never meets it), the reply names
`attach_bodies(parent=..., child=..., mode="weld")` before the lift, and the
JSON block carries `unpinched` with the per-finger forces. Contact counts now
include only contacts that push: "Closed on 'red_cube' (4 contacts)" was three
zero-force pairs and one jaw. The README says the hero pick uses the weld.
