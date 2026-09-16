### Fixed: a joint write says what unit its numbers are in

`set_joint_positions` takes a MuJoCo joint coordinate - radians on a hinge,
metres on a slide - and nothing said so. The published `positions` description
read "Joint name -> position mapping for set_joint_positions", and a value the
joint's range does not contain was refused as
`shoulder_pan=-96.2 outside [-1.92, 1.92]`: a pair of bounds with no unit, whose
cheapest reading is "clamp to the bound". A caller mirroring a real arm onto its
sim twin arrives at exactly that message, because the driver on the other side
reports the other unit (`drivers/feetech` reads an SO-arm in degrees), and
clamping lands a pose 110 degrees from the one intended.

The unit is now derived from the joint's type
(`simulation.mujoco.scene_ops.joint_position_unit`) rather than asserted, so it
is right for a slide joint too, and it is stated where the caller reads: the
published `positions` / `velocities` descriptions, the docstrings, and the
range refusal - which additionally names the conversion when converting the
value really would land it inside the range, e.g.
`shoulder_pan=-96.2 outside [-1.92, 1.92] rad (radians, not degrees: -96.2 deg
= -1.679 rad)`. A value no conversion rescues is not told it is degrees, so the
sentence states a fact about the call rather than guessing at intent.
