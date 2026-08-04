### Fixed: a pose_tool joint target the arm cannot honor is refused instead of clamped to the end stop

`MotorController.degrees_to_position` clamps its argument into the motor's
configured `range` before scaling it onto the 12-bit `Goal_Position` register, so
every target outside that range shared one encoding -- the mechanical limit --
while the success text echoed the value the caller asked for. `nan` reached the
limit through the clamp itself, because `min(max_deg, nan)` returns `max_deg`.
Measured on `shoulder_pan` (configured `(-180, 180)`), `position=nan`,
`position=inf` and `position=5000` each returned `status="success"` having
written `Goal_Position` 4095 -- a full-travel slam to the end stop at servo
speed, indistinguishable from a deliberate `position=180` -- and reported
`Moved shoulder_pan to nan deg`. `position=True` was read as a real 1-degree
move, `bool` being an `int` subclass.

`move_motor`, `move_multiple` and `incremental_move` now refuse a target they
cannot honor before the port is opened, so the arm never travels on such a
request. Finiteness, numeric-ness and `bool` are delegated to the shared
`finite_number_error`, so an off-domain target is reported in the words every
other surface uses; only the per-joint bounds are decided in `pose_tool`,
because they are a property of the arm it drives. A `delta` is bounded by the
joint's *full travel* rather than its endpoints, since a displacement is not
absolute and only a magnitude exceeding the whole range is unhonorable from
every starting position.

The bounds now have one authority: the per-joint table is module-level and
`MotorController` is configured from a copy of it, so the table the guard checks
cannot disagree with the table the servo is driven from.

The clamp itself stays, and is now unreachable from those three actions. Its
remaining callers supply targets from somewhere other than the caller's
arguments -- a stored pose, which `PoseManager.validate_pose` already refuses out
of bounds, and `reset_to_home`'s own in-range literals -- so removing it would
turn those paths into raises, which is a separate change.
