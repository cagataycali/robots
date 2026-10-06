### Fixed: `move_to` says when the arm pushed an object, and a missed close says where it went

`move_to` is not collision-aware, and on the README pick (open, `move_to` the
cube's centre, close) the SO-100 descent shoved the cube 2.6 cm while the reply
said only "reached". The close then answered "Closed on nothing ... move_to the
object first", which sent the caller back to the same target and the same push.
`move_to` now names every loose body the arm touched and moved at least 5 mm,
with where it is now (`pushed` in the JSON block), and a close that touches
nothing names the nearest object and its current position (`nearest_object`)
instead of the advice that looped.
