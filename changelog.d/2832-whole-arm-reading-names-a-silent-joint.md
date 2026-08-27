### Fixed: a whole-arm reading answers for every joint it was asked for

`MotorController.read_all_positions` skips a motor whose reply did not verify, so
it returns a subset of `motor_configs` carrying no record of what fell out. Both
tool actions that consume it guarded only the all-empty case, so a truthy partial
reading passed straight through. On a six-joint SO-101 with one dead servo,
`read_all` reported five positions as "Current robot positions" and `store_pose`
persisted those five under a name, which every later `load_pose` then drove
towards while answering "Moved to pose". `validate_pose` checks bounds rather
than arity, so nothing downstream could notice.

`emergency_stop` had already settled the disposition for a partial result over
the configured motor set: it answers `status="error"` and names the joints plus
what that means, because a caller told the whole arm was released when part of it
is still driven is worse than no stop at all. `read_all` now answers the same way
and still carries the positions that did arrive, which are what a caller
diagnosing a dead servo needs. `store_pose` refuses instead, because that one
persists - `incremental_move` refuses on an unreadable position for the same
reason, and a stored pose misrepresents itself on every load rather than once.

The gap is derived by one helper shared by both readers, the same way
`_smooth_move` derives its own: compare what came back against what was
expected. A bus where nothing answers keeps its own diagnosis, and every motor is
still attempted - the reported outcome changes, the packets do not.

The two buses already pinned were the extremes, all servos answering and none
answering, and those are exactly the two on which a partial-drop guard and its
absence are indistinguishable. The mixed bus is now covered.
