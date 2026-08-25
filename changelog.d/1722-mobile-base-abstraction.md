### Changed: one drive contract for every mobile base - `MobileBaseRobot` + `Transport`

`RosBridgedRobot` and `RtpsRobot` were written by mirroring each other, so most
of each class was the same code: name validation, the `drive`/`stop` contract,
the `tools` property with suffix mangling, the `from_<preset>()` idiom. The
duplication was not free, and its cost is a matter of record: the velocity and
navigation-goal guards had to be written into both classes separately, once per
transport, to close the same defect in each - `drive(linear=float("nan"))`
publishing NaN onto `cmd_vel` (a `min`/`max` clamp passes `nan` through
silently), `drive(duration=float("inf"))` becoming an unbounded publish loop.
Every future mobile base would have paid that tax again, and the third one to
be added would have been the first to be forgotten.

Those guards now exist once. The base validates a drive request against the same
shared numeric domain the rest of the package uses, rather than restating the
rule, so a velocity clamp cannot start accepting a value that a control-loop
frequency rejects. A regression test asserts the delegation itself, not just its
current verdicts.

`MobileBaseRobot` now owns the invariant half - validation, the drive contract
and its safety semantics, the `init_services` enable handshake, `get_pose` /
`get_scan`, and the `tools` property. A robot class supplies only what varies:
a `Transport` (how bytes move) and, when the platform is not differential-drive,
a `_cmd_fields` override (what the command message looks like - the kinematics
seam). `Transport` requires `publish` and `echo` only; `service_call` and
`action_send_goal` are separate optional-capability protocols, because
`use_rtps` has neither and a protocol forcing it to declare them would make it
lie. The base asks (`robot.supports(...)`) and reflects the answer: an
`init_services` handshake wired onto a transport that cannot call services is
refused at construction rather than on the track, and tools are built from what
is actually wired, so an agent is never handed one that can only answer "not
configured".

`RosBridgedRobot` inherits the hardened contract above. The `stop_<node>` agent
tool it was missing - an agent could start motion and had to infer `drive(0, 0)`
to end it - is now emitted by the shared base for every transport, so the gap
cannot reopen for one bridge while the others have it.

The operator-approval gate is consolidated the same way. Commanding a robot over
`use_ros` needs an operator decision, so the command tools are declared
`@tool(context=True)` and the context is threaded from the tool through the base
to the transport, which is the single place it is handed to `use_ros`. Whether a
transport forwards it is derived from its own tool's signature rather than from a
list: a gating tool that is not forwarded to turns the whole command surface into
a per-call refusal, and a non-gating tool that is forwarded to raises. Both
directions are now failing tests. The structural guard that pins the wiring reads
the classes in the bridge's MRO instead of one file, so it follows the tools to
wherever they are declared - a guard that scanned only the bridge would have
stopped covering `drive` and `stop` the moment they moved to the base. The
outcome cases keep the real gate and stand the DDS backend down beneath it,
resolving only the interface the robot declares: a resolver that answered every
type string would hand back a velocity message whatever `cmd_vel_type` said, so
a robot declaring an interface with nowhere to put `linear.x` would still read as
having driven.

Speed and duration limits stay unset by default on both ported classes - neither
knows the limits of the third-party robot it drives, and inventing one would
silently cap an existing caller. A limit left `None` means "this platform
declares no limit", never zero.
