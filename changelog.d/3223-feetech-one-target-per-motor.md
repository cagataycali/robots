### Fixed: a Feetech motor spelled twice is refused rather than collapsed to one target

`FeetechDriver.send_action` reduced every action key to the motor it names with
`{str(key).removesuffix(".pos"): value}`. A mapping carrying both `"gripper"`
and `"gripper.pos"` therefore became a single entry: insertion order decided
which of the two targets survived, one motor went onto the SYNC_WRITE frame
instead of two commands, and the envelope returned `success` naming only the
survivor. `{"gripper": 0.0, "gripper.pos": 100.0}` opened the gripper fully;
the same pair in the other order closed it.

Both spellings are first class in this one driver, which is what makes the
mixed mapping reachable rather than hypothetical.
`strands_robots.bus_access.read_joints` takes its `bus.sync_read` fast path and
returns `"<motor>.pos"` keys, while the tool schema (`"joint name -> degrees"`)
and the success envelope's `commanded` block both speak the bare name -- so a
caller that reads the arm, edits one joint by the name the envelope reported,
and sends the result back holds both spellings of that motor.

The reduction now has one owner. It refuses a doubled motor before anything is
written, naming every motor spelled more than once and the keys that spell it,
so a caller fixing one collision does not have to discover the next. The
refusal does not depend on the two values agreeing: a rule that fires only when
the targets differ is a rule an agent cannot plan against, and one motor takes
one target either way. Every single-spelling path is unchanged, including the
read-modify-send round trip that produces a mixed mapping in the first place.
