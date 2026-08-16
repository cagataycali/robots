### Fixed: `robot_state_keys` is validated as a list of distinct joint names on every surface

`robot_state_keys` is the ordered list of joint/motor names a policy emits as its
action-dict keys - the names `send_action` resolves - so it decides which
actuator each action value is sent to. Nine of the eleven surfaces that bind it
validated nothing about its shape.

Its two sibling setters on `Policy` did. `set_control_frequency` and
`set_rtc_observed_delay` are concrete on the base class and each raises through a
shared domain in `strands_robots.utils`; `set_robot_state_keys` is
`@abstractmethod` with a `pass` body, so there was no shared implementation to
carry one and each provider re-implemented the setter without it. On one policy
instance, `set_control_frequency(True)` and `set_rtc_observed_delay(1.5)` raised
while `set_robot_state_keys("wrist")` and `set_robot_state_keys([1])` were
accepted.

A single joint name passed as a bare string is the mistake that matters:
`set_robot_state_keys("shoulder_pan.pos")` bound 16 per-character joints, and
`get_actions` then emitted 13-key action dicts - the width silently shrinking
from 16 to 13 as the repeated characters collapsed in the dict - so a robot would
have been commanded on keys named `'s'`, `'h'`, `'o'` and `'.'`. A repeated name
was bound at width 3 and emitted at width 2, the same collapse, which also
narrows `lerobot_async`'s `{key: float for key in self.robot_state_keys}`
hardware-feature map so it declares fewer columns than `align_action_values` is
handed. A `Mapping` was accepted with its values silently discarded, a one-shot
iterator was bound and exhausted by the first read, and non-string or blank
entries were bound as given.

Those nine surfaces now resolve the value through
`strands_robots.utils.name_list_error`, the shared domain that already governed
the policy `image_keys` parameters and the simulation `cameras` subset, so a
mistake reports the same way wherever it lands and names the parameter. The two
delegating wrappers (`CompositePolicy`, `PersistentPolicy`) forward to the policy
that binds, so the verdict is identical through them.

`WBCPolicy` and `MotionBricksPolicy` are unchanged: they resolve every G1 joint
they drive by name inside the caller's list, so all five malformed shapes already
failed that membership check with a message naming the missing joints. They also
tolerate a repeated name on purpose - it resolves to its first occurrence, which
`test_flat_state_name_resolved_first_occurrence_wins` pins - and wiring them to
the shared domain would have turned that reviewed decision into a refusal. Their
existing totality is now pinned behaviourally so the exemption cannot drift into
a silent accept.

One coercion had to go with the guards. `PolicyServer._dispatch` handled
`MSG_SET_STATE_KEYS` as `set_robot_state_keys(list(message.get("keys", [])))`,
and `list("wrist")` is `['w', 'r', 'i', 's', 't']` - five *distinct, non-blank*
names that pass every shape check. The coercion therefore laundered a mis-typed
parameter into a well-formed joint list that no guard could recognise. It now
forwards the wire value verbatim, for the reason the neighbouring `hz` handler
already documents: coercing on the server lets the wire through a value the
in-process API refuses, and the policy owns the domain. `RemotePolicy` validates
ahead of its own `list(...)` for the same reason on the outbound side.

`robot_state_keys=None` and an empty list keep their existing "auto-detect"
meaning: like every other consumer of the shared domain, the check is gated on a
truthy value. Only inputs that previously bound a layout the caller never wrote
now raise, so no working caller changes behaviour.
