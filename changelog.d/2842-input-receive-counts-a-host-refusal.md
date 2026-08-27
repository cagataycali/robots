### Fixed: a teleop frame the follower refused is counted, not reported applied

A host reports "I did not apply this" in two shapes, and only one of them is an
exception. `HardwareRobot.send_action` catches every exception and returns
`{"status": "error"}` - its docstring gives the reason, "so the teleop loop can
count errors without exceptions tearing down the hot loop" - and a simulation
host answers the same way for an action key it cannot resolve to an actuator, or
for a `robot_name` that is not in the world. `InputReceiver._on_input` read only
the exception, so the shape a host is *designed* to produce went uncounted. Both
apply routes dropped the verdict on the way out:
`InputReceiver._default_apply` discarded what
`bus_access.write_action` had carried out for it ("Returns: Whatever the driver's
`send_action()` returns"), and the closure `start_teleop_receive` installs for a
simulation host discarded it too.

Driving 30 frames into a MuJoCo `so101` through that closure, at the receiver's
own nominal rate:

| action keys | joints reached | frames_received | errors |
|---|---|---|---|
| `1` / `2` / `3` | 0.098 (driven) | 30 | 0 |
| `shoulder_pan.pos` / ... | 0.012 (sagged) | 30 | 0 |

Both rows commanded 1.5. The second is a real SO-101 leader's own key spelling
reaching a follower that names its joints `1`..`6` - the mismatch
`attach_teleop`'s `map_fn` exists to bridge on the local path, for which the mesh
path has no equivalent - so every frame was refused and the arm only sagged under
gravity. The two rows were identical in the report: 30 frames at 49 Hz, zero
errors, zero rejected, zero rate-dropped, zero slew-rejected. Nothing in `stats`
separated a stream the follower applied in full from one it refused in full.

The local teleop loop already read that envelope, and
`TeleopMixin._teleop_stats` writes the vocabulary down: "soft: `send_action`
returns `{"status": "error"}` -> errors += 1 AND frames += 1 (an unpowered
follower gives errors == frames)". Its comment ends by naming the outcome the
derivation exists to refuse - "0 frames, 0 errors and 'success': a silent no-op"
- which is what the receive path reported. `_teleop_loop`'s own comment claims
the two paths judge a leader frame identically, "whether it reaches a follower
over the network or on this host"; that held for the per-joint slew bound and not
for the outcome. Both routes now carry the verdict out and `_on_input` counts it
the way the local loop does, so a follower that refuses everything reads as
`errors == frames_received`.

Deliberately unchanged, and pinned: the frame still counts in `frames_received`
(it was delivered and attempted, exactly as the local path counts it), the slew
baseline still advances, and `rejected` is untouched - that total names a guard on
the receiving side which refused the frame and never applied it. The check reads
`== "error"` rather than `!= "success"`, because the envelope vocabulary here is
wider than those two (`ok`, `running`, `idle`, `timeout` are all statuses this
package returns) and the local loop reads `== "error"` too.
